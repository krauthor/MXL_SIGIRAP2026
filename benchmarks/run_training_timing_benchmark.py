import sys
import os
import argparse
import gc
import time
import json
import itertools
import socket
import subprocess
from datetime import datetime

import numpy as np
import torch
import torch.utils.data as torch_data
import yaml

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.data.load_data import load_data
from src.data.dataset import ChoiceDataset, choice_collate
from src.models.create_model import create_model
from src.pipelines.unified_training import compute_forward_loss, move_batch_to_device, MXL_MODELS

from torch._dynamo.utils import counters as dynamo_counters

NO_CUDAGRAPH_MODE = "max-autotune-no-cudagraphs"
CONFIG_PATH = "benchmarks/training_timing_benchmark_config.yaml"
COMPILE_MODE = "max-autotune"


SEED = 8

def get_node_info():
    node_name = os.environ.get("SLURM_NODELIST", socket.gethostname())
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True
        ).strip()
    except Exception:
        gpu_info = "N/A"
    return {"node": node_name, "gpu_info": gpu_info}


def expand_mxl_sweep(sweep_config):
    mxl_defaults = {k: v for k, v in sweep_config.items() if k != "cov_configs"}
    num_samples = mxl_defaults["num_prior_samples"]
    configs = {}
    for cov_entry in sweep_config["cov_configs"]:
        cov_type = cov_entry["cov_type"]
        l2_weight = cov_entry["l2_weight"]
        for cov_rank in cov_entry["cov_rank"]:
            if cov_type == "diag":
                label = f"MXL_diag_l2{l2_weight}_s{num_samples}"
            else:
                label = f"MXL_{cov_type}_r{cov_rank}_s{num_samples}"
            configs[label] = {
                **mxl_defaults,
                "model_name": "MXL",
                "cov_type": cov_type,
                "cov_rank": cov_rank,
                "l2_weight": l2_weight,
            }
    return configs


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


NUM_STAGED_BATCHES = 8


def stage_batches(train_data_loader, device, n=NUM_STAGED_BATCHES):
    it = iter(train_data_loader)
    staged = []
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        staged.append(move_batch_to_device(next(it), device))
    _sync(device)
    return staged, (time.perf_counter() - t0) / n


def benchmark_model(label, hyperparams, staged_batches, num_users, num_items,
                    no_choice_id, padding_id, never_chosen_id,
                    seed, num_warmup_batches, num_timed_batches, device):
    model_name = hyperparams["model_name"]
    print(f"\n{'='*60}")
    print(f"Benchmarking: {label}")
    print(f"  model_name={model_name}, cov_type={hyperparams.get('cov_type', 'N/A')}, "
          f"cov_rank={hyperparams.get('cov_rank', 'N/A')}, "
          f"num_prior_samples={hyperparams.get('num_prior_samples', 'N/A')}, "
          f"l2_weight={hyperparams.get('l2_weight', 'N/A')}")
    print(f"{'='*60}")

    l2_weight = hyperparams.get("l2_weight", 0)
    beta = hyperparams.get("beta_max", 0.1) if model_name in MXL_MODELS else 0.0

    def build():
        torch.manual_seed(seed)
        model = create_model(
            model_name=model_name,
            num_users=num_users,
            num_items=num_items,
            hyperparams=hyperparams,
            no_choice_id=no_choice_id,
            padding_id=padding_id,
            never_chosen_id=never_chosen_id,
        )
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
        batch_cycle = itertools.cycle(staged_batches)

        def run_step():
            data_input, labels = next(batch_cycle)
            optimizer.zero_grad()
            loss, _ = compute_forward_loss(
                model, model_name, data_input, labels["choice_pos_onehot"],
                beta=beta, l2_weight=l2_weight, kl_over_batch=True,
            )
            loss.backward()
            optimizer.step()
            return loss

        return model, optimizer, run_step

    def teardown(model, optimizer):
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        torch._dynamo.reset()

    torch._dynamo.reset()
    dynamo_counters.clear()

    model, optimizer, run_step = build()

    used_mode = COMPILE_MODE
    oom = None
    print(f"  Compiling (backend=inductor, mode={used_mode}, dynamic=False)...")
    t0 = time.perf_counter()
    model.compile(backend="inductor", mode=used_mode, dynamic=False)
    try:
        for _ in range(num_warmup_batches):
            run_step()
        _sync(device)
    except torch.cuda.OutOfMemoryError:
        print(f"  Warning: compile mode '{used_mode}' hit CUDA OOM. "
              f"Rebuilding and retrying with '{NO_CUDAGRAPH_MODE}'.")
        teardown(model, optimizer)
        used_mode = NO_CUDAGRAPH_MODE
        dynamo_counters.clear()
        model, optimizer, run_step = build()
        model.compile(backend="inductor", mode=used_mode, dynamic=False)
        try:
            for _ in range(num_warmup_batches):
                run_step()
            _sync(device)
        except torch.cuda.OutOfMemoryError as e2:
            oom = str(e2).split("\n")[0]
    compile_warmup_sec = time.perf_counter() - t0
    if oom is None:
        print(f"  Compile + warmup done in {compile_warmup_sec:.1f}s (mode={used_mode}).")

    is_mxl = model_name in MXL_MODELS
    meta = {
        "model_name": model_name,
        "cov_type": hyperparams.get("cov_type") if is_mxl else None,
        "cov_rank": hyperparams.get("cov_rank") if is_mxl else None,
        "num_prior_samples": hyperparams.get("num_prior_samples") if is_mxl else None,
        "l2_weight": hyperparams.get("l2_weight", 0),
        "out_dim": hyperparams.get("out_dim"),
    }

    if oom is not None:
        print(f"  OOM: {oom}")
        teardown(model, optimizer)
        return {
            **meta,
            "oom": True,
            "error": oom,
            "batch_times_sec": [],
            "mean_batch_sec": None,
            "std_batch_sec": None,
            "median_batch_sec": None,
            "num_timed_batches": 0,
            "num_warmup_batches": num_warmup_batches,
            "compiled": True,
            "compile_backend": "inductor",
            "compile_mode": used_mode,
            "compile_warmup_sec": compile_warmup_sec,
            "dynamo_frames_ok": 0,
            "ran_compiled": False,
        }

    print(f"  Timing: {num_timed_batches} batches...")
    batch_times = []
    for i in range(num_timed_batches):
        _sync(device)
        t0 = time.perf_counter()
        run_step()
        _sync(device)
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)

    mean_time = np.mean(batch_times)
    std_time = np.std(batch_times)
    median_time = np.median(batch_times)

    frames_ok = int(dynamo_counters["frames"].get("ok", 0))
    ran_compiled = frames_ok > 0
    print(f"  Result: mean={mean_time:.6f}s, std={std_time:.6f}s, median={median_time:.6f}s"
          f"  [dynamo frames ok={frames_ok}]")
    if not ran_compiled:
        print(f"  Warning: {label} requested compile mode '{used_mode}' but dynamo "
              f"compiled no frames — these are EAGER timings.")

    return {
        **meta,
        "oom": False,
        "dynamo_frames_ok": frames_ok,
        "ran_compiled": ran_compiled,
        "batch_times_sec": batch_times,
        "mean_batch_sec": float(mean_time),
        "std_batch_sec": float(std_time),
        "median_batch_sec": float(median_time),
        "num_timed_batches": num_timed_batches,
        "num_warmup_batches": num_warmup_batches,
        "kl_scope": "batch",
        "compiled": True,
        "compile_backend": "inductor",
        "compile_mode": used_mode,
        "compile_warmup_sec": compile_warmup_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Training timing benchmark for MNL/MXL models.")
    parser.add_argument("--output_dir", type=str, default="data/benchmark_results",
                        help="Output directory for the results JSON.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Warning: No CUDA device available, timings will not be comparable.")

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    shared = config.get("shared", {})
    num_warmup_batches = shared.get("num_warmup_batches", 20)
    num_timed_batches = shared.get("num_timed_batches", 50)

    all_configs = {}

    for label, model_cfg in config["models"].items():
        all_configs[label] = {**shared, **model_cfg}

    for label, model_cfg in expand_mxl_sweep(config["mxl_sweep"]).items():
        all_configs[label] = {**shared, **model_cfg}

    print(f"Total configs to benchmark: {len(all_configs)}")
    for label in all_configs:
        print(f"  - {label}")

    print("\nLoading dataset...")
    data = load_data(
        dataset_name=shared.get("dataset_name", "ContentWise_LastActionSplit"),
        seed=42,
    )

    num_users = int(data["max_user_id"]) + 1
    num_items = int(data["max_item_id"]) + 1
    no_choice_id = data["no_choice_id"]
    padding_id = data["padding_id"]
    never_chosen_id = data["never_chosen_id"]

    train_data_loader = torch_data.DataLoader(
        dataset=ChoiceDataset(data["train"]),
        batch_size=shared.get("batch_size", 128),
        shuffle=True,
        collate_fn=choice_collate,
        drop_last=True,
        num_workers=0,
    )
    print(f"Dataset loaded: {num_users} users, {num_items} items, "
          f"{len(train_data_loader)} batches/epoch")

    staged_batches, data_sec = stage_batches(train_data_loader, device)
    print(f"Staged {len(staged_batches)} batches on {device} "
          f"({data_sec*1000:.2f} ms/batch of host collation + H2D; excluded from timings)")

    results = {}
    for label, hp in all_configs.items():
        result = benchmark_model(
            label=label,
            hyperparams=hp,
            staged_batches=staged_batches,
            num_users=num_users,
            num_items=num_items,
            no_choice_id=no_choice_id,
            padding_id=padding_id,
            never_chosen_id=never_chosen_id,
            seed=SEED,
            num_warmup_batches=num_warmup_batches,
            num_timed_batches=num_timed_batches,
            device=device,
        )
        results[label] = result

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"training_timing_benchmark_seed{SEED}_{timestamp}.json")

    output = {
        "seed": SEED,
        "timestamp": timestamp,
        "config_path": CONFIG_PATH,
        "dataset_name": shared.get("dataset_name", "ContentWise_LastActionSplit"),
        "out_dim": shared.get("out_dim"),
        "batch_size": shared.get("batch_size"),
        "num_warmup_batches": num_warmup_batches,
        "num_timed_batches": num_timed_batches,
        "num_users": num_users,
        "num_items": num_items,
        "num_batches_per_epoch": len(train_data_loader),
        "device": device.type,
        "kl_scope": "batch",
        "num_staged_batches": len(staged_batches),
        "data_loading_sec_per_batch": data_sec,
        "compiled": True,
        "compile_backend": "inductor",
        "compile_mode": COMPILE_MODE,
        "node_info": get_node_info(),
        "results": results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    print(f"\n{'='*92}")
    print(f"{'Label':<35} {'Mean (s)':>10} {'Std (s)':>10} {'Median (s)':>10} {'Status':>22}")
    print(f"{'-'*92}")
    for label, r in results.items():
        if r.get("oom"):
            print(f"{label:<35} {'OOM':>10} {'-':>10} {'-':>10} {'out of memory':>22}")
            continue
        status = r["compile_mode"] if r["ran_compiled"] else "EAGER FALLBACK"
        print(f"{label:<35} {r['mean_batch_sec']:>10.6f} {r['std_batch_sec']:>10.6f} "
              f"{r['median_batch_sec']:>10.6f} {status:>22}")
    print(f"{'='*92}")

    n_oom = sum(1 for r in results.values() if r.get("oom"))
    n_eager = sum(1 for r in results.values() if not r["oom"] and not r["ran_compiled"])
    print(f"{len(results)} configs: {n_oom} out of memory, {n_eager} fell back to eager.")
    if n_eager:
        print("Warning: eager fallbacks are not comparable to the compiled timings.")


if __name__ == "__main__":
    main()
