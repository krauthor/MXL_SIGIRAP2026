import sys
import os
import argparse
import json
import time
import socket
import subprocess
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.models.create_model import create_model
from src.utils.slate_generation import (
    generate_slate_top_k_torch,
    generate_slate_greedy_batched_torch,
    generate_slate_topk_avg_choice_prob_torch,
    generate_slate_top1_per_sample_torch,
    generate_slate_top_k_greedy_torch,
)

METHOD_NAMES = {
    "direct_topk": "Top-k MNL",
    "top1_per_sample": "Pref. Top-1",
    "mean_topk": "Top-k MXL",
    "greedy": "Greedy",
    "topk_greedy": "Top-k Greedy",
}

NO_CHOICE_ID = 0


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


def _setup_model(model_name, n_items, d, num_prior_samples, device):
    hyperparams = {
        "out_dim": d,
        "num_prior_samples": num_prior_samples,
    }
    if model_name != "MNL":
        hyperparams["cov_type"] = "diag"

    model = create_model(
        model_name, hyperparams,
        num_users=1, num_items=n_items,
        padding_id=None, no_choice_id=None, never_chosen_id=None,
    )
    model.to(device)
    model.eval()
    return model


def setup_mnl_model(n_items, d, device):
    return _setup_model("MNL", n_items, d, 1, device)


def setup_mxl_model(n_items, d, S, device):
    return _setup_model("MXL", n_items, d, S, device)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_fn_torch(fn, n_repeats, device):
    times = []
    for _ in range(n_repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


SELECT_COMPILE_MODE = "max-autotune"


def run_benchmark(n_items=1000, d=512, k=20, S=512,
                  n_warmup=20, n_repeats=10, seed=42, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compile_kw = dict(dynamic=False, mode=SELECT_COMPILE_MODE)
    c_top_k = torch.compile(generate_slate_top_k_torch, **compile_kw)
    c_top1 = torch.compile(generate_slate_top1_per_sample_torch, **compile_kw)
    c_avg_prob = torch.compile(generate_slate_topk_avg_choice_prob_torch, **compile_kw)
    c_greedy = torch.compile(generate_slate_greedy_batched_torch, **compile_kw)
    c_top_k_greedy = torch.compile(generate_slate_top_k_greedy_torch, **compile_kw)

    print("Setting up MNL model...")
    mnl_model = setup_mnl_model(n_items, d, device)
    print("Setting up MXL model...")
    mxl_model = setup_mxl_model(n_items, d, S, device)

    input_data = {
        "user_ids": torch.zeros(1, dtype=torch.long, device=device),
        "option_id": torch.arange(n_items, dtype=torch.long, device=device).reshape(1, -1),
    }

    def mnl_score():
        with torch.no_grad():
            utils, _ = mnl_model(input_data)
        return utils[0, :, 0]

    def mxl_score():
        with torch.no_grad():
            utils, _ = mxl_model(input_data)
        return utils[0]

    @torch.compile(**compile_kw)
    def compute_mean_scores(utils):
        return torch.softmax(utils, dim=0).mean(dim=1)

    print(f"Warming up scoring ({n_warmup} iterations)...")
    for _ in range(n_warmup):
        mnl_score()
        mxl_score()
    _sync(device)

    print(f"Warming up selection ({n_warmup} iterations)...")
    warmup_scores_mnl = mnl_score()
    warmup_utils = mxl_score()
    warmup_mean_scores = compute_mean_scores(warmup_utils).clone()
    gen = torch.Generator(device=device).manual_seed(seed)

    for _ in range(n_warmup):
        c_top_k(warmup_scores_mnl, k, include_no_choice=False,
                                   no_choice_id=NO_CHOICE_ID)
        c_top1(warmup_utils, k, include_no_choice=False,
                                             no_choice_id=NO_CHOICE_ID, generator=gen)
        c_avg_prob(warmup_utils, k, include_no_choice=False,
                                                  no_choice_id=NO_CHOICE_ID)
        c_greedy(warmup_utils, k, include_no_choice=False,
                                            no_choice_id=NO_CHOICE_ID)
        c_top_k_greedy(warmup_utils, warmup_mean_scores, k,
                                          include_no_choice=False, no_choice_id=NO_CHOICE_ID)
    _sync(device)

    methods = {}

    def measure(name, description, score_fn, select_fn, total_fn):
        print(f"Timing {name}...")
        score_t = time_fn_torch(score_fn, n_repeats, device)
        select_t = time_fn_torch(select_fn, n_repeats, device)
        total_times = []
        for _ in range(n_repeats):
            _sync(device)
            t0 = time.perf_counter()
            total_fn()
            _sync(device)
            t1 = time.perf_counter()
            total_times.append(t1 - t0)
        methods[name] = {
            "description": description,
            "score_phase_sec": score_t,
            "select_phase_sec": select_t,
            "total_sec": total_times,
        }

    scores_mnl = mnl_score()
    measure(
        "direct_topk",
        "MNL scoring + top-k selection",
        mnl_score,
        lambda: c_top_k(scores_mnl, k, include_no_choice=False,
                                           no_choice_id=NO_CHOICE_ID),
        lambda: c_top_k(mnl_score(), k, include_no_choice=False,
                                           no_choice_id=NO_CHOICE_ID),
    )

    utils_mxl = mxl_score()

    measure(
        "top1_per_sample",
        "MXL sampling + top-1-per-sample selection",
        mxl_score,
        lambda: c_top1(
            utils_mxl, k, include_no_choice=False, no_choice_id=NO_CHOICE_ID, generator=gen),
        lambda: c_top1(
            mxl_score(), k, include_no_choice=False, no_choice_id=NO_CHOICE_ID, generator=gen),
    )

    measure(
        "mean_topk",
        "MXL sampling + mean scores + top-k selection",
        mxl_score,
        lambda: c_avg_prob(
            utils_mxl, k, include_no_choice=False, no_choice_id=NO_CHOICE_ID),
        lambda: c_avg_prob(
            mxl_score(), k, include_no_choice=False, no_choice_id=NO_CHOICE_ID),
    )

    measure(
        "greedy",
        "MXL sampling + full greedy selection",
        mxl_score,
        lambda: c_greedy(
            utils_mxl, k, include_no_choice=False, no_choice_id=NO_CHOICE_ID),
        lambda: c_greedy(
            mxl_score(), k, include_no_choice=False, no_choice_id=NO_CHOICE_ID),
    )

    mean_scores_mxl = compute_mean_scores(utils_mxl).clone()

    def topk_greedy_total():
        u = mxl_score()
        ms = compute_mean_scores(u)
        return c_top_k_greedy(
            u, ms, k, include_no_choice=False, no_choice_id=NO_CHOICE_ID)

    measure(
        "topk_greedy",
        "MXL sampling + top-k init + greedy refinement",
        mxl_score,
        lambda: c_top_k_greedy(
            utils_mxl, mean_scores_mxl, k, include_no_choice=False, no_choice_id=NO_CHOICE_ID),
        topk_greedy_total,
    )

    return {
        "setup": {
            "n_items": n_items,
            "d": d,
            "k": k,
            "S": S,
            "n_warmup": n_warmup,
            "n_repeats": n_repeats,
            "seed": seed,
            "device": device.type,
            "no_choice_id": NO_CHOICE_ID,
            "select_compile_mode": SELECT_COMPILE_MODE,
        },
        "methods": methods,
    }


def print_summary(results):
    setup = results["setup"]
    print(f"\n{'='*90}")
    print(f"Slate Generation Benchmark — I={setup['n_items']}, d={setup['d']}, "
          f"K={setup['k']}, S={setup['S']}, "
          f"repeats={setup['n_repeats']}")
    print(f"{'='*90}")

    k = setup["k"]
    header = f"{'Method':<25} {'Score (ms)':>12} {'Sel/slot (ms)':>14} {'Total (ms)':>12} {'Score %':>8}"
    print(header)
    print("-" * len(header))

    for name, m in results["methods"].items():
        display_name = METHOD_NAMES.get(name, name)
        score_ms = np.mean(m["score_phase_sec"]) * 1000
        select_ms = np.mean(m["select_phase_sec"]) * 1000
        select_per_slot_ms = select_ms / k
        total_ms = np.mean(m["total_sec"]) * 1000
        score_pct = score_ms / (score_ms + select_ms) * 100 if (score_ms + select_ms) > 0 else 0
        print(f"{display_name:<25} {score_ms:>11.3f} {select_per_slot_ms:>13.3f} {total_ms:>11.3f} {score_pct:>7.1f}%")

    print(f"{'='*90}")
    print(f"Times are means in milliseconds. Sel/slot = select phase / K ({k} slots). Score % = score share of score+select.")


def main():
    parser = argparse.ArgumentParser(description="Slate generation timing benchmark.")
    parser.add_argument("--n_items", type=int, default=1000, help="Number of items (default: 1000)")
    parser.add_argument("--d", type=int, default=512, help="Embedding dimension (default: 512)")
    parser.add_argument("--k", type=int, default=20, help="Slate size (default: 20)")
    parser.add_argument("--S", type=int, default=512, help="Number of user samples (default: 512)")
    parser.add_argument("--n_warmup", type=int, default=20, help="Warmup iterations (default: 20)")
    parser.add_argument("--n_repeats", type=int, default=10, help="Timed iterations (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default="data/benchmark_results",
                        help="Output directory for results JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Warning: No CUDA device available, timings will not be comparable.")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results = run_benchmark(
        n_items=args.n_items,
        d=args.d,
        k=args.k,
        S=args.S,
        n_warmup=args.n_warmup,
        n_repeats=args.n_repeats,
        seed=args.seed,
        device=device,
    )
    results["method_names"] = METHOD_NAMES
    results["node_info"] = get_node_info()
    print_summary(results)
    output_path = os.path.join(
        args.output_dir,
        f"slate_generation_benchmark_{timestamp}.json",
    )

    serializable = json.loads(json.dumps(results, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x))

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
