#!/usr/bin/env python3
# The Triton kernels in this project were developed with the assistance of
# Claude Opus and Claude Fable (Anthropic).


import argparse
import datetime
import json
import math
import os
import subprocess
import sys

try:
    import torch
except ImportError:
    torch = None

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

DEVICE = "cuda"
SEED = 42
S = 512
B_TRAIN = 1024
CHUNK_SAMPLES = 128
TRAIN_PATHS = ["torch_fp32_tf32", "torch_autocast_bf16", "triton_persist", "triton_chunked",
               "triton_improved_persist", "triton_improved_chunked"]

CONFIGS = [
    {"name": "contentwise", "D": 64, "N": 2431, "I_train": 32},
    {"name": "stockholm", "D": 512, "N": 1046, "I_train": 466},
]
REGIMES = ["dense", "padded"]


def make_valid(B, I, regime, device):
    if regime == "dense":
        return None
    lo = math.ceil(0.6 * I)
    return torch.randint(lo, I + 1, (B,), device=device, dtype=torch.int32)


def make_train_data(cfg, regime):
    D, N, I = cfg["D"], cfg["N"], cfg["I_train"]
    torch.manual_seed(SEED)
    mu = torch.randn(B_TRAIN, D, device=DEVICE, requires_grad=True)
    log_sigma = (0.1 * torch.randn(B_TRAIN, D, device=DEVICE)).requires_grad_(True)
    table = torch.randn(N, D, device=DEVICE, requires_grad=True)
    option_id = torch.randint(0, N, (B_TRAIN, I), device=DEVICE)
    valid = make_valid(B_TRAIN, I, regime, DEVICE)
    vc = valid.float() if valid is not None else torch.full((B_TRAIN,), float(I), device=DEVICE)
    tgt = torch.floor(torch.rand(B_TRAIN, device=DEVICE) * vc).long().clamp_(max=I - 1)
    coef = torch.randn(B_TRAIN, S, device=DEVICE)
    pad_mask = None
    if valid is not None:
        pad_mask = torch.arange(I, device=DEVICE)[None, :] >= valid[:, None]
    return mu, log_sigma, table, option_id, valid, pad_mask, tgt, coef


def torch_train_core(mu, log_sigma, table, option_id, tgt, eps):
    x = mu[:, None, :] + torch.exp(log_sigma)[:, None, :] * eps
    V = table[option_id]
    z = torch.bmm(x, V.transpose(1, 2))
    zt = z[torch.arange(z.shape[0], device=z.device), :, tgt]
    return zt - torch.logsumexp(z, dim=-1)


def torch_train_core_masked(mu, log_sigma, table, option_id, tgt, eps, pad_mask):
    x = mu[:, None, :] + torch.exp(log_sigma)[:, None, :] * eps
    V = table[option_id]
    z = torch.bmm(x, V.transpose(1, 2))
    z = z.masked_fill(pad_mask[:, None, :], float("-inf"))
    zt = z[torch.arange(z.shape[0], device=z.device), :, tgt]
    return zt - torch.logsumexp(z, dim=-1)


def fresh_compile(fn):
    torch._dynamo.reset()
    return torch.compile(fn, dynamic=False, mode="max-autotune")


WARMUP = 5
ITERS = 100
MEM_ITERS = 3


def time_fn(fn, iters=ITERS, warmup=WARMUP):
    import statistics
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {
        "median_ms": statistics.median(ts),
        "mean_ms": statistics.fmean(ts),
        "std_ms": statistics.stdev(ts) if len(ts) > 1 else 0.0,
        "p10_ms": ts[int(0.10 * len(ts))],
        "p90_ms": ts[int(0.90 * len(ts))],
        "iters": iters,
    }


def peak_mem_mb(fn, iters=MEM_ITERS):
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1024**2


def measure(fn):
    stats = time_fn(fn)
    stats["peak_mem_mb"] = peak_mem_mb(fn)
    return stats


def bench_training_cell(cfg, regime):
    from src.models.fused_mxl_triton import triton_mxl_logprobs
    from src.models.fused_mxl_triton_improved import triton_mxl_logprobs as triton_improved_logprobs
    mu, log_sigma, table, option_id, valid, pad_mask, tgt, coef = make_train_data(cfg, regime)
    md, ld, td = mu.detach(), log_sigma.detach(), table.detach()

    with torch.no_grad():
        prev_prec = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        eps32 = torch.randn(B_TRAIN, S, cfg["D"], device=DEVICE)
        lg_ref = (torch_train_core_masked(md, ld, td, option_id, tgt, eps32, pad_mask)
                  if pad_mask is not None else
                  torch_train_core(md, ld, td, option_id, tgt, eps32))
        lg_tri = triton_mxl_logprobs(md, ld, td, option_id, tgt, S, valid_count=valid,
                                     eps=eps32, precision="ieee")
        lg_improved = triton_improved_logprobs(md, ld, td, option_id, tgt, S, valid_count=valid,
                                   eps=eps32, precision="ieee")
        diff = (lg_ref - lg_tri).abs().max().item()
        diff_improved = (lg_ref - lg_improved).abs().max().item()
        torch.set_float32_matmul_precision(prev_prec)
        assert diff < 1e-2, f"training parity failed ({cfg['name']}/{regime}): {diff}"
        assert diff_improved < 1e-2, f"improved-kernel training parity failed ({cfg['name']}/{regime}): {diff_improved}"
        del eps32, lg_ref, lg_tri, lg_improved
    torch.cuda.empty_cache()

    core = fresh_compile(torch_train_core_masked if pad_mask is not None else torch_train_core)

    def torch_step_fp32():
        eps = torch.randn(B_TRAIN, S, cfg["D"], device=DEVICE)
        args = (mu, log_sigma, table, option_id, tgt, eps)
        lg = core(*args, pad_mask) if pad_mask is not None else core(*args)
        torch.autograd.grad((coef * lg).sum(), (mu, log_sigma, table))

    def torch_step_autocast():
        eps = torch.randn(B_TRAIN, S, cfg["D"], device=DEVICE)
        args = (mu, log_sigma, table, option_id, tgt, eps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = core(*args, pad_mask) if pad_mask is not None else core(*args)
        torch.autograd.grad((coef * lg.float()).sum(), (mu, log_sigma, table))

    def tri_persist_step():
        lg = triton_mxl_logprobs(mu, log_sigma, table, option_id, tgt, S, valid_count=valid)
        torch.autograd.grad((coef * lg).sum(), (mu, log_sigma, table))

    def tri_chunked_step():
        lg = triton_mxl_logprobs(mu, log_sigma, table, option_id, tgt, S, valid_count=valid,
                                 chunk_samples=CHUNK_SAMPLES)
        torch.autograd.grad((coef * lg).sum(), (mu, log_sigma, table))

    def tri_improved_persist_step():
        lg = triton_improved_logprobs(mu, log_sigma, table, option_id, tgt, S, valid_count=valid)
        torch.autograd.grad((coef * lg).sum(), (mu, log_sigma, table))

    def tri_improved_chunked_step():
        lg = triton_improved_logprobs(mu, log_sigma, table, option_id, tgt, S, valid_count=valid,
                                chunk_samples=CHUNK_SAMPLES)
        torch.autograd.grad((coef * lg).sum(), (mu, log_sigma, table))

    cell = {
        "torch_fp32_tf32": measure(torch_step_fp32),
        "torch_autocast_bf16": measure(torch_step_autocast),
        "triton_persist": measure(tri_persist_step),
        "triton_chunked": measure(tri_chunked_step),
        "triton_improved_persist": measure(tri_improved_persist_step),
        "triton_improved_chunked": measure(tri_improved_chunked_step),
        "parity_max_diff": diff,
        "parity_max_diff_improved": diff_improved,
    }
    torch.cuda.empty_cache()
    return cell


def fmt_cell(s):
    return f"{s['median_ms']:8.2f} ±{s['std_ms']:5.2f}  {s['peak_mem_mb']:9.1f}"


def print_table(title, results, paths):
    width = 23 + 28 * len(paths) + 30
    print(f"\n{'=' * width}\n{title}   (per path: median ms ± std | marginal peak VRAM MB)\n{'=' * width}")
    header = f"{'config':<14}{'regime':<9}" + "".join(f"{p:<28}" for p in paths)
    print(header)
    print("-" * width)
    for cfg in CONFIGS:
        for regime in REGIMES:
            cell = results[cfg["name"]][regime]
            row = f"{cfg['name']:<14}{regime:<9}" + "".join(f"{fmt_cell(cell[p]):<28}" for p in paths)
            print(row + f"   (parity {cell['parity_max_diff']:.1e} / improved "
                        f"{cell['parity_max_diff_improved']:.1e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/benchmark_results")
    args = parser.parse_args()

    assert torch is not None and torch.cuda.is_available(), "paper benchmark needs a GPU"
    torch.set_float32_matmul_precision("high")
    import triton
    gpu = torch.cuda.get_device_name(0)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                            capture_output=True, text=True).stdout.strip()
    print(f"GPU: {gpu} | torch {torch.__version__} | triton {triton.__version__} | commit {commit[:12]}")
    print(f"Training: B={B_TRAIN}, S={S} | "
          f"chunk_samples={CHUNK_SAMPLES} | seed={SEED} | {ITERS} iters")

    training = {}
    for cfg in CONFIGS:
        training[cfg["name"]] = {}
        for regime in REGIMES:
            print(f"\n--- training  {cfg['name']}/{regime} (I={cfg['I_train']}, D={cfg['D']}) ---")
            training[cfg["name"]][regime] = bench_training_cell(cfg, regime)

    print_table("TRAINING (fwd+bwd, vanilla NLL, B=1024)", training,
                TRAIN_PATHS)

    out = {
        "meta": {
            "gpu": gpu, "torch": torch.__version__, "triton": triton.__version__,
            "git_commit": commit, "seed": SEED, "iters": ITERS, "warmup": WARMUP,
            "S": S, "B_train": B_TRAIN,
            "chunk_samples": CHUNK_SAMPLES, "configs": CONFIGS, "regimes": REGIMES,
            "timestamp": datetime.datetime.now().isoformat(),
            "notes": "eps generation inside timing for all paths; fresh max-autotune "
                     "compile per cell; parity ieee<1e-2; torch timed at fp32/TF32 (the "
                     "original table) and under autocast(bfloat16), Triton at precision=bf16 "
                     "(fp32 accumulate); triton_improved_* columns are "
                     "post-acceptance work (fused_mxl_triton_improved), not in the paper",
        },
        "training": training,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(args.output_dir, f"paper_mxl_kernel_benchmark_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
