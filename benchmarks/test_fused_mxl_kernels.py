#!/usr/bin/env python3
# The Triton kernels in this project were developed with the assistance of
# Claude Opus and Claude Fable (Anthropic).


import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.fused_mxl_triton import triton_mxl_logprobs

torch.set_float32_matmul_precision("high")


def _grad_rel_err(a, b):
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-8)


def test_triton_backward(B=4, S=128, I=50, D=64, seed=0, logprobs=triton_mxl_logprobs):
    if not torch.cuda.is_available():
        print("  (Triton backward skipped: no CUDA)")
        return
    device = "cuda"
    torch.manual_seed(0)
    mu = torch.randn(B, D, device=device, requires_grad=True)
    log_sigma = (0.1 * torch.randn(B, D, device=device)).requires_grad_(True)
    table = torch.randn(B * I, D, device=device, requires_grad=True)
    ids = torch.arange(B * I, device=device, dtype=torch.int32).reshape(B, I)
    target_idx = torch.randint(0, I, (B,), device=device)
    coef = torch.randn(B, S, device=device)
    eps = torch.randn(B, S, D, device=device)

    def ref_logg(mu_, ls_, T_):
        x = mu_[:, None, :] + torch.exp(ls_)[:, None, :] * eps
        z = torch.matmul(x, T_[ids.long()].transpose(1, 2))
        zt = z[torch.arange(B, device=device), :, target_idx]
        return zt - torch.logsumexp(z, dim=-1)

    lg_tri = logprobs(mu, log_sigma, table, ids, target_idx, S, eps=eps, precision="ieee")
    lg_ref = ref_logg(mu, log_sigma, table)
    fdiff = (lg_tri - lg_ref).abs().max().item()
    print(f"  forward log_g triton-vs-ref max diff = {fdiff:.2e}")
    assert fdiff < 1e-2, f"forward mismatch: {fdiff}"

    leaves = (mu, log_sigma, table)
    g_tri = torch.autograd.grad((coef * lg_tri).sum(), leaves)
    g_ref = torch.autograd.grad((coef * lg_ref).sum(), leaves)
    for nm, a, b in zip(("mu", "log_sigma", "item_table"), g_tri, g_ref):
        rel = _grad_rel_err(a, b)
        print(f"    grad[{nm}] rel err vs reference = {rel:.2e}")
        assert rel < 5e-2, f"grad {nm} mismatch: {rel}"
    print("✓ Triton backward matches PyTorch reference (same eps, bf16 tolerance)")


def test_triton_padded_bias(B=6, S=96, I=37, D=48, seed=0, logprobs=triton_mxl_logprobs):
    if not torch.cuda.is_available():
        print("  (Triton padded/bias test skipped: no CUDA)")
        return
    device = "cuda"
    torch.manual_seed(7)
    mu = torch.randn(B, D, device=device)
    log_sigma = 0.1 * torch.randn(B, D, device=device)
    N = 3 * I // 2
    table = torch.randn(N, D, device=device)
    ids = torch.randint(0, N, (B, I), device=device, dtype=torch.int32)
    coef = torch.randn(B, S, device=device)
    eps = torch.randn(B, S, D, device=device)
    valid = torch.randint(3, I + 1, (B,), device=device, dtype=torch.int32)
    valid[0] = I
    target_idx = (torch.rand(B, device=device) * (valid.float() - 0.5)).long()
    bias_full = 0.5 * torch.randn(B, I, device=device)
    pad = torch.arange(I, device=device)[None, :] >= valid[:, None]

    def ref_logg(mu_, ls_, T_, b_, eps_=eps):
        x = mu_[:, None, :] + torch.exp(ls_)[:, None, :] * eps_
        z = torch.matmul(x, T_[ids.long()].transpose(1, 2))
        if b_ is not None:
            z = z + b_[:, None, :]
        z = z.masked_fill(pad[:, None, :], float("-inf"))
        zt = z[torch.arange(B, device=device), :, target_idx]
        return zt - torch.logsumexp(z, dim=-1)

    for bias in (bias_full, None):
        tag = "bias" if bias is not None else "no-bias"
        leaves = [t.clone().requires_grad_(True) for t in (mu, log_sigma, table)]
        b_leaf = bias.clone().requires_grad_(True) if bias is not None else None
        lg_tri = logprobs(*leaves, ids, target_idx, S, bias=b_leaf, valid_count=valid, eps=eps,
                          precision="ieee")
        lg_ref = ref_logg(*leaves, b_leaf)
        fdiff = (lg_tri - lg_ref).abs().max().item()
        print(f"  [{tag}] forward log_g padded triton-vs-ref max diff = {fdiff:.2e}")
        assert fdiff < 1e-2, f"padded forward mismatch ({tag}): {fdiff}"

        names = ["mu", "log_sigma", "item_table"]
        if bias is not None:
            leaves.append(b_leaf)
            names.append("bias")
        g_tri = torch.autograd.grad((coef * lg_tri).sum(), leaves)
        g_ref = torch.autograd.grad((coef * lg_ref).sum(), leaves)
        for nm, a, b in zip(names, g_tri, g_ref):
            rel = _grad_rel_err(a, b)
            print(f"    [{tag}] grad[{nm}] rel err vs reference = {rel:.2e}")
            assert rel < 5e-2, f"padded grad {nm} mismatch ({tag}): {rel}"
        g_T = g_tri[2]
        used_rows = torch.unique(ids[~pad].long())
        only_pad_rows = torch.ones(N, dtype=torch.bool, device=device)
        only_pad_rows[used_rows] = False
        assert g_T[only_pad_rows].abs().max().item() == 0.0 if only_pad_rows.any() else True, \
            "table rows referenced only by padded slots got gradient"
        if bias is not None:
            assert g_tri[3][pad].abs().max().item() == 0.0, "grad_bias not zero on padded slots"

    eps16 = eps.to(torch.bfloat16)
    lg16 = logprobs(mu, log_sigma, table, ids, target_idx, S, bias=bias_full,
                    valid_count=valid, eps=eps16, precision="ieee")
    lg_ref16 = ref_logg(mu, log_sigma, table, bias_full, eps16.to(torch.float32))
    d16 = (lg16 - lg_ref16).abs().max().item()
    print(f"  [bf16-eps] forward log_g max diff = {d16:.2e}")
    assert d16 < 1e-2, f"bf16-eps forward mismatch: {d16}"
    print("✓ Triton padded/bias path (table-direct, duplicate ids) matches PyTorch reference")


def test_triton_chunked(B=6, S=128, S_c=32, I=37, D=48):
    if not torch.cuda.is_available():
        print("  (Triton chunked test skipped: no CUDA)")
        return
    device = "cuda"
    torch.manual_seed(11)
    mu = torch.randn(B, D, device=device)
    log_sigma = 0.1 * torch.randn(B, D, device=device)
    table = torch.randn(B * I, D, device=device)
    ids = torch.arange(B * I, device=device, dtype=torch.int32).reshape(B, I)
    coef = torch.randn(B, S, device=device)
    valid = torch.randint(3, I + 1, (B,), device=device, dtype=torch.int32)
    valid[0] = I
    target_idx = (torch.rand(B, device=device) * (valid.float() - 0.5)).long()
    bias = 0.5 * torch.randn(B, I, device=device)
    SEED0 = 4242

    torch.manual_seed(SEED0)
    step_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    gen = torch.Generator(device=device).manual_seed(step_seed)
    buf = torch.empty(B, S_c, D, device=device, dtype=torch.bfloat16)
    eps_full = torch.cat(
        [torch.randn(B, S_c, D, generator=gen, out=buf).clone() for _ in range(S // S_c)], dim=1)

    def run(**kwargs):
        leaves = [t.clone().requires_grad_(True) for t in (mu, log_sigma, table, bias)]
        lg = triton_mxl_logprobs(*leaves[:3], ids, target_idx, S, bias=leaves[3],
                                 valid_count=valid, precision="ieee", **kwargs)
        return lg.detach(), torch.autograd.grad((coef * lg).sum(), leaves)

    torch.manual_seed(SEED0)
    lg_chunk, g_chunk = run(chunk_samples=S_c)
    lg_ref, g_ref = run(eps=eps_full)

    fdiff = (lg_chunk - lg_ref).abs().max().item()
    print(f"  chunked log_g vs unchunked (same noise) max diff = {fdiff:.2e}")
    assert fdiff < 1e-3, f"chunked forward mismatch: {fdiff}"
    for nm, a, b in zip(("mu", "log_sigma", "item_table", "bias"), g_chunk, g_ref):
        rel = _grad_rel_err(a, b)
        print(f"    chunked grad[{nm}] rel err vs unchunked = {rel:.2e}")
        assert rel < 1e-2, f"chunked grad {nm} mismatch: {rel}"
    pad = torch.arange(I, device=device)[None, :] >= valid[:, None]
    assert g_chunk[2][ids[pad].long()].abs().max().item() == 0.0, \
        "chunked table rows referenced only by padded slots got gradient"
    assert g_chunk[3][pad].abs().max().item() == 0.0, "chunked grad_bias not zero on padded slots"

    lg1 = triton_mxl_logprobs(mu, log_sigma, table, ids, target_idx, S, bias=bias,
                              valid_count=valid, chunk_samples=S_c)
    lg2 = triton_mxl_logprobs(mu, log_sigma, table, ids, target_idx, S, bias=bias,
                              valid_count=valid, chunk_samples=S_c)
    assert (lg1 - lg2).abs().max().item() > 0.0, "chunked mode reused the same noise across calls"
    print("✓ Triton chunked mode matches unchunked (same noise) and resamples per step")


def _fp32_reference(test):
    def wrapped(*args, **kwargs):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return test(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
    return wrapped


def test_triton_improved_chunked(B=6, S=128, S_c=32, I=37, D=48):
    if not torch.cuda.is_available():
        print("  (improved-kernel chunked test skipped: no CUDA)")
        return
    from src.models.fused_mxl_triton_improved import _standard_noise, triton_mxl_logprobs as improved
    device = "cuda"
    torch.manual_seed(11)
    mu = torch.randn(B, D, device=device)
    log_sigma = 0.1 * torch.randn(B, D, device=device)
    table = torch.randn(B * I, D, device=device)
    ids = torch.arange(B * I, device=device, dtype=torch.int32).reshape(B, I)
    coef = torch.randn(B, S, device=device)
    valid = torch.randint(3, I + 1, (B,), device=device, dtype=torch.int32)
    valid[0] = I
    target_idx = (torch.rand(B, device=device) * (valid.float() - 0.5)).long()
    bias = 0.5 * torch.randn(B, I, device=device)
    SEED0 = 4242
    torch.manual_seed(SEED0)
    eps_full = _standard_noise(B, S, D, int(torch.randint(0, 2**31 - 1, (1,)).item()), device)

    def run(**kwargs):
        leaves = [t.clone().requires_grad_(True) for t in (mu, log_sigma, table, bias)]
        lg = improved(*leaves[:3], ids, target_idx, S, bias=leaves[3], valid_count=valid,
                precision="ieee", **kwargs)
        return lg.detach(), torch.autograd.grad((coef * lg).sum(), leaves)

    lg_ref, g_ref = run(eps=eps_full)
    for label, kwargs in (("persist", {}), ("chunked", dict(chunk_samples=S_c))):
        torch.manual_seed(SEED0)
        lg, g = run(**kwargs)
        fdiff = (lg - lg_ref).abs().max().item()
        print(f"  improved {label} log_g vs eps path (same noise) max diff = {fdiff:.2e}")
        assert fdiff < 1e-3, f"improved {label} forward mismatch: {fdiff}"
        for nm, a, b in zip(("mu", "log_sigma", "item_table", "bias"), g, g_ref):
            rel = _grad_rel_err(a, b)
            assert rel < 1e-2, f"improved {label} grad {nm} mismatch: {rel}"
        pad = torch.arange(I, device=device)[None, :] >= valid[:, None]
        assert g[2][ids[pad].long()].abs().max().item() == 0.0, \
            f"improved {label}: table rows referenced only by padded slots got gradient"
        assert g[3][pad].abs().max().item() == 0.0, f"improved {label}: grad_bias not zero on padding"
    lg1 = improved(mu, log_sigma, table, ids, target_idx, S, bias=bias, valid_count=valid,
             chunk_samples=S_c)
    lg2 = improved(mu, log_sigma, table, ids, target_idx, S, bias=bias, valid_count=valid,
             chunk_samples=S_c)
    assert (lg1 - lg2).abs().max().item() > 0.0, "improved kernels reused the same noise across calls"
    print("✓ improved kernels: drawn noise (persist and chunked) matches the eps path and resamples")


def test_triton_improved():
    if not torch.cuda.is_available():
        print("  (improved-kernel tests skipped: no CUDA)")
        return
    from src.models.fused_mxl_triton_improved import triton_mxl_logprobs as improved
    for shape in (dict(), dict(I=150, D=256)):
        _fp32_reference(test_triton_backward)(logprobs=improved, **shape)
    for shape in (dict(), dict(I=137, D=200)):
        _fp32_reference(test_triton_padded_bias)(logprobs=improved, **shape)
    for shape in (dict(), dict(I=137, D=200)):
        test_triton_improved_chunked(**shape)


if __name__ == "__main__":
    has_cuda = torch.cuda.is_available()
    print("Testing the fused MXL Triton kernels...")
    print(f"Device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if has_cuda else 'cpu'}")

    print("\n1. Triton backward gradients")
    test_triton_backward()

    print("\n2. Triton padded slates + bias")
    test_triton_padded_bias()

    print("\n3. Triton sample-chunked eps (chunk_samples)")
    test_triton_chunked()

    print("\n4. Improved Triton training kernels (post-acceptance work)")
    test_triton_improved()

    print("\n✓ All tests passed!")
