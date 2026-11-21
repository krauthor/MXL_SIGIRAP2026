# The Triton kernels in this project were developed with the assistance of
# Claude Opus and Claude Fable (Anthropic).
#
# Post-acceptance work: written after the paper was accepted and not used for any result in
# it. The paper's kernels are in src/models/fused_mxl_triton.py.


import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from triton.language.extra.cuda import libdevice
from triton.language.random import philox

from src.models.fused_mxl_triton import _check_shapes


def _prune(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    max_i = triton.next_power_of_2(max(int(args["num_slots"]), 16))
    max_s = triton.next_power_of_2(max(int(args["num_samples"]), 16))
    keep = [c for c in configs
            if c.kwargs.get("BLOCK_I", 16) <= max_i and c.kwargs.get("BLOCK_S", 16) <= max_s]
    return keep or configs


def _prune_one_tile(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    min_i = triton.next_power_of_2(max(int(args["num_slots"]), 16))
    max_s = triton.next_power_of_2(max(int(args["num_samples"]), 16))
    keep = [c for c in configs if c.kwargs["BLOCK_I"] == min_i and c.kwargs["BLOCK_S"] <= max_s]
    return keep or [c for c in configs if c.kwargs["BLOCK_I"] >= min_i]


def _configs(names, rows):
    return [triton.Config(dict(zip(names, row[:-2])), num_warps=row[-2], num_stages=row[-1])
            for row in rows]


def _even(*pairs):
    return lambda args: all(args[extent] % args[block] == 0 for extent, block in pairs)


_KEY = ["num_samples", "num_slots", "latent_dim", "HAS_VALID"]
_SMALL_CONFIGS = _configs(["BLOCK_S", "BLOCK_I"], [
    (s, i, w, st) for i in (16, 32, 64)
    for s, w, st in [(32, 4, 2), (64, 4, 2), (128, 4, 2), (128, 8, 2), (64, 8, 1), (256, 8, 2)]])
_TILE_CONFIGS = _configs(["BLOCK_S", "BLOCK_I", "BLOCK_K"], [
    (128, 128, 32, 8, 3), (256, 128, 32, 8, 2), (128, 256, 32, 8, 3), (256, 64, 32, 8, 3),
    (128, 64, 32, 4, 4), (128, 128, 64, 8, 2), (64, 128, 32, 4, 4), (64, 64, 32, 4, 3)])
_ITEM_CONFIGS = _configs(["BLOCK_I", "BLOCK_D", "BLOCK_K"], [
    (64, 128, 32, 4, 4), (64, 128, 64, 4, 3), (128, 128, 32, 8, 3), (64, 256, 32, 8, 3),
    (128, 128, 64, 8, 2), (32, 128, 64, 4, 3), (64, 64, 64, 4, 3)])
_NOISE_CONFIGS = _configs(["BLOCK_S"], [(16, 4, 1), (32, 4, 1), (32, 8, 1), (64, 8, 1)])

_GRADS = ["grad_mu_ptr", "grad_log_sigma_ptr", "grad_table_ptr", "grad_bias_ptr"]
_SMALL_MAX_DIM = 128
_SMALL_MAX_SLOTS = 64
_MAX_TILES_BLOCK = 64
_GRAD_SCORES_BUDGET = 1 << 27


@triton.jit
def _dot(a, b, acc, PRECISION: tl.constexpr):
    if PRECISION == "bf16" or a.dtype == tl.bfloat16:
        out = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
    else:
        out = tl.dot(a, b, acc, input_precision=PRECISION)
    return out


@triton.jit
def _normal_pair(r0, r1):
    u1 = 2.0 - ((r0 >> 9) | 0x3F800000).to(tl.float32, bitcast=True)
    u2 = ((r1 >> 9) | 0x3F800000).to(tl.float32, bitcast=True) - 1.0
    radius = tl.sqrt(-1.3862943611198906 * libdevice.fast_log2f(u1))
    angle = 6.283185307179586 * u2 - 3.141592653589793
    return radius * libdevice.fast_cosf(angle), radius * libdevice.fast_sinf(angle)


@triton.jit
def _noise(seed, user, samples, cols, QUARTER: tl.constexpr):
    c0 = (samples[:, None] * QUARTER + cols[None, :]).to(tl.uint32)
    zero = c0 * 0
    r0, r1, r2, r3 = philox(seed, c0, zero + user.to(tl.uint32), zero, zero, 10)
    n0, n1 = _normal_pair(r0, r1)
    n2, n3 = _normal_pair(r2, r3)
    return n0, n1, n2, n3


@triton.jit
def _pick(x0, x1, x2, x3, j: tl.constexpr):
    if j == 0:
        out = x0
    elif j == 1:
        out = x1
    elif j == 2:
        out = x2
    else:
        out = x3
    return out


@triton.jit
def _row(valid_count_ptr, target_idx_ptr, user, num_slots, HAS_VALID: tl.constexpr):
    num_valid = tl.load(valid_count_ptr + user).to(tl.int32) if HAS_VALID else num_slots
    return num_valid, tl.load(target_idx_ptr + user).to(tl.int32)


@triton.jit
def _quarter_items(items_ptr, mu_ptr, user, items, cols, j: tl.constexpr,
                   QUARTER: tl.constexpr, latent_dim: tl.constexpr):
    dims = j * QUARTER + cols
    dim_mask = (cols < QUARTER) & (dims < latent_dim)
    v = tl.load(items_ptr + items[None, :] * latent_dim + dims[:, None], mask=dim_mask[:, None],
                other=0.0)
    mu = tl.load(mu_ptr + user * latent_dim + dims, mask=dim_mask, other=0.0)
    return v, tl.sum(v.to(tl.float32) * mu[:, None], axis=0)


@triton.jit
def _quarter_noise(a_ptr, log_sigma_ptr, normals, user, a_rows, sample_mask, cols,
                   j: tl.constexpr, QUARTER: tl.constexpr, latent_dim: tl.constexpr,
                   GEN: tl.constexpr):
    dims = j * QUARTER + cols
    dim_mask = (cols < QUARTER) & (dims < latent_dim)
    mask = sample_mask[:, None] & dim_mask[None, :]
    if GEN:
        sigma = tl.exp(tl.load(log_sigma_ptr + user * latent_dim + dims, mask=dim_mask,
                               other=0.0))
        a = tl.where(mask, sigma[None, :] * normals, 0.0)
    else:
        a = tl.load(a_ptr + a_rows[:, None] + dims[None, :], mask=mask, other=0.0)
    return a


@triton.jit
def _tile_scores(a_ptr, items_ptr, a_rows, items, sample_mask, latent_dim: tl.constexpr,
                 BLOCK_S: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_K: tl.constexpr,
                 EVEN: tl.constexpr, PRECISION: tl.constexpr):
    acc = tl.zeros((BLOCK_S, BLOCK_I), tl.float32)
    ks = tl.arange(0, BLOCK_K)
    for k in range(0, latent_dim, BLOCK_K):
        dims = k + ks
        if EVEN:
            a = tl.load(a_ptr + a_rows[:, None] + dims[None, :])
            v = tl.load(items_ptr + items[None, :] * latent_dim + dims[:, None])
        else:
            dim_mask = dims < latent_dim
            a = tl.load(a_ptr + a_rows[:, None] + dims[None, :],
                        mask=sample_mask[:, None] & dim_mask[None, :], other=0.0)
            v = tl.load(items_ptr + items[None, :] * latent_dim + dims[:, None],
                        mask=dim_mask[:, None], other=0.0)
        acc = _dot(a, v, acc, PRECISION)
    return acc


@triton.jit
def _tile_logsumexp(scores, slots, target):
    row_max = tl.max(scores, axis=1)
    row_sum = tl.sum(tl.exp(scores - row_max[:, None]), axis=1)
    return row_max, row_sum, tl.sum(tl.where(slots[None, :] == target, scores, 0.0), axis=1)


@triton.autotune(configs=_NOISE_CONFIGS, key=["num_samples", "latent_dim", "EPS_INPUT"])
@triton.jit(do_not_specialize=["seed", "user0"])
def _noise_kernel(eps_ptr, log_sigma_ptr, out_ptr, seed, user0, num_samples,
                  latent_dim: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_Q: tl.constexpr,
                  EPS_INPUT: tl.constexpr):
    QUARTER: tl.constexpr = (latent_dim + 3) // 4
    pid = tl.program_id(0)
    blocks = tl.cdiv(num_samples, BLOCK_S)
    local = pid // blocks
    user = user0 + local
    samples = (pid % blocks) * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_Q)
    if not EPS_INPUT:
        n0, n1, n2, n3 = _noise(seed, user, samples, cols, QUARTER)
    for j in tl.static_range(4):
        dims = j * QUARTER + cols
        dim_mask = (cols < QUARTER) & (dims < latent_dim)
        mask = (samples < num_samples)[:, None] & dim_mask[None, :]
        sigma = tl.exp(tl.load(log_sigma_ptr + user * latent_dim + dims, mask=dim_mask,
                               other=0.0))
        if EPS_INPUT:
            eps = tl.load(eps_ptr + (user.to(tl.int64) * num_samples + samples[:, None])
                          * latent_dim + dims[None, :], mask=mask, other=0.0).to(tl.float32)
        else:
            eps = _pick(n0, n1, n2, n3, j)
        tl.store(out_ptr + (local.to(tl.int64) * num_samples + samples[:, None]) * latent_dim
                 + dims[None, :], (sigma[None, :] * eps).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _offset_kernel(mu_ptr, items_ptr, option_id_ptr, bias_ptr, valid_count_ptr, offset_ptr,
                   num_slots, latent_dim: tl.constexpr, BLOCK_I: tl.constexpr,
                   BLOCK_K: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr):
    pid = tl.program_id(0)
    tiles = tl.cdiv(num_slots, BLOCK_I)
    user = pid // tiles
    num_valid = tl.load(valid_count_ptr + user).to(tl.int32) if HAS_VALID else num_slots
    slots = (pid % tiles) * BLOCK_I + tl.arange(0, BLOCK_I)
    slot_mask = slots < num_valid
    slot_row = user.to(tl.int64) * num_slots
    items = tl.load(option_id_ptr + slot_row + slots, mask=slot_mask, other=0).to(tl.int64)
    offset = tl.zeros((BLOCK_I,), tl.float32)
    for k in range(0, latent_dim, BLOCK_K):
        dims = k + tl.arange(0, BLOCK_K)
        dim_mask = dims < latent_dim
        v = tl.load(items_ptr + items[:, None] * latent_dim + dims[None, :],
                    mask=dim_mask[None, :], other=0.0)
        mu = tl.load(mu_ptr + user * latent_dim + dims, mask=dim_mask, other=0.0)
        offset += tl.sum(v.to(tl.float32) * mu[None, :], axis=1)
    if HAS_BIAS:
        offset += tl.load(bias_ptr + slot_row + slots, mask=slot_mask, other=0.0)
    tl.store(offset_ptr + slot_row + slots, offset, mask=slots < num_slots)


@triton.autotune(configs=_SMALL_CONFIGS, key=_KEY + ["HAS_BIAS", "GEN", "PRECISION"],
                 prune_configs_by={"early_config_prune": _prune_one_tile})
@triton.jit(do_not_specialize=["seed"])
def _forward_small_kernel(
    a_ptr, mu_ptr, log_sigma_ptr, items_ptr, option_id_ptr, bias_ptr, target_idx_ptr,
    valid_count_ptr, log_g_ptr, lse_ptr, seed, num_samples, num_slots,
    latent_dim: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_I: tl.constexpr,
    BLOCK_Q: tl.constexpr, GEN: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
    PRECISION: tl.constexpr,
):
    QUARTER: tl.constexpr = (latent_dim + 3) // 4
    pid = tl.program_id(0)
    blocks = tl.cdiv(num_samples, BLOCK_S)
    user = pid // blocks
    num_valid, target = _row(valid_count_ptr, target_idx_ptr, user, num_slots, HAS_VALID)
    samples = (pid % blocks) * BLOCK_S + tl.arange(0, BLOCK_S)
    sample_mask = samples < num_samples
    slots = tl.arange(0, BLOCK_I)
    slot_mask = slots < num_valid
    slot_row = user.to(tl.int64) * num_slots
    items = tl.load(option_id_ptr + slot_row + slots, mask=slot_mask, other=0).to(tl.int64)
    a_rows = (user.to(tl.int64) * num_samples + samples) * latent_dim
    cols = tl.arange(0, BLOCK_Q)
    if GEN:
        n0, n1, n2, n3 = _noise(seed, user, samples, cols, QUARTER)
    else:
        n0, n1, n2, n3 = 0.0, 0.0, 0.0, 0.0
    acc = tl.zeros((BLOCK_S, BLOCK_I), tl.float32)
    offset = tl.zeros((BLOCK_I,), tl.float32)
    for j in tl.static_range(4):
        a = _quarter_noise(a_ptr, log_sigma_ptr, _pick(n0, n1, n2, n3, j), user, a_rows,
                           sample_mask, cols, j, QUARTER, latent_dim, GEN)
        v, part = _quarter_items(items_ptr, mu_ptr, user, items, cols, j, QUARTER, latent_dim)
        acc = _dot(a, v, acc, PRECISION)
        offset += part
    if HAS_BIAS:
        offset += tl.load(bias_ptr + slot_row + slots, mask=slot_mask, other=0.0)
    scores = tl.where(slot_mask[None, :], acc + offset[None, :], float("-inf"))
    row_max, row_sum, target_score = _tile_logsumexp(scores, slots, target)
    lse = row_max + tl.log(row_sum)
    out = user.to(tl.int64) * num_samples + samples
    tl.store(log_g_ptr + out, tl.minimum(target_score - lse, 0.0), mask=sample_mask)
    tl.store(lse_ptr + out, lse, mask=sample_mask)


@triton.autotune(configs=_TILE_CONFIGS, key=_KEY + ["PRECISION"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.heuristics({"EVEN": _even(("latent_dim", "BLOCK_K"), ("num_samples", "BLOCK_S"))})
@triton.jit(do_not_specialize=["user0"])
def _forward_tile_kernel(
    a_ptr, items_ptr, option_id_ptr, target_idx_ptr, valid_count_ptr, offset_ptr,
    part_max_ptr, part_sum_ptr, target_score_ptr, log_g_ptr, lse_ptr,
    user0, num_samples, num_slots, max_tiles,
    latent_dim: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr, EVEN: tl.constexpr, HAS_VALID: tl.constexpr, PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles = tl.cdiv(num_slots, BLOCK_I)
    blocks = tl.cdiv(num_samples, BLOCK_S)
    tile = pid % tiles
    local = pid // (tiles * blocks)
    user = user0 + local
    num_valid, target = _row(valid_count_ptr, target_idx_ptr, user, num_slots, HAS_VALID)
    start = tile * BLOCK_I
    if start >= num_valid:
        return
    samples = (pid // tiles) % blocks * BLOCK_S + tl.arange(0, BLOCK_S)
    sample_mask = samples < num_samples
    slots = start + tl.arange(0, BLOCK_I)
    slot_mask = slots < num_valid
    slot_row = user.to(tl.int64) * num_slots
    items = tl.load(option_id_ptr + slot_row + slots, mask=slot_mask, other=0).to(tl.int64)
    a_rows = (local.to(tl.int64) * num_samples + samples) * latent_dim
    acc = _tile_scores(a_ptr, items_ptr, a_rows, items, sample_mask, latent_dim, BLOCK_S,
                       BLOCK_I, BLOCK_K, EVEN, PRECISION)
    offset = tl.load(offset_ptr + slot_row + slots, mask=slot_mask, other=0.0)
    scores = tl.where(slot_mask[None, :], acc + offset[None, :], float("-inf"))
    row_max, row_sum, target_score = _tile_logsumexp(scores, slots, target)
    out = user.to(tl.int64) * num_samples + samples
    if tiles == 1:
        lse = row_max + tl.log(row_sum)
        tl.store(log_g_ptr + out, tl.minimum(target_score - lse, 0.0), mask=sample_mask)
        tl.store(lse_ptr + out, lse, mask=sample_mask)
    else:
        part = (user.to(tl.int64) * max_tiles + tile) * num_samples + samples
        tl.store(part_max_ptr + part, row_max, mask=sample_mask)
        tl.store(part_sum_ptr + part, row_sum, mask=sample_mask)
        if (target >= start) & (target < start + BLOCK_I):
            tl.store(target_score_ptr + out, target_score, mask=sample_mask)


@triton.jit(do_not_specialize=["user0"])
def _merge_kernel(part_max_ptr, part_sum_ptr, target_score_ptr, valid_count_ptr, log_g_ptr,
                  lse_ptr, user0, num_samples, num_slots, max_tiles, tile_size,
                  BLOCK_S: tl.constexpr, HAS_VALID: tl.constexpr):
    pid = tl.program_id(0)
    blocks = tl.cdiv(num_samples, BLOCK_S)
    user = user0 + pid // blocks
    samples = (pid % blocks) * BLOCK_S + tl.arange(0, BLOCK_S)
    sample_mask = samples < num_samples
    num_valid = tl.load(valid_count_ptr + user).to(tl.int32) if HAS_VALID else num_slots
    run_max = tl.full((BLOCK_S,), float("-inf"), tl.float32)
    run_sum = tl.zeros((BLOCK_S,), tl.float32)
    for tile in range(0, tl.cdiv(num_valid, tile_size)):
        part = (user.to(tl.int64) * max_tiles + tile) * num_samples + samples
        tile_max = tl.load(part_max_ptr + part, mask=sample_mask, other=0.0)
        new_max = tl.maximum(run_max, tile_max)
        run_sum = (run_sum * tl.exp(run_max - new_max)
                   + tl.load(part_sum_ptr + part, mask=sample_mask, other=0.0)
                   * tl.exp(tile_max - new_max))
        run_max = new_max
    lse = run_max + tl.log(run_sum)
    out = user.to(tl.int64) * num_samples + samples
    target_score = tl.load(target_score_ptr + out, mask=sample_mask, other=0.0)
    tl.store(lse_ptr + out, lse, mask=sample_mask)
    tl.store(log_g_ptr + out, tl.minimum(target_score - lse, 0.0), mask=sample_mask)


@triton.autotune(configs=_SMALL_CONFIGS, key=_KEY + ["HAS_BIAS", "GEN"], reset_to_zero=_GRADS,
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit(do_not_specialize=["seed"])
def _backward_small_kernel(
    a_ptr, mu_ptr, log_sigma_ptr, items_ptr, option_id_ptr, bias_ptr, target_idx_ptr,
    valid_count_ptr, grad_log_g_ptr, lse_ptr,
    grad_mu_ptr, grad_log_sigma_ptr, grad_table_ptr, grad_bias_ptr,
    seed, num_samples, num_slots,
    latent_dim: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_I: tl.constexpr,
    BLOCK_Q: tl.constexpr, GEN: tl.constexpr, HAS_BIAS: tl.constexpr, HAS_VALID: tl.constexpr,
):
    QUARTER: tl.constexpr = (latent_dim + 3) // 4
    pid = tl.program_id(0)
    tiles = tl.cdiv(num_slots, BLOCK_I)
    user = pid // tiles
    num_valid, target = _row(valid_count_ptr, target_idx_ptr, user, num_slots, HAS_VALID)
    start = (pid % tiles) * BLOCK_I
    if start >= num_valid:
        return
    slots = start + tl.arange(0, BLOCK_I)
    slot_mask = slots < num_valid
    slot_row = user.to(tl.int64) * num_slots
    items = tl.load(option_id_ptr + slot_row + slots, mask=slot_mask, other=0).to(tl.int64)
    is_target = tl.where(slots == target, 1.0, 0.0)
    cols = tl.arange(0, BLOCK_Q)
    v0, o0 = _quarter_items(items_ptr, mu_ptr, user, items, cols, 0, QUARTER, latent_dim)
    v1, o1 = _quarter_items(items_ptr, mu_ptr, user, items, cols, 1, QUARTER, latent_dim)
    v2, o2 = _quarter_items(items_ptr, mu_ptr, user, items, cols, 2, QUARTER, latent_dim)
    v3, o3 = _quarter_items(items_ptr, mu_ptr, user, items, cols, 3, QUARTER, latent_dim)
    offset = o0 + o1 + o2 + o3
    if HAS_BIAS:
        offset += tl.load(bias_ptr + slot_row + slots, mask=slot_mask, other=0.0)
    h0 = tl.zeros((BLOCK_I, BLOCK_Q), tl.float32)
    h1 = tl.zeros((BLOCK_I, BLOCK_Q), tl.float32)
    h2 = tl.zeros((BLOCK_I, BLOCK_Q), tl.float32)
    h3 = tl.zeros((BLOCK_I, BLOCK_Q), tl.float32)
    colsum = tl.zeros((BLOCK_I,), tl.float32)
    sample_row = user.to(tl.int64) * num_samples
    for s0 in range(0, num_samples, BLOCK_S):
        samples = s0 + tl.arange(0, BLOCK_S)
        sample_mask = samples < num_samples
        a_rows = (sample_row + samples) * latent_dim
        if GEN:
            n0, n1, n2, n3 = _noise(seed, user, samples, cols, QUARTER)
        else:
            n0, n1, n2, n3 = 0.0, 0.0, 0.0, 0.0
        a0 = _quarter_noise(a_ptr, log_sigma_ptr, n0, user, a_rows, sample_mask, cols, 0,
                            QUARTER, latent_dim, GEN).to(tl.bfloat16)
        a1 = _quarter_noise(a_ptr, log_sigma_ptr, n1, user, a_rows, sample_mask, cols, 1,
                            QUARTER, latent_dim, GEN).to(tl.bfloat16)
        a2 = _quarter_noise(a_ptr, log_sigma_ptr, n2, user, a_rows, sample_mask, cols, 2,
                            QUARTER, latent_dim, GEN).to(tl.bfloat16)
        a3 = _quarter_noise(a_ptr, log_sigma_ptr, n3, user, a_rows, sample_mask, cols, 3,
                            QUARTER, latent_dim, GEN).to(tl.bfloat16)
        scores = tl.dot(a0, v0.to(tl.bfloat16))
        scores = tl.dot(a1, v1.to(tl.bfloat16), scores)
        scores = tl.dot(a2, v2.to(tl.bfloat16), scores)
        scores = tl.dot(a3, v3.to(tl.bfloat16), scores)
        lse = tl.load(lse_ptr + sample_row + samples, mask=sample_mask, other=0.0)
        grad = tl.load(grad_log_g_ptr + sample_row + samples, mask=sample_mask, other=0.0)
        prob = tl.exp(scores + offset[None, :] - lse[:, None])
        grad_scores = grad[:, None] * (is_target[None, :] - prob)
        grad_scores = tl.where(slot_mask[None, :] & sample_mask[:, None], grad_scores, 0.0)
        colsum += tl.sum(grad_scores, axis=0)
        grad_scores_t = tl.trans(grad_scores.to(tl.bfloat16))
        h0 = tl.dot(grad_scores_t, a0, h0)
        h1 = tl.dot(grad_scores_t, a1, h1)
        h2 = tl.dot(grad_scores_t, a2, h2)
        h3 = tl.dot(grad_scores_t, a3, h3)
    if HAS_BIAS:
        tl.store(grad_bias_ptr + slot_row + slots, colsum, mask=slot_mask)
    for j in tl.static_range(4):
        dims = j * QUARTER + cols
        dim_mask = (cols < QUARTER) & (dims < latent_dim)
        h = _pick(h0, h1, h2, h3, j)
        v = tl.trans(_pick(v0, v1, v2, v3, j)).to(tl.float32)
        mu = tl.load(mu_ptr + user * latent_dim + dims, mask=dim_mask, other=0.0)
        tl.atomic_add(grad_table_ptr + items[:, None] * latent_dim + dims[None, :],
                      colsum[:, None] * mu[None, :] + h,
                      mask=slot_mask[:, None] & dim_mask[None, :], sem="relaxed")
        tl.atomic_add(grad_mu_ptr + user * latent_dim + dims,
                      tl.sum(colsum[:, None] * v, axis=0), mask=dim_mask, sem="relaxed")
        tl.atomic_add(grad_log_sigma_ptr + user * latent_dim + dims,
                      tl.sum(v * h, axis=0), mask=dim_mask, sem="relaxed")


@triton.autotune(configs=_TILE_CONFIGS, key=_KEY, reset_to_zero=["colsum_ptr"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.heuristics({"EVEN": _even(("latent_dim", "BLOCK_K"), ("num_samples", "BLOCK_S"))})
@triton.jit(do_not_specialize=["user0", "a_row0"])
def _backward_scores_kernel(
    a_ptr, items_ptr, option_id_ptr, target_idx_ptr, valid_count_ptr, offset_ptr,
    grad_log_g_ptr, lse_ptr, grad_scores_ptr, colsum_ptr,
    user0, a_row0, num_samples, num_slots,
    latent_dim: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr, EVEN: tl.constexpr, HAS_VALID: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles = tl.cdiv(num_slots, BLOCK_I)
    blocks = tl.cdiv(num_samples, BLOCK_S)
    tile = pid % tiles
    local = pid // (tiles * blocks)
    user = user0 + local
    num_valid, target = _row(valid_count_ptr, target_idx_ptr, user, num_slots, HAS_VALID)
    start = tile * BLOCK_I
    if start >= num_valid:
        return
    samples = (pid // tiles) % blocks * BLOCK_S + tl.arange(0, BLOCK_S)
    sample_mask = samples < num_samples
    slots = start + tl.arange(0, BLOCK_I)
    slot_mask = slots < num_valid
    slot_row = user.to(tl.int64) * num_slots
    items = tl.load(option_id_ptr + slot_row + slots, mask=slot_mask, other=0).to(tl.int64)
    a_rows = ((a_row0 + local).to(tl.int64) * num_samples + samples) * latent_dim
    acc = _tile_scores(a_ptr, items_ptr, a_rows, items, sample_mask, latent_dim, BLOCK_S,
                       BLOCK_I, BLOCK_K, EVEN, "bf16")
    offset = tl.load(offset_ptr + slot_row + slots, mask=slot_mask, other=0.0)
    sample_row = user.to(tl.int64) * num_samples + samples
    lse = tl.load(lse_ptr + sample_row, mask=sample_mask, other=0.0)
    grad = tl.load(grad_log_g_ptr + sample_row, mask=sample_mask, other=0.0)
    prob = tl.exp(acc + offset[None, :] - lse[:, None])
    grad_scores = grad[:, None] * (tl.where(slots[None, :] == target, 1.0, 0.0) - prob)
    grad_scores = tl.where(slot_mask[None, :] & sample_mask[:, None], grad_scores, 0.0)
    tl.store(grad_scores_ptr + (local.to(tl.int64) * num_slots + slots[:, None]) * num_samples
             + samples[None, :], tl.trans(grad_scores).to(tl.bfloat16),
             mask=slot_mask[:, None] & sample_mask[None, :])
    tl.atomic_add(colsum_ptr + slot_row + slots, tl.sum(grad_scores, axis=0), mask=slot_mask,
                  sem="relaxed")


@triton.autotune(configs=_ITEM_CONFIGS, key=_KEY, reset_to_zero=_GRADS[:3],
                 prune_configs_by={"early_config_prune": _prune})
@triton.heuristics({"EVEN": _even(("latent_dim", "BLOCK_D"), ("num_samples", "BLOCK_K"))})
@triton.jit(do_not_specialize=["user0", "a_row0"])
def _backward_items_kernel(
    a_ptr, mu_ptr, items_ptr, option_id_ptr, valid_count_ptr, grad_scores_ptr, colsum_ptr,
    grad_mu_ptr, grad_log_sigma_ptr, grad_table_ptr,
    user0, a_row0, num_samples, num_slots,
    latent_dim: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr, EVEN: tl.constexpr, HAS_VALID: tl.constexpr,
):
    pid = tl.program_id(0)
    dim_tiles = tl.cdiv(latent_dim, BLOCK_D)
    slot_tiles = tl.cdiv(num_slots, BLOCK_I)
    local = pid // (dim_tiles * slot_tiles)
    user = user0 + local
    num_valid = tl.load(valid_count_ptr + user).to(tl.int32) if HAS_VALID else num_slots
    start = (pid // dim_tiles) % slot_tiles * BLOCK_I
    if start >= num_valid:
        return
    slots = start + tl.arange(0, BLOCK_I)
    slot_mask = slots < num_valid
    dims = (pid % dim_tiles) * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dims < latent_dim
    ks = tl.arange(0, BLOCK_K)
    g_ptrs = (grad_scores_ptr + (local.to(tl.int64) * num_slots + slots[:, None]) * num_samples
              + ks[None, :])
    a_ptrs = (a_ptr + ((a_row0 + local).to(tl.int64) * num_samples + ks[:, None]) * latent_dim
              + dims[None, :])
    h = tl.zeros((BLOCK_I, BLOCK_D), tl.float32)
    for k in range(0, num_samples, BLOCK_K):
        if EVEN:
            g = tl.load(g_ptrs, mask=slot_mask[:, None], other=0.0)
            a = tl.load(a_ptrs)
        else:
            sample_mask = (k + ks) < num_samples
            g = tl.load(g_ptrs, mask=slot_mask[:, None] & sample_mask[None, :], other=0.0)
            a = tl.load(a_ptrs, mask=sample_mask[:, None] & dim_mask[None, :], other=0.0)
        h = tl.dot(g, a.to(tl.bfloat16), h)
        g_ptrs += BLOCK_K
        a_ptrs += BLOCK_K * latent_dim
    slot_row = user.to(tl.int64) * num_slots
    colsum = tl.load(colsum_ptr + slot_row + slots, mask=slot_mask, other=0.0)
    items = tl.load(option_id_ptr + slot_row + slots, mask=slot_mask, other=0).to(tl.int64)
    mask = slot_mask[:, None] & dim_mask[None, :]
    v = tl.load(items_ptr + items[:, None] * latent_dim + dims[None, :], mask=mask,
                other=0.0).to(tl.float32)
    mu = tl.load(mu_ptr + user * latent_dim + dims, mask=dim_mask, other=0.0)
    tl.atomic_add(grad_table_ptr + items[:, None] * latent_dim + dims[None, :],
                  colsum[:, None] * mu[None, :] + h, mask=mask, sem="relaxed")
    tl.atomic_add(grad_mu_ptr + user * latent_dim + dims, tl.sum(colsum[:, None] * v, axis=0),
                  mask=dim_mask, sem="relaxed")
    tl.atomic_add(grad_log_sigma_ptr + user * latent_dim + dims, tl.sum(v * h, axis=0),
                  mask=dim_mask, sem="relaxed")


_FIELDS = ["mu", "log_sigma", "item_table", "option_id", "target_idx", "bias", "valid_count",
           "eps"]
_PLANS = {}
_LAUNCHES = {}
_DUMMIES = {}


def _launch(kernel, grid, key, **args):
    entry = _LAUNCHES.get((kernel, key))
    if entry is None:
        compiled = kernel[grid](**args)
        meta = dict(kernel.best_config.kwargs) if hasattr(kernel, "best_config") else {}
        jit = kernel
        while not isinstance(jit, triton.runtime.JITFunction):
            jit = jit.fn
        names = [p.name for p in jit.params if not p.is_constexpr]
        _LAUNCHES[(kernel, key)] = (compiled, names, tuple(grid({**args, **meta})) + (1, 1), meta)
        return meta
    compiled, names, size, meta = entry
    values = [args[name] for name in names]
    stream = torch._C._cuda_getCurrentRawStream(torch.cuda.current_device())
    compiled.run(size[0], size[1], size[2], stream, compiled.function, compiled.packed_metadata,
                 None, None, None,
                 *[v.data_ptr() if isinstance(v, torch.Tensor) else v for v in values])
    return meta


def _block_q(latent_dim):
    return max(16, triton.next_power_of_2((latent_dim + 3) // 4))


class _Plan:

    def __init__(self, key, t, num_samples, chunk_samples, precision):
        (B, D), I, N = t["mu"].shape, t["option_id"].shape[-1], len(t["item_table"])
        if triton.next_power_of_2(D) < 16:
            raise ValueError(f"latent dim must be >= 9, got {D}")
        chunk = chunk_samples or num_samples
        if num_samples % chunk:
            raise ValueError(f"chunk_samples ({chunk}) must divide num_samples ({num_samples})")
        if t["eps"] is not None and chunk != num_samples:
            raise ValueError("eps cannot be combined with chunk_samples")
        _check_shapes(dict(B=B, D=D, I=I, N=N, S=num_samples), **t)
        self.key, self.B, self.D, self.I, self.S, self.N = key, B, D, I, num_samples, N
        self.precision = precision
        self.eps_given = t["eps"] is not None
        self.small = D <= _SMALL_MAX_DIM and I <= _SMALL_MAX_SLOTS
        self.rows = B if chunk == num_samples else max(1, B * chunk // num_samples)
        self.noise_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.max_tiles = triton.cdiv(I, _MAX_TILES_BLOCK)
        sizes = [B * D, B * D, N * D, B * I]
        ends = [sum(sizes[:k + 1]) for k in range(4)]
        self.grad_slices = list(zip([0] + ends[:3], ends, (D, D, D, I)))
        device = t["mu"].device
        if device not in _DUMMIES:
            _DUMMIES[device] = torch.empty(1, device=device)
        self.dummy = _DUMMIES[device]
        self.flags = dict(HAS_BIAS=t["bias"] is not None, HAS_VALID=t["valid_count"] is not None)
        self.sizes = dict(num_samples=num_samples, num_slots=I, latent_dim=D)

    def inputs(self, t, items):
        return dict(mu_ptr=t["mu"], log_sigma_ptr=t["log_sigma"], items_ptr=items,
                    option_id_ptr=t["option_id"], target_idx_ptr=t["target_idx"],
                    bias_ptr=self.dummy if t["bias"] is None else t["bias"],
                    valid_count_ptr=self.dummy if t["valid_count"] is None else t["valid_count"])


def _plan(tensors, num_samples, chunk_samples, precision):
    tensors = [None if x is None else x.contiguous() for x in tensors]
    key = (num_samples, chunk_samples, precision) + tuple(
        None if x is None else (x.shape, x.dtype, x.device, x.data_ptr() % 16) for x in tensors)
    t = dict(zip(_FIELDS, tensors))
    if key not in _PLANS:
        _PLANS[key] = _Plan(key, t, num_samples, chunk_samples, precision)
    return _PLANS[key], t


def _write_noise(p, t, out, user0, rows, seed):
    _launch(_noise_kernel, lambda META: (rows * triton.cdiv(p.S, META["BLOCK_S"]),),
            (p.key, "noise", rows, out.dtype),
            eps_ptr=p.dummy if t.get("eps") is None else t["eps"], log_sigma_ptr=t["log_sigma"],
            out_ptr=out, seed=seed, user0=user0, num_samples=p.S, latent_dim=p.D,
            BLOCK_Q=triton.next_power_of_2((p.D + 3) // 4), EPS_INPUT=p.eps_given)


def _standard_noise(num_rows, num_samples, latent_dim, seed, device):
    out = torch.empty(num_rows, num_samples, latent_dim, device=device)
    _noise_kernel[lambda META: (num_rows * triton.cdiv(num_samples, META["BLOCK_S"]),)](
        eps_ptr=out, log_sigma_ptr=torch.zeros(num_rows, latent_dim, device=device),
        out_ptr=out, seed=seed, user0=0, num_samples=num_samples, latent_dim=latent_dim,
        BLOCK_Q=triton.next_power_of_2((latent_dim + 3) // 4), EPS_INPUT=False)
    return out


def _forward_small(p, t, items, seed, log_g, lse):
    kept = None
    if p.eps_given:
        kept = torch.empty(p.B, p.S, p.D, device=log_g.device, dtype=p.noise_dtype)
        _write_noise(p, t, kept, 0, p.B, seed)
    _launch(_forward_small_kernel,
            lambda META: (p.B * triton.cdiv(p.S, META["BLOCK_S"]),), (p.key, "forward"),
            a_ptr=p.dummy if kept is None else kept, **p.inputs(t, items), log_g_ptr=log_g,
            lse_ptr=lse, seed=seed, **p.sizes, BLOCK_Q=_block_q(p.D), GEN=kept is None,
            **p.flags, PRECISION=p.precision)
    return kept


def _forward_large(p, t, items, seed, log_g, lse):
    B, S, I, device = p.B, p.S, p.I, log_g.device
    valid_count = p.inputs(t, items)["valid_count_ptr"]
    offsets = torch.empty(B, I, device=device)
    _launch(_offset_kernel, lambda META: (B * triton.cdiv(I, 32),), (p.key, "offsets"),
            mu_ptr=t["mu"], items_ptr=items, option_id_ptr=t["option_id"],
            bias_ptr=p.dummy if t["bias"] is None else t["bias"], valid_count_ptr=valid_count,
            offset_ptr=offsets, num_slots=I, latent_dim=p.D, BLOCK_I=32,
            BLOCK_K=min(128, triton.next_power_of_2(p.D)), **p.flags, num_warps=4)
    part_max = torch.empty(B, p.max_tiles, S, device=device)
    parts = dict(part_max_ptr=part_max, part_sum_ptr=torch.empty_like(part_max),
                 target_score_ptr=torch.empty_like(log_g), log_g_ptr=log_g, lse_ptr=lse)
    noise = torch.empty(p.rows, S, p.D, device=device, dtype=p.noise_dtype)
    for user0 in range(0, B, p.rows):
        rows = min(p.rows, B - user0)
        _write_noise(p, t, noise, user0, rows, seed)
        meta = _launch(_forward_tile_kernel,
                       lambda META: (rows * triton.cdiv(S, META["BLOCK_S"])
                                     * triton.cdiv(I, META["BLOCK_I"]),),
                       (p.key, "forward", rows),
                       a_ptr=noise, items_ptr=items, option_id_ptr=t["option_id"],
                       target_idx_ptr=t["target_idx"], valid_count_ptr=valid_count,
                       offset_ptr=offsets, **parts, user0=user0, **p.sizes,
                       max_tiles=p.max_tiles, HAS_VALID=p.flags["HAS_VALID"],
                       PRECISION=p.precision)
        if triton.cdiv(I, meta["BLOCK_I"]) > 1:
            _launch(_merge_kernel, lambda META: (rows * triton.cdiv(S, 128),),
                    (p.key, "merge", rows), **parts, valid_count_ptr=valid_count, user0=user0,
                    num_samples=S, num_slots=I, max_tiles=p.max_tiles,
                    tile_size=meta["BLOCK_I"], BLOCK_S=128, HAS_VALID=p.flags["HAS_VALID"],
                    num_warps=4)
    return (noise if p.rows == B else None), offsets


def _backward_small(p, t, items, seed, kept, grad_log_g, lse, grads):
    _launch(_backward_small_kernel,
            lambda META: (p.B * triton.cdiv(p.I, META["BLOCK_I"]),), (p.key, "backward"),
            a_ptr=p.dummy if kept is None else kept, **p.inputs(t, items),
            grad_log_g_ptr=grad_log_g, lse_ptr=lse, **grads, seed=seed,
            num_samples=p.S, num_slots=p.I, latent_dim=p.D, BLOCK_Q=_block_q(p.D),
            GEN=kept is None, **p.flags)


def _backward_large(p, t, items, seed, kept, offsets, grad_log_g, lse, grads):
    B, S, I, D, device = p.B, p.S, p.I, p.D, lse.device
    valid_count = p.inputs(t, items)["valid_count_ptr"]
    rows = max(1, min(p.rows, _GRAD_SCORES_BUDGET // (2 * S * I)))
    if kept is None:
        noise = torch.empty(rows, S, D, device=device, dtype=torch.bfloat16)
    grad_scores = torch.empty(rows, I, S, device=device, dtype=torch.bfloat16)
    for user0 in range(0, B, rows):
        n = min(rows, B - user0)
        if kept is None:
            _write_noise(p, t, noise, user0, n, seed)
        a, a_row0 = (noise, 0) if kept is None else (kept, user0)
        _launch(_backward_scores_kernel,
                lambda META: (n * triton.cdiv(S, META["BLOCK_S"])
                              * triton.cdiv(I, META["BLOCK_I"]),),
                (p.key, "scores", n, a.dtype),
                a_ptr=a, items_ptr=items, option_id_ptr=t["option_id"],
                target_idx_ptr=t["target_idx"], valid_count_ptr=valid_count, offset_ptr=offsets,
                grad_log_g_ptr=grad_log_g, lse_ptr=lse, grad_scores_ptr=grad_scores,
                colsum_ptr=grads["grad_bias_ptr"], user0=user0, a_row0=a_row0, **p.sizes,
                HAS_VALID=p.flags["HAS_VALID"])
        _launch(_backward_items_kernel,
                lambda META: (n * triton.cdiv(D, META["BLOCK_D"])
                              * triton.cdiv(I, META["BLOCK_I"]),),
                (p.key, "items", n, a.dtype),
                a_ptr=a, mu_ptr=t["mu"], items_ptr=items, option_id_ptr=t["option_id"],
                valid_count_ptr=valid_count, grad_scores_ptr=grad_scores,
                colsum_ptr=grads["grad_bias_ptr"], grad_mu_ptr=grads["grad_mu_ptr"],
                grad_log_sigma_ptr=grads["grad_log_sigma_ptr"],
                grad_table_ptr=grads["grad_table_ptr"], user0=user0, a_row0=a_row0, **p.sizes,
                HAS_VALID=p.flags["HAS_VALID"])


class _TritonMXLLogProbs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mu, log_sigma, item_table, option_id, bias, target_idx, valid_count,
                num_samples, chunk_samples, eps, precision):
        p, t = _plan((mu, log_sigma, item_table, option_id, target_idx, bias, valid_count, eps),
                     num_samples, chunk_samples, precision)
        seed = 0 if p.eps_given else int(torch.randint(0, 2**31 - 1, (1,)).item())
        items = (t["item_table"].to(torch.bfloat16) if precision == "bf16" and not p.small
                 else t["item_table"])
        log_g = torch.empty(p.B, p.S, device=mu.device)
        lse = torch.empty(p.B, p.S, device=mu.device)
        if p.small:
            kept, offsets = _forward_small(p, t, items, seed, log_g, lse), None
        else:
            kept, offsets = _forward_large(p, t, items, seed, log_g, lse)
        ctx.plan, ctx.seed, ctx.inputs, ctx.buffers = p, seed, t, (items, lse, kept, offsets)
        return log_g

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_log_g):
        p, seed, t, (items, lse, kept, offsets) = ctx.plan, ctx.seed, ctx.inputs, ctx.buffers
        flat = torch.zeros(p.grad_slices[-1][1], device=lse.device)
        grad_mu, grad_log_sigma, grad_table, grad_bias = (
            flat[a:b].view(-1, n) for a, b, n in p.grad_slices)
        grads = dict(grad_mu_ptr=grad_mu, grad_log_sigma_ptr=grad_log_sigma,
                     grad_table_ptr=grad_table, grad_bias_ptr=grad_bias)
        grad_log_g = grad_log_g.contiguous()
        if p.small:
            _backward_small(p, t, items, seed, kept, grad_log_g, lse, grads)
        else:
            _backward_large(p, t, items, seed, kept, offsets, grad_log_g, lse, grads)
        out = [g if x is None or x.dtype == g.dtype else g.to(x.dtype)
               for g, x in zip((grad_mu, grad_log_sigma, grad_table, grad_bias),
                               (t["mu"], t["log_sigma"], t["item_table"], t["bias"]))]
        return (out[0], out[1], out[2], None, None if t["bias"] is None else out[3],
                None, None, None, None, None, None)


def triton_mxl_logprobs(mu, log_sigma, item_table, option_id, target_idx, num_samples,
                        bias=None, valid_count=None, chunk_samples=None, eps=None,
                        precision="bf16"):
    if precision not in ("bf16", "tf32", "ieee"):
        raise ValueError(f"precision must be 'bf16', 'tf32' or 'ieee', got {precision!r}")
    return _TritonMXLLogProbs.apply(mu, log_sigma, item_table, option_id, bias, target_idx,
                                    valid_count, num_samples, chunk_samples, eps, precision)
