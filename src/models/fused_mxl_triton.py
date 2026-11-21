# The Triton kernels in this project were developed with the assistance of
# Claude Opus and Claude Fable (Anthropic).


from collections import namedtuple

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


def _prune_configs_to_shape(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    max_i = triton.next_power_of_2(max(int(args["num_slots"]), 16))
    max_s = triton.next_power_of_2(max(int(args["num_samples"]), 16))
    keep = [c for c in configs
            if c.kwargs.get("BLOCK_I", 16) <= max_i and c.kwargs.get("BLOCK_S", 16) <= max_s]
    return keep or configs


_AUTOTUNE = dict(key=["num_samples", "num_slots", "latent_dim", "HAS_BIAS", "PRECISION"],
                 prune_configs_by={"early_config_prune": _prune_configs_to_shape})
_FORWARD_CONFIGS = [triton.Config({"BLOCK_S": s, "BLOCK_I": i}, num_warps=w, num_stages=st)
                    for s, i, w, st in [(16, 32, 4, 2), (16, 32, 4, 3), (32, 32, 4, 2),
                                        (32, 32, 8, 2), (32, 32, 8, 3), (64, 32, 8, 1),
                                        (64, 32, 8, 2), (16, 16, 4, 2)]]
_BACKWARD_CONFIGS = [triton.Config({"BLOCK_S": s, "BLOCK_I": i}, num_warps=8, num_stages=st)
                     for s, i, st in [(16, 32, 2), (16, 32, 3), (16, 64, 2), (16, 64, 1),
                                      (32, 32, 2), (16, 16, 3), (32, 16, 1)]]


@triton.jit
def _matmul(a, b, PRECISION: tl.constexpr):
    if PRECISION == "bf16":
        out = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        out = tl.dot(a, b, input_precision=PRECISION)
    return out


@triton.jit
def _load_posterior(mu_ptr, log_sigma_ptr, user, dims, dim_mask, latent_dim):
    offsets = user * latent_dim + dims
    mu = tl.load(mu_ptr + offsets, mask=dim_mask, other=0.0)
    sigma = tl.exp(tl.load(log_sigma_ptr + offsets, mask=dim_mask, other=0.0))
    return mu, sigma


@triton.jit
def _load_eps(eps_ptr, user, samples, sample_mask, dims, dim_mask, num_samples, latent_dim):
    offsets = (user.to(tl.int64) * num_samples + samples[:, None]) * latent_dim + dims[None, :]
    return tl.load(eps_ptr + offsets, mask=sample_mask[:, None] & dim_mask[None, :],
                   other=0.0).to(tl.float32)


@triton.jit
def _online_logsumexp(run_max, run_sum, scores):
    new_max = tl.maximum(run_max, tl.max(scores, axis=1))
    rescale = tl.exp(run_max - new_max)
    return new_max, run_sum * rescale + tl.sum(tl.exp(scores - new_max[:, None]), axis=1)


@triton.autotune(configs=_FORWARD_CONFIGS, **_AUTOTUNE)
@triton.jit
def _train_forward_kernel(
    mu_ptr, log_sigma_ptr, item_table_ptr, option_id_ptr, bias_ptr, target_idx_ptr,
    valid_count_ptr, eps_ptr, log_g_ptr, lse_ptr,
    num_samples, num_slots, total_samples, sample_offset,
    latent_dim: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr, PRECISION: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    blocks_per_user = tl.cdiv(num_samples, BLOCK_S)
    user = pid // blocks_per_user
    samples = (pid % blocks_per_user) * BLOCK_S + tl.arange(0, BLOCK_S)
    sample_mask = samples < num_samples
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < latent_dim
    num_valid = tl.load(valid_count_ptr + user)
    slot_row = user.to(tl.int64) * num_slots
    sample_row = user.to(tl.int64) * total_samples + sample_offset

    mu, sigma = _load_posterior(mu_ptr, log_sigma_ptr, user, dims, dim_mask, latent_dim)
    eps = _load_eps(eps_ptr, user, samples, sample_mask, dims, dim_mask, num_samples, latent_dim)
    x = mu[None, :] + sigma[None, :] * eps

    target = tl.load(target_idx_ptr + user).to(tl.int32)
    target_item = tl.load(option_id_ptr + slot_row + target).to(tl.int64)
    target_vec = tl.load(item_table_ptr + target_item * latent_dim + dims, mask=dim_mask, other=0.0)
    target_score = tl.sum(x * target_vec[None, :], axis=1)
    if HAS_BIAS:
        target_score += tl.load(bias_ptr + slot_row + target)

    run_max = tl.full((BLOCK_S,), float("-inf"), dtype=tl.float32)
    run_sum = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for start in range(0, num_valid, BLOCK_I):
        slots = start + tl.arange(0, BLOCK_I)
        slot_mask = slots < num_valid
        items = tl.load(option_id_ptr + slot_row + slots, mask=slot_mask, other=0).to(tl.int64)
        item_vecs = tl.load(item_table_ptr + items[:, None] * latent_dim + dims[None, :],
                            mask=slot_mask[:, None] & dim_mask[None, :], other=0.0)
        scores = _matmul(x, tl.trans(item_vecs), PRECISION)
        if HAS_BIAS:
            scores += tl.load(bias_ptr + slot_row + slots, mask=slot_mask, other=0.0)[None, :]
        scores = tl.where(slot_mask[None, :], scores, float("-inf"))
        run_max, run_sum = _online_logsumexp(run_max, run_sum, scores)

    lse = run_max + tl.log(run_sum)
    log_g = tl.minimum(target_score - lse, 0.0)
    tl.store(log_g_ptr + sample_row + samples, log_g, mask=sample_mask)
    tl.store(lse_ptr + sample_row + samples, lse, mask=sample_mask)


@triton.autotune(configs=_BACKWARD_CONFIGS, **_AUTOTUNE,
                 reset_to_zero=["grad_mu_ptr", "grad_log_sigma_ptr", "grad_item_table_ptr",
                                "grad_bias_ptr"])
@triton.jit
def _train_backward_kernel(
    mu_ptr, log_sigma_ptr, item_table_ptr, option_id_ptr, bias_ptr, target_idx_ptr,
    valid_count_ptr, eps_ptr, grad_log_g_ptr, lse_ptr,
    grad_mu_ptr, grad_log_sigma_ptr, grad_item_table_ptr, grad_bias_ptr,
    num_samples, num_slots, total_samples, sample_offset,
    latent_dim: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_user = tl.cdiv(num_slots, BLOCK_I)
    user = pid // tiles_per_user
    tile_start = (pid % tiles_per_user) * BLOCK_I
    num_valid = tl.load(valid_count_ptr + user)
    if tile_start >= num_valid:
        return
    slots = tile_start + tl.arange(0, BLOCK_I)
    slot_mask = slots < num_valid
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < latent_dim
    slot_row = user.to(tl.int64) * num_slots
    sample_row = user.to(tl.int64) * total_samples + sample_offset
    target = tl.load(target_idx_ptr + user).to(tl.int32)
    is_target = tl.where(slots == target, 1.0, 0.0)
    mu, sigma = _load_posterior(mu_ptr, log_sigma_ptr, user, dims, dim_mask, latent_dim)

    items = tl.load(option_id_ptr + slot_row + slots, mask=slot_mask, other=0).to(tl.int64)
    item_offsets = items[:, None] * latent_dim + dims[None, :]
    item_mask = slot_mask[:, None] & dim_mask[None, :]
    item_vecs = tl.load(item_table_ptr + item_offsets, mask=item_mask, other=0.0).to(tl.bfloat16)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + slot_row + slots, mask=slot_mask, other=0.0)

    grad_item_vecs = tl.zeros((BLOCK_I, BLOCK_D), dtype=tl.float32)
    grad_bias = tl.zeros((BLOCK_I,), dtype=tl.float32)
    grad_mu = tl.zeros((BLOCK_D,), dtype=tl.float32)
    grad_log_sigma = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start in range(0, num_samples, BLOCK_S):
        samples = start + tl.arange(0, BLOCK_S)
        sample_mask = samples < num_samples
        lse = tl.load(lse_ptr + sample_row + samples, mask=sample_mask, other=0.0)
        grad_log_g = tl.load(grad_log_g_ptr + sample_row + samples, mask=sample_mask, other=0.0)
        eps = _load_eps(eps_ptr, user, samples, sample_mask, dims, dim_mask, num_samples,
                        latent_dim)
        x = (mu[None, :] + sigma[None, :] * eps).to(tl.bfloat16)

        scores = tl.dot(x, tl.trans(item_vecs))
        if HAS_BIAS:
            scores += bias[None, :]
        probs = tl.exp(tl.where(slot_mask[None, :], scores, float("-inf")) - lse[:, None])
        grad_scores = grad_log_g[:, None] * (is_target[None, :] - probs)
        grad_scores = tl.where(slot_mask[None, :] & sample_mask[:, None], grad_scores, 0.0)

        grad_item_vecs += tl.dot(tl.trans(grad_scores.to(tl.bfloat16)), x)
        if HAS_BIAS:
            grad_bias += tl.sum(grad_scores, axis=0)
        grad_x = tl.dot(grad_scores.to(tl.bfloat16), item_vecs)
        grad_mu += tl.sum(grad_x, axis=0)
        grad_log_sigma += tl.sum(grad_x * eps, axis=0)

    tl.atomic_add(grad_item_table_ptr + item_offsets, grad_item_vecs, mask=item_mask)
    if HAS_BIAS:
        grad_bias += tl.load(grad_bias_ptr + slot_row + slots, mask=slot_mask, other=0.0)
        tl.store(grad_bias_ptr + slot_row + slots, grad_bias, mask=slot_mask)
    user_dims = user * latent_dim + dims
    tl.atomic_add(grad_mu_ptr + user_dims, grad_mu, mask=dim_mask)
    tl.atomic_add(grad_log_sigma_ptr + user_dims, grad_log_sigma * sigma, mask=dim_mask)


_SHAPES = dict(mu="BD", log_sigma="BD", item_table="ND", option_id="BI", target_idx="B",
               bias="BI", valid_count="B", eps="BSD", item_embs="BID")
_INDEX_DTYPES = dict(option_id=torch.int32, target_idx=torch.int64, valid_count=torch.int32)
_Inputs = namedtuple("_Inputs", "mu log_sigma item_table option_id target_idx bias valid_count")


def _check_shapes(sizes, **tensors):
    for name, tensor in tensors.items():
        shape = tuple(sizes[dim] for dim in _SHAPES[name])
        if tensor is not None and tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {_SHAPES[name]} = {shape}, "
                             f"got {tuple(tensor.shape)}")


def _prepare(inputs, eps, num_samples):
    (B, D), I, device = inputs.mu.shape, inputs.option_id.shape[-1], inputs.mu.device
    if triton.next_power_of_2(D) < 16:
        raise ValueError(f"latent dim must be >= 9, got {D}")
    if inputs.valid_count is None:
        inputs = inputs._replace(valid_count=torch.full((B,), I, device=device))
    _check_shapes(dict(B=B, D=D, I=I, N=len(inputs.item_table), S=num_samples),
                  eps=eps, **inputs._asdict())
    return _Inputs(**{name: None if t is None else
                      t.to(device, _INDEX_DTYPES.get(name, t.dtype)).contiguous()
                      for name, t in inputs._asdict().items()})


def _launch(kernel, tiled, inputs, eps, total_samples, sample_offset, **kwargs):
    (B, num_samples, D), num_slots = eps.shape, inputs.option_id.shape[1]
    extent, block = (num_samples, "BLOCK_S") if tiled == "samples" else (num_slots, "BLOCK_I")
    args = {f"{name}_ptr": tensor for name, tensor in inputs._asdict().items()} | kwargs
    dummy = torch.empty(1, device=inputs.mu.device)
    kernel[lambda META: (B * triton.cdiv(extent, META[block]),)](
        **{name: dummy if value is None else value for name, value in args.items()},
        eps_ptr=eps.contiguous(), num_samples=num_samples, num_slots=num_slots,
        total_samples=total_samples, sample_offset=sample_offset, latent_dim=D,
        BLOCK_D=triton.next_power_of_2(D), HAS_BIAS=inputs.bias is not None)


def _draw_eps_chunks(mu, num_samples, chunk, seed):
    B, D = mu.shape
    gen = None if seed is None else torch.Generator(device=mu.device).manual_seed(seed)
    buffer = torch.empty(B, chunk, D, device=mu.device, dtype=torch.bfloat16)
    for offset in range(0, num_samples, chunk):
        yield offset, torch.randn(B, chunk, D, generator=gen, out=buffer)


class _TritonMXLLogProbs(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mu, log_sigma, item_table, option_id, bias, target_idx, valid_count,
                num_samples, chunk_samples, eps, precision):
        chunk = chunk_samples or num_samples
        if num_samples % chunk:
            raise ValueError(f"chunk_samples ({chunk}) must divide num_samples ({num_samples})")
        if eps is not None and chunk != num_samples:
            raise ValueError("eps cannot be combined with chunk_samples")
        seed = None if chunk == num_samples else int(torch.randint(0, 2**31 - 1, (1,)).item())
        inputs = _prepare(_Inputs(mu, log_sigma, item_table, option_id, target_idx, bias,
                                  valid_count), eps, num_samples)
        log_g = torch.empty(len(mu), num_samples, device=mu.device, dtype=torch.float32)
        lse = torch.empty_like(log_g)
        chunks = [(0, eps)] if eps is not None else _draw_eps_chunks(mu, num_samples, chunk, seed)
        for offset, chunk_eps in chunks:
            _launch(_train_forward_kernel, "samples", inputs, chunk_eps, num_samples, offset,
                    log_g_ptr=log_g, lse_ptr=lse, PRECISION=precision)
        ctx.save_for_backward(*inputs, lse, chunk_eps if seed is None else None)
        ctx.noise = (num_samples, chunk, seed)
        return log_g

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_log_g):
        *saved, lse, kept_eps = ctx.saved_tensors
        inputs = _Inputs(*saved)
        num_samples, chunk, seed = ctx.noise
        chunks = (((0, kept_eps),) if kept_eps is not None
                  else _draw_eps_chunks(inputs.mu, num_samples, chunk, seed))
        params = (inputs.mu, inputs.log_sigma, inputs.item_table, inputs.bias)
        grads = [None if p is None else torch.zeros_like(p, dtype=torch.float32) for p in params]
        grad_mu, grad_log_sigma, grad_item_table, grad_bias = grads
        grad_log_g = grad_log_g.contiguous()
        for offset, eps in chunks:
            _launch(_train_backward_kernel, "slots", inputs, eps, num_samples, offset,
                    grad_log_g_ptr=grad_log_g, lse_ptr=lse, grad_mu_ptr=grad_mu,
                    grad_log_sigma_ptr=grad_log_sigma, grad_item_table_ptr=grad_item_table,
                    grad_bias_ptr=grad_bias)
        grad_mu, grad_log_sigma, grad_item_table, grad_bias = (
            None if g is None else g.to(p.dtype) for g, p in zip(grads, params))
        return (grad_mu, grad_log_sigma, grad_item_table, None, grad_bias,
                None, None, None, None, None, None)


def triton_mxl_logprobs(mu, log_sigma, item_table, option_id, target_idx, num_samples,
                        bias=None, valid_count=None, chunk_samples=None, eps=None,
                        precision="bf16"):
    if precision not in ("bf16", "tf32", "ieee"):
        raise ValueError(f"precision must be 'bf16', 'tf32' or 'ieee', got {precision!r}")
    return _TritonMXLLogProbs.apply(mu, log_sigma, item_table, option_id, bias, target_idx,
                                    valid_count, num_samples, chunk_samples, eps, precision)
