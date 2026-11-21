import torch


def var(samples, num_samples_per_kernel):
    return (
        torch.sum(
            torch.square(
                samples - torch.mean(samples, dim=-1, keepdim=True)
            ),
            dim=-1,
        )
        / num_samples_per_kernel
        / (num_samples_per_kernel - 1)
    )


def var3(samples, num_samples_per_kernel):
    return (
        torch.sum(
            torch.pow(samples - torch.mean(samples, dim=-1, keepdim=True), 3),
            dim=-1,
        )
        / num_samples_per_kernel
        / (num_samples_per_kernel - 1)
        / (num_samples_per_kernel - 2)
    )


def unbias_loglikelihood(ll, samples, num_samples_per_kernel):
    return (
        torch.squeeze(ll)
        + 1 / 2 * var(samples, num_samples_per_kernel) / (torch.square(samples.mean(dim=-1)) + 1e-10)
        - 1 / 3 * var3(samples, num_samples_per_kernel) / (torch.pow(samples.mean(dim=-1), 3) + 1e-10)
    )


def log_likelihood(probs, labels, logits):
    ll = torch.sum(labels * torch.log(probs), dim=-1)

    logits_max = torch.max(logits, dim=-2, keepdim=True).values
    log_probs_per_kernel = logits - logits_max - torch.log(
        torch.sum(torch.exp(logits - logits_max), dim=-2, keepdim=True)
    )
    probs_per_kernel = torch.exp(log_probs_per_kernel)
    probs_per_kernel = (labels[:, :, None] * probs_per_kernel).sum(dim=1)

    unbiased_ll = unbias_loglikelihood(ll[:, None], probs_per_kernel, probs_per_kernel.shape[1])

    return unbiased_ll


def kl_gaussian(user_mean, user_std):
    user_var = torch.square(user_std)
    return -0.5 * torch.sum(1.0 + torch.log(user_var) - user_var - torch.square(user_mean), dim=-1)


def kl_gaussian_aat(user_mean, A, log_sigma):
    D = user_mean.shape[-1]
    sigma_sq = torch.square(torch.exp(log_sigma))

    trace_term = torch.sum(A ** 2, dim=(-2, -1)) + torch.sum(sigma_sq, dim=-1)
    mean_term = torch.sum(user_mean ** 2, dim=-1)

    inv_sigma_sq = 1.0 / sigma_sq
    A_scaled = A * inv_sigma_sq[..., None]
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    inner = eye + torch.einsum('...ji,...jk->...ik', A, A_scaled)
    logdet_inner = torch.linalg.slogdet(inner).logabsdet
    log_det_term = 2.0 * torch.sum(log_sigma, dim=-1) + logdet_inner

    return 0.5 * (trace_term + mean_term - D - log_det_term)


def elbo(
    logits,
    probs,
    user_embs,
    model,
    labels_onehot,
    beta=1,
    l2_weight=0,
    precomputed_kl=None,
    return_component_values=False,
):
    seq_backbone = getattr(model, "user_backbone", "id") != "id"

    if seq_backbone:
        item_reg = torch.zeros(model.num_items, device=logits.device)
        user_reg = torch.zeros(1, device=logits.device)
        param_reg = l2_weight * model.model_param_l2() if l2_weight > 0 else torch.zeros((), device=logits.device)
    elif l2_weight > 0:
        item_idx = torch.arange(model.num_items, device=logits.device)
        item_reg = l2_weight * torch.square(model.get_item_embeddings(item_idx)).sum(dim=-1)

        user_reg = l2_weight * torch.square(user_embs).sum(dim=-1).mean(dim=-1)
        param_reg = torch.zeros((), device=logits.device)
    else:
        item_reg = torch.zeros(model.num_items, device=logits.device)
        user_reg = torch.zeros(1, device=logits.device)
        param_reg = torch.zeros((), device=logits.device)

    ll = log_likelihood(probs=probs, labels=labels_onehot, logits=logits)

    kl = precomputed_kl if precomputed_kl is not None else model.kl_divergence()

    elbo_val = ll.mean() - beta * kl.mean()

    elbo_val = elbo_val - (item_reg.mean() + user_reg.mean()) - param_reg

    loss = -torch.mean(elbo_val)

    if return_component_values:
        return loss, ll, kl, logits, probs, user_reg, item_reg
    else:
        return loss
