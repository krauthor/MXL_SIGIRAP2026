import numpy as np
import torch
from typing import Optional


def generate_slate_greedy_batched(
    utilities: np.ndarray,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_items, n_users = utilities.shape
    selected = []
    remaining = np.arange(n_items)

    if no_choice_id is not None:
        remaining = remaining[remaining != no_choice_id]

    exp_no_choice_utils = np.exp(utilities[no_choice_id, :]) if no_choice_id is not None else np.zeros(n_users)

    for _ in range(k):
        if len(selected) > 0:
            selected_exp_sum = np.sum(np.exp(utilities[selected, :]), axis=0)
        else:
            selected_exp_sum = np.zeros(n_users)

        remaining_utils = utilities[remaining, :]

        remaining_exp = np.exp(remaining_utils)

        candidate_exp_sums = remaining_exp + selected_exp_sum[np.newaxis, :]

        candidate_no_choice_probs = exp_no_choice_utils / (exp_no_choice_utils[np.newaxis, :] + candidate_exp_sums)

        if weights is not None:
            candidate_objectives = np.sum(weights * candidate_no_choice_probs, axis=1) / np.sum(weights)
        else:
            candidate_objectives = np.mean(candidate_no_choice_probs, axis=1)

        best_idx = np.argmin(candidate_objectives)
        best_item = remaining[best_idx]

        selected.append(best_item)
        remaining = np.delete(remaining, best_idx)

    if include_no_choice and no_choice_id is not None:
        slate = np.concatenate([[no_choice_id], selected])
    else:
        slate = selected

    return np.array(slate)


def _weighted_mean_torch(
    probs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is None:
        return probs.mean(dim=1)
    return (weights[None, :] * probs).sum(dim=1) / weights.sum()


@torch.no_grad()
def _greedy_batched_torch_core(
    exp_utilities: torch.Tensor,
    k: int,
    exp_no_choice: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    available: torch.Tensor,
) -> torch.Tensor:
    selected = torch.zeros(k, dtype=torch.long, device=exp_utilities.device)
    selected_exp_sum = torch.zeros(
        exp_utilities.shape[1], dtype=exp_utilities.dtype, device=exp_utilities.device
    )
    available = available.clone()

    for i in range(k):
        candidate_exp_sums = exp_utilities + selected_exp_sum[None, :]

        if exp_no_choice is not None:
            no_choice_probs = exp_no_choice[None, :] / (exp_no_choice[None, :] + candidate_exp_sums)
        else:
            no_choice_probs = torch.zeros_like(candidate_exp_sums)

        objectives = _weighted_mean_torch(no_choice_probs, weights)
        objectives = objectives.masked_fill_(~available, float("inf"))

        best_item = torch.argmin(objectives).reshape(1)
        selected[i : i + 1] = best_item
        selected_exp_sum = selected_exp_sum + exp_utilities.index_select(0, best_item).squeeze(0)
        available.index_fill_(0, best_item, False)

    return selected


@torch.no_grad()
def generate_slate_greedy_batched_torch(
    utilities: torch.Tensor,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n_items = utilities.shape[0]
    exp_utilities = torch.exp(utilities)

    exp_no_choice = exp_utilities[no_choice_id] if no_choice_id is not None else None

    available = torch.ones(n_items, dtype=torch.bool, device=utilities.device)
    if no_choice_id is not None:
        available[no_choice_id] = False

    selected = _greedy_batched_torch_core(exp_utilities, k, exp_no_choice, weights, available)

    if include_no_choice and no_choice_id is not None:
        no_choice = torch.tensor([no_choice_id], dtype=torch.long, device=utilities.device)
        selected = torch.cat([no_choice, selected])

    return selected


def generate_slate_top_k_greedy(
    utilities: np.ndarray,
    scores: np.ndarray,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_items, n_users = utilities.shape


    scores_copy = scores.copy()

    if no_choice_id is not None:
        scores_copy[no_choice_id] = -np.inf

    exp_no_choice_utils = np.exp(utilities[no_choice_id, :]) if no_choice_id is not None else np.zeros(n_users)

    sorted_indices = np.argsort(-scores_copy)
    slate = list(sorted_indices[:k])

    slate_set = set(slate)
    remaining = [i for i in range(n_items) if i not in slate_set and i != no_choice_id]


    for pos in range(k):
        if len(remaining) == 0:
            break

        current_item = slate[pos]

        other_slate_items = [slate[i] for i in range(k) if i != pos]

        if len(other_slate_items) > 0:
            other_utils = utilities[other_slate_items, :]
            other_exp_sum = np.sum(np.exp(other_utils), axis=0)
        else:
            other_exp_sum = np.zeros(n_users)

        candidates = np.array(remaining)
        candidate_utils = utilities[candidates, :]
        candidate_exp = np.exp(candidate_utils)

        total_exp_sums = candidate_exp + other_exp_sum[np.newaxis, :]

        no_choice_probs = exp_no_choice_utils / (exp_no_choice_utils + total_exp_sums)

        if weights is not None:
            candidate_objectives = np.sum(weights * no_choice_probs, axis=1) / np.sum(weights)
        else:
            candidate_objectives = np.mean(no_choice_probs, axis=1)

        current_exp = np.exp(utilities[current_item, :])
        current_total_exp = other_exp_sum + current_exp
        current_no_choice_probs = exp_no_choice_utils / (exp_no_choice_utils + current_total_exp)

        if weights is not None:
            current_objective = np.sum(weights * current_no_choice_probs) / np.sum(weights)
        else:
            current_objective = np.mean(current_no_choice_probs)

        best_candidate_idx = np.argmin(candidate_objectives)
        best_candidate_objective = candidate_objectives[best_candidate_idx]

        if best_candidate_objective < current_objective:
            best_replacement = candidates[best_candidate_idx]

            old_item = slate[pos]
            remaining.append(old_item)

            slate[pos] = best_replacement
            remaining.remove(best_replacement)

            slate_set.discard(old_item)
            slate_set.add(best_replacement)


    slate_array = np.array(slate)

    if include_no_choice and no_choice_id is not None:
        slate_array = np.concatenate([[no_choice_id], slate_array])

    return slate_array


@torch.no_grad()
def _top_k_greedy_torch_core(
    exp_utilities: torch.Tensor,
    initial_slate: torch.Tensor,
    k: int,
    exp_no_choice: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    all_item_mask: torch.Tensor,
) -> torch.Tensor:
    n_items = exp_utilities.shape[0]
    slate = initial_slate.clone()

    for pos in range(k):
        cur = slate[pos : pos + 1]
        slate_mask = torch.zeros(n_items, dtype=torch.bool, device=exp_utilities.device)
        slate_mask.index_fill_(0, slate, True)
        slate_mask.index_fill_(0, cur, False)

        other_exp_sum = (exp_utilities * slate_mask[:, None]).sum(dim=0)
        candidate_exp_sums = exp_utilities + other_exp_sum[None, :]

        if exp_no_choice is not None:
            no_choice_probs = exp_no_choice[None, :] / (exp_no_choice[None, :] + candidate_exp_sums)
        else:
            no_choice_probs = torch.zeros_like(candidate_exp_sums)

        objectives = _weighted_mean_torch(no_choice_probs, weights)
        current_obj = objectives.index_select(0, cur)

        candidate_mask = all_item_mask & (~slate_mask)
        objectives = objectives.masked_fill_(~candidate_mask, float("inf"))

        best_item = torch.argmin(objectives).reshape(1)
        best_obj = objectives.index_select(0, best_item)
        slate[pos : pos + 1] = torch.where(best_obj < current_obj, best_item, cur)

    return slate


@torch.no_grad()
def generate_slate_top_k_greedy_torch(
    utilities: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n_items = utilities.shape[0]
    exp_utilities = torch.exp(utilities)

    exp_no_choice = exp_utilities[no_choice_id] if no_choice_id is not None else None

    all_item_mask = torch.ones(n_items, dtype=torch.bool, device=utilities.device)
    if no_choice_id is not None:
        all_item_mask[no_choice_id] = False

    scores_masked = scores
    if no_choice_id is not None:
        scores_masked = scores.clone()
        scores_masked[no_choice_id] = -float("inf")
    initial_slate = torch.topk(scores_masked, k).indices

    selected = _top_k_greedy_torch_core(
        exp_utilities, initial_slate, k, exp_no_choice, weights, all_item_mask
    )

    if include_no_choice and no_choice_id is not None:
        no_choice = torch.tensor([no_choice_id], dtype=torch.long, device=utilities.device)
        selected = torch.cat([no_choice, selected])

    return selected


def generate_slate_top_k(
    scores: np.ndarray,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
) -> np.ndarray:
    scores = scores.copy()
    if no_choice_id is not None:
        scores[no_choice_id] = -np.inf

    top_k_indices = np.argsort(-scores)[:k]

    if include_no_choice and no_choice_id is not None:
        slate = np.concatenate([[no_choice_id], top_k_indices])
    else:
        slate = top_k_indices

    return slate


@torch.no_grad()
def generate_slate_top_k_torch(
    scores: torch.Tensor,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
) -> torch.Tensor:
    if no_choice_id is not None:
        scores = scores.clone()
        scores[no_choice_id] = -float("inf")

    top_k_indices = torch.topk(scores, k).indices

    if include_no_choice and no_choice_id is not None:
        no_choice = torch.tensor([no_choice_id], dtype=torch.long, device=scores.device)
        top_k_indices = torch.cat([no_choice, top_k_indices])

    return top_k_indices


def generate_slate_sample_probability(
    probs: np.ndarray,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
    rng: np.random.Generator = None,
) -> np.ndarray:

    probs = probs.copy()
    if no_choice_id is not None:
        probs[no_choice_id] = 0.0
        probs = probs / probs.sum()

    slate = rng.choice(len(probs), size=k, replace=False, p=probs)

    if include_no_choice and no_choice_id is not None:
        slate = np.concatenate([[no_choice_id], slate])

    return slate


@torch.no_grad()
def generate_slate_topk_avg_choice_prob_torch(
    scores: torch.Tensor,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
) -> torch.Tensor:
    avg_choice_prob = torch.softmax(scores, dim=0).mean(dim=1)
    return generate_slate_top_k_torch(
        scores=avg_choice_prob,
        k=k,
        include_no_choice=include_no_choice,
        no_choice_id=no_choice_id,
    )


def generate_slate_top1_per_sample(
    scores: np.ndarray,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
    sample_probs: Optional[np.ndarray] = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    n_items, num_samples = scores.shape

    scores = scores.copy()
    if no_choice_id is not None:
        scores[no_choice_id, :] = -np.inf

    if sample_probs is not None:
        sample_probs = sample_probs / sample_probs.sum()

    sampled_columns = rng.choice(num_samples, size=k, replace=True, p=sample_probs)

    selected_items = []

    for col_idx in sampled_columns:
        sample_scores = scores[:, col_idx].copy()

        for item_id in selected_items:
            sample_scores[item_id] = -1e10

        top_item = np.argmax(sample_scores)
        selected_items.append(int(top_item))

    slate = np.array(selected_items, dtype=np.int32)

    if include_no_choice and no_choice_id is not None:
        slate = np.concatenate([[no_choice_id], slate])

    return slate


@torch.no_grad()
def _top1_per_sample_torch_core(
    scores: torch.Tensor,
    k: int,
    sampled_columns: torch.Tensor,
) -> torch.Tensor:
    n_items = scores.shape[0]
    selected = torch.zeros(k, dtype=torch.long, device=scores.device)
    mask = torch.zeros(n_items, dtype=torch.bool, device=scores.device)

    for i in range(k):
        col_scores = scores.index_select(1, sampled_columns[i : i + 1]).squeeze(1)
        col_scores = col_scores.masked_fill_(mask, -float("inf"))
        top_item = torch.argmax(col_scores).reshape(1)
        selected[i : i + 1] = top_item
        mask.index_fill_(0, top_item, True)

    return selected


@torch.no_grad()
def generate_slate_top1_per_sample_torch(
    scores: torch.Tensor,
    k: int,
    include_no_choice: bool = True,
    no_choice_id: Optional[int] = None,
    sample_probs: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    n_items, n_samples = scores.shape

    if no_choice_id is not None:
        scores = scores.clone()
        scores[no_choice_id, :] = -float("inf")

    if sample_probs is not None:
        sampled_columns = torch.multinomial(
            sample_probs.to(dtype=scores.dtype), k, replacement=True, generator=generator
        )
    else:
        sampled_columns = torch.randint(
            0, n_samples, (k,), device=scores.device, generator=generator
        )

    selected = _top1_per_sample_torch_core(scores, k, sampled_columns)

    if include_no_choice and no_choice_id is not None:
        no_choice = torch.tensor([no_choice_id], dtype=torch.long, device=scores.device)
        selected = torch.cat([no_choice, selected])

    return selected
