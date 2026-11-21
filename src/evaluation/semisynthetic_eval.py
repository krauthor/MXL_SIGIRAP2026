import os
import numpy as np
import torch
from typing import Dict, Any, Optional
from tqdm.auto import tqdm

from src.utils.slate_generation import (
    generate_slate_top_k,
    generate_slate_top_k_greedy,
    generate_slate_greedy_batched,
    generate_slate_top1_per_sample,
)
from src.utils.slate_metrics import (
    compute_slate_item_coverage,
    compute_slate_diversity,
)
from src.data.dataset import build_user_histories, build_no_choice_rate
from src.data.generate_synthetic_from_model import _sample_choice_from_slate_numpy
from src.utils.load_existing_model import load_trained_model


_ALL_POLICIES = ["top_k", "top_k_greedy", "greedy", "diverse_embeddings"]
_POLICY_MAP = {
    "MXL": _ALL_POLICIES, "MXL_SASRec": _ALL_POLICIES,
    "MNL_Multi": _ALL_POLICIES, "MNL_Multi_SASRec": _ALL_POLICIES,
    "MNL": ["top_k"], "MNL_SASRec": ["top_k"],
    "Intercept": ["top_k"],
    "MNL_Max": ["top_k", "diverse_embeddings"],
    "MNL_Max_SASRec": ["top_k", "diverse_embeddings"],
}


def get_policies_for_model(model_name: str):
    return _POLICY_MAP.get(model_name)


def _forward_scores(model, input_dict):
    out = model.forward(input_dict)
    return out[0], out[1]


def _build_model_inputs(model, split_arrays, sl, device):
    users, option_row, histories, rates = split_arrays
    B = sl.stop - sl.start
    inp = {"option_id": option_row.expand(B, -1)}
    if getattr(model, "user_backbone", "id") != "id":
        max_seq_len = model.max_seq_len
        inp["history"] = torch.from_numpy(histories[max_seq_len][sl]).to(device)
        if getattr(model, "use_no_choice_rate", False):
            inp["no_choice_rate"] = torch.from_numpy(rates[sl]).float().to(device)
    else:
        inp["user_ids"] = torch.from_numpy(users[sl].astype(np.int64)).to(device)
    return inp


def _seen_items_for_split(users_all, choices_all, split_start, split_len, no_choice_id, padding_id):
    from collections import defaultdict
    seen = defaultdict(list)
    out = [None] * split_len
    end = split_start + split_len
    for j in range(len(users_all)):
        u = int(users_all[j])
        if split_start <= j < end:
            out[j - split_start] = np.asarray(seen[u], dtype=np.int64)
        c = int(choices_all[j])
        if c != no_choice_id and c != padding_id:
            seen[u].append(c)
    return out


def _generate_slate_for_policy(
    policy: str,
    trained_utilities: np.ndarray,
    trained_probs: np.ndarray,
    user_weights: Optional[np.ndarray],
    per_emb_utils: Optional[np.ndarray],
    slate_size: int,
    no_choice_id: Optional[int],
    rng_np: np.random.Generator,
) -> np.ndarray:
    include_no_choice = no_choice_id is not None

    if policy == "top_k":
        return generate_slate_top_k(
            scores=trained_probs, k=slate_size,
            include_no_choice=include_no_choice, no_choice_id=no_choice_id,
        )
    elif policy == "diverse_embeddings":
        scores = per_emb_utils if per_emb_utils is not None else trained_utilities
        return generate_slate_top1_per_sample(
            scores=scores, k=slate_size,
            include_no_choice=include_no_choice, no_choice_id=no_choice_id,
            sample_probs=user_weights, rng=rng_np,
        )
    elif policy == "greedy":
        return generate_slate_greedy_batched(
            utilities=trained_utilities, k=slate_size,
            include_no_choice=include_no_choice, no_choice_id=no_choice_id,
            weights=user_weights,
        )
    elif policy == "top_k_greedy":
        return generate_slate_top_k_greedy(
            utilities=trained_utilities, scores=trained_probs, k=slate_size,
            include_no_choice=include_no_choice, no_choice_id=no_choice_id,
            weights=user_weights,
        )
    else:
        raise ValueError(f"Invalid slate_generation_policy: {policy}")


def evaluate_semisynthetic_metrics(
    model,
    model_name: str,
    data: Dict[str, Any],
    synthetic_metadata: Dict[str, Any],
    device,
    no_choice_id: Optional[int],
    padding_id: Optional[int],
    never_chosen_id: Optional[int],
    split: str = "test",
    slate_size: int = 4,
    num_eval_samples: Optional[int] = None,
    batch_size: int = 128,
    seed: Optional[int] = None,
    slate_num_prior_samples: Optional[int] = 128,
    num_random_diversity_tuples: int = 10_000,
    exclude_seen: bool = False,
    fallback_source_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    policies_to_evaluate = get_policies_for_model(model_name)
    if policies_to_evaluate is None:
        print(f"Semi-synthetic slate evaluation is not supported for "
              f"model '{model_name}' — skipping.")
        return {}

    if seed is None:
        seed = 0

    source_model_saved_path = synthetic_metadata.get("source_model_saved_path")
    if not (source_model_saved_path and os.path.exists(source_model_saved_path)):
        if fallback_source_dir and os.path.exists(fallback_source_dir):
            source_model_saved_path = fallback_source_dir
        else:
            raise ValueError(
                f"Source model not found (saved path: {source_model_saved_path}, "
                f"fallback: {fallback_source_dir}). Synthetic data must include a "
                f"saved source model."
            )
    if verbose:
        print(f"Loading source model from: {source_model_saved_path}")
    oracle, _, oracle_metadata = load_trained_model(
        source_model_saved_path,
        no_choice_id=no_choice_id, padding_id=padding_id, never_chosen_id=never_chosen_id,
    )
    oracle.to(device)
    oracle.eval()
    source_model_name = oracle_metadata.get("model_name", synthetic_metadata.get("source_model_name"))

    for attr, key in [("temperature", "temperature"),
                      ("no_choice_boost", "no_choice_utility_boost"),
                      ("intercept_scale", "intercept_scale")]:
        val = synthetic_metadata.get(key)
        if val is not None:
            setattr(oracle, attr, float(val))
    if verbose:
        print(f"Oracle simulation params (from generation metadata): "
              f"temperature={getattr(oracle, 'temperature', None)}, "
              f"no_choice_boost={getattr(oracle, 'no_choice_boost', None)}, "
              f"intercept_scale={getattr(oracle, 'intercept_scale', None)}")

    model.eval()

    saved_num_prior = {}
    if slate_num_prior_samples is not None:
        for m in (model, oracle):
            if hasattr(m, "num_prior_samples"):
                saved_num_prior[id(m)] = m.num_prior_samples
                m.num_prior_samples = min(m.num_prior_samples, slate_num_prior_samples)

    try:
        return _evaluate(
            model=model, model_name=model_name, oracle=oracle,
            source_model_name=source_model_name,
            policies_to_evaluate=policies_to_evaluate,
            data=data, synthetic_metadata=synthetic_metadata, device=device,
            no_choice_id=no_choice_id, padding_id=padding_id,
            split=split, slate_size=slate_size, num_eval_samples=num_eval_samples,
            batch_size=batch_size, seed=seed,
            num_random_diversity_tuples=num_random_diversity_tuples,
            exclude_seen=exclude_seen,
            never_chosen_id=never_chosen_id, verbose=verbose,
        )
    finally:
        for m in (model, oracle):
            if id(m) in saved_num_prior:
                m.num_prior_samples = saved_num_prior[id(m)]


def _evaluate(
    model, model_name, oracle, source_model_name, policies_to_evaluate,
    data, synthetic_metadata, device, no_choice_id, padding_id,
    split, slate_size, num_eval_samples, batch_size, seed,
    num_random_diversity_tuples, exclude_seen=False, never_chosen_id=None, verbose=True,
):
    num_items = data["max_item_id"] + 1
    include_no_choice = no_choice_id is not None

    users_all = np.concatenate([data[s]["user"] for s in ("train", "val", "test")])
    choices_all = np.concatenate([
        data[s]["options"][np.arange(len(data[s]["choices_pos"])), data[s]["choices_pos"]]
        for s in ("train", "val", "test")
    ])
    split_starts = {"train": 0, "val": len(data["train"]["user"]),
                    "test": len(data["train"]["user"]) + len(data["val"]["user"])}
    split_start = split_starts[split]
    split_len = len(data[split]["user"])

    needed_seq_lens = set()
    needs_rate = False
    for m in (model, oracle):
        if getattr(m, "user_backbone", "id") != "id":
            needed_seq_lens.add(m.max_seq_len)
            needs_rate = needs_rate or getattr(m, "use_no_choice_rate", False)

    histories = {}
    for msl in needed_seq_lens:
        full = build_user_histories(
            users_all, choices_all, msl,
            exclude_ids=(no_choice_id,), pad_id=padding_id if padding_id is not None else 0,
        )
        histories[msl] = full[split_start:split_start + split_len]
    rates = None
    if needs_rate and no_choice_id is not None:
        rates = build_no_choice_rate(users_all, choices_all, no_choice_id)[
            split_start:split_start + split_len]

    seen_split = None
    if exclude_seen:
        seen_split = _seen_items_for_split(
            users_all, choices_all, split_start, split_len, no_choice_id, padding_id)

    users_split = data[split]["user"]
    num_obs = split_len if num_eval_samples is None else min(split_len, num_eval_samples)

    is_mnl_multi = model_name in ("MNL_Multi", "MNL_Multi_SASRec")
    is_mnl_max = model_name in ("MNL_Max", "MNL_Max_SASRec")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Semi-Synthetic Evaluation - {split} set ({num_obs} observations)")
        print(f"{'='*60}")
        print(f"Slate size: {slate_size}")
        print(f"Candidate pool: full catalog ({num_items} items)")
        if exclude_seen:
            print("Excluding already-consumed items from slates (no-repeat eval)")
        print(f"Synthetic generation policy: {synthetic_metadata.get('slate_policy', 'unknown')}")
        print(f"Source model (choice simulation): {source_model_name}")
        print(f"Trained model (slate generation): {model_name}")
        print(f"Slate generation policies to evaluate: {policies_to_evaluate}")
        print(f"{'='*60}\n")

    with torch.no_grad():
        source_item_embeddings = oracle.get_item_embeddings(
            torch.arange(num_items, dtype=torch.long, device=device)
        ).detach().cpu().numpy()
    X_norm = source_item_embeddings / (np.linalg.norm(source_item_embeddings, axis=1, keepdims=True) + 1e-10)
    sim_matrix = X_norm @ X_norm.T
    rng_baseline = np.random.default_rng(seed)
    random_item_tuples_sims = []
    for _ in range(num_random_diversity_tuples):
        idxs = rng_baseline.choice(sim_matrix.shape[0], size=slate_size, replace=False)
        pairwise_sims = sim_matrix[np.ix_(idxs, idxs)]
        triu_indices = np.triu_indices_from(pairwise_sims, k=1)
        random_item_tuples_sims.append(pairwise_sims[triu_indices].mean())

    policy_data = {
        policy: {"generated_slates": [], "acceptance_flags": []}
        for policy in policies_to_evaluate
    }
    policy_rngs_np = {policy: np.random.default_rng(seed) for policy in policies_to_evaluate}

    option_row = torch.arange(num_items, dtype=torch.long, device=device)
    split_arrays = (users_split, option_row, histories, rates)

    batch_iterator = range(0, num_obs, batch_size)
    if verbose:
        batch_iterator = tqdm(batch_iterator, desc="Evaluating observations (all policies)",
                              total=(num_obs + batch_size - 1) // batch_size)

    for batch_start in batch_iterator:
        sl = slice(batch_start, min(batch_start + batch_size, num_obs))
        B = sl.stop - sl.start

        with torch.no_grad():
            trained_inp = _build_model_inputs(model, split_arrays, sl, device)
            trained_utils, trained_probs = _forward_scores(model, trained_inp)

            oracle_inp = _build_model_inputs(oracle, split_arrays, sl, device)
            oracle_utils, _ = _forward_scores(oracle, oracle_inp)

            per_emb_utils_batch = None
            if is_mnl_max:
                per_emb_utils_batch, _ = model.call_each_emb_separately(trained_inp)
            user_weights_batch = None
            if is_mnl_multi:
                user_weights_batch = model.get_user_embedding_weights(trained_inp)

        trained_utils = trained_utils.detach().cpu().numpy()
        trained_probs = trained_probs.detach().cpu().numpy()
        oracle_utils = oracle_utils.detach().cpu().numpy()
        if per_emb_utils_batch is not None:
            per_emb_utils_batch = per_emb_utils_batch.detach().cpu().numpy()
        if user_weights_batch is not None:
            user_weights_batch = user_weights_batch.detach().cpu().numpy()

        for i in range(B):
            obs_trained_utils = trained_utils[i]
            obs_trained_probs = trained_probs[i]
            obs_oracle_utils = oracle_utils[i]
            obs_weights = user_weights_batch[i] if user_weights_batch is not None else None
            obs_per_emb_utils = per_emb_utils_batch[i] if per_emb_utils_batch is not None else None

            if seen_split is not None:
                seen = seen_split[sl.start + i]
                if seen.size:
                    obs_trained_utils = obs_trained_utils.copy(); obs_trained_utils[seen] = -1e9
                    obs_trained_probs = obs_trained_probs.copy(); obs_trained_probs[seen] = -np.inf
                    if obs_per_emb_utils is not None:
                        obs_per_emb_utils = obs_per_emb_utils.copy(); obs_per_emb_utils[seen] = -1e9

            for policy in policies_to_evaluate:
                rng_np = policy_rngs_np[policy]

                slate = _generate_slate_for_policy(
                    policy=policy,
                    trained_utilities=obs_trained_utils,
                    trained_probs=obs_trained_probs,
                    user_weights=obs_weights,
                    per_emb_utils=obs_per_emb_utils,
                    slate_size=slate_size,
                    no_choice_id=no_choice_id,
                    rng_np=rng_np,
                )
                slate = np.asarray(slate, dtype=np.int64)
                policy_data[policy]["generated_slates"].append(slate)

                _, choice_item = _sample_choice_from_slate_numpy(
                    slate=slate,
                    user_utilities=obs_oracle_utils,
                    rng=rng_np,
                )
                if include_no_choice:
                    policy_data[policy]["acceptance_flags"].append(choice_item != no_choice_id)
                else:
                    policy_data[policy]["acceptance_flags"].append(True)

    all_results = {}
    for policy in policies_to_evaluate:
        generated_slates = np.array(policy_data[policy]["generated_slates"])
        acceptance_flags = np.array(policy_data[policy]["acceptance_flags"])

        metrics = {}
        metrics['slate_acceptance_rate'] = float(np.mean(acceptance_flags))
        metrics['item_coverage'] = compute_slate_item_coverage(
            slates=generated_slates, total_items=num_items,
            padding_id=padding_id, no_choice_id=no_choice_id,
        )
        diversity_metrics = compute_slate_diversity(
            slates=generated_slates,
            item_embeddings=source_item_embeddings,
            padding_id=padding_id, no_choice_id=no_choice_id,
            random_item_tuples_sims=random_item_tuples_sims,
        )
        metrics.update({k: float(v) for k, v in diversity_metrics.items()})

        valid_slate_sizes = []
        for slate in generated_slates:
            valid_items = slate
            if padding_id is not None:
                valid_items = valid_items[valid_items != padding_id]
            if include_no_choice:
                valid_items = valid_items[valid_items != no_choice_id]
            valid_slate_sizes.append(len(valid_items))
        metrics['avg_slate_size'] = float(np.mean(valid_slate_sizes))

        all_results[policy] = metrics

        if verbose:
            print(f"\nResults for {policy}:")
            print(f"  Slate Acceptance Rate: {metrics['slate_acceptance_rate']:.4f}")
            print(f"  Item Coverage: {metrics['item_coverage']:.4f}")
            if 'embedding_diversity_percentile' in metrics:
                print(f"  Embedding Diversity Percentile vs Random: {metrics['embedding_diversity_percentile']:.2f}")
            print(f"  Average Slate Size: {metrics['avg_slate_size']:.2f}")

    return all_results
