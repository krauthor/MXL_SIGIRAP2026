import os
import sys
import json
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Literal, Dict, Any, Tuple, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)).split("src")[0])

from src.utils.slate_generation import generate_slate_sample_probability

SlatePolicy = Literal["sample_probability"]


def _forward_scores(model, input_dict) -> Tuple[torch.Tensor, torch.Tensor]:
    out = model.forward(input_dict)
    return out[0], out[1]


def _sample_choice_from_slate_numpy(
    slate: np.ndarray,
    user_utilities: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    if user_utilities.ndim > 1:
        num_samples = user_utilities.shape[-1]
        sample_dim = rng.choice(num_samples)
        user_utilities = user_utilities[:, sample_dim]

    utilities_for_choice = user_utilities[slate].copy()

    gumbel_noise = rng.gumbel(size=utilities_for_choice.shape)
    choice_pos = np.argmax(utilities_for_choice + gumbel_noise)
    choice = slate[choice_pos]

    return int(choice_pos), int(choice)


def _load_base_dataset(base_dataset: str, sample_limit: Optional[int], verbose: bool) -> Dict[str, Any]:
    if base_dataset == "ContentWise_LastActionSplit":
        from src.data.load_data import load_contentwise_last_action_split_data
        if verbose:
            print("Loading original ContentWise (LastActionSplit) data for structure...")
        return load_contentwise_last_action_split_data(sample_limit=sample_limit)
    elif base_dataset == "Stockholm_LastActionSplit":
        from src.data.load_data import load_stockholm_last_action_split_data
        if verbose:
            print("Loading original Stockholm (LastActionSplit) data for structure...")
        return load_stockholm_last_action_split_data(sample_limit=sample_limit)
    else:
        raise ValueError(
            f"Unknown base dataset: {base_dataset}. Supported: "
            f"'ContentWise_LastActionSplit', 'Stockholm_LastActionSplit'"
        )


def _rollout_seq(
    model,
    original_users: np.ndarray,
    original_options: np.ndarray,
    num_items: int,
    slate_size: int,
    no_choice_id: Optional[int],
    padding_id: Optional[int],
    allow_repeat_items: bool,
    seed: int,
    batch_size: int,
    device: torch.device,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    num_observations = len(original_users)
    slate_width = original_options.shape[1]
    include_no_choice = no_choice_id is not None
    max_seq_len = model.max_seq_len

    sorted_idx = np.argsort(original_users, kind="stable")
    _, counts = np.unique(original_users[sorted_idx], return_counts=True)
    group_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    num_groups = len(counts)
    max_steps = int(counts.max())

    hist_state = [[] for _ in range(num_groups)]
    nc_total = np.zeros(num_groups, dtype=np.int64)
    nc_count = np.zeros(num_groups, dtype=np.int64)
    excluded = [set() for _ in range(num_groups)] if not allow_repeat_items else None

    synthetic_options = np.zeros_like(original_options)
    synthetic_choices_pos = np.zeros(num_observations, dtype=np.int32)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_items_row = torch.arange(num_items, dtype=torch.long, device=device)
    use_rate = getattr(model, "use_no_choice_rate", False)

    progress = tqdm(total=num_observations, desc="Rollout (sequence)", disable=not verbose)
    for t in range(max_steps):
        active = np.nonzero(counts > t)[0]
        for cs in range(0, len(active), batch_size):
            chunk = active[cs:cs + batch_size]
            B = len(chunk)
            obs_idx = sorted_idx[group_start[chunk] + t]

            hist = np.zeros((B, max_seq_len), dtype=np.int64)
            for i, g in enumerate(chunk):
                h = hist_state[g][-max_seq_len:]
                if h:
                    hist[i, :len(h)] = h
            inp = {
                "option_id": all_items_row.expand(B, -1),
                "history": torch.from_numpy(hist).to(device),
            }
            if use_rate:
                rate = np.where(nc_total[chunk] > 0, nc_count[chunk] / np.maximum(nc_total[chunk], 1), 0.0)
                inp["no_choice_rate"] = torch.from_numpy(rate).float().to(device)

            with torch.no_grad():
                utils, probs = _forward_scores(model, inp)
            utils = utils.detach().cpu()
            probs = probs.detach().cpu()

            excl_mask = None
            if not allow_repeat_items:
                excl_rows = [i for i, g in enumerate(chunk) if excluded[g]]
                if excl_rows:
                    excl_mask = torch.zeros((B, num_items), dtype=torch.bool)
                    for i in excl_rows:
                        excl_mask[i, list(excluded[chunk[i]])] = True
                    utils = utils.masked_fill(excl_mask.unsqueeze(-1), -1e4)
                    probs = torch.softmax(utils, dim=1).mean(dim=-1)

            scores = probs.clone()
            if padding_id is not None:
                scores[:, padding_id] = 0.0
            if include_no_choice:
                scores[:, no_choice_id] = 0.0
            if excl_mask is not None:
                scores = scores.masked_fill(excl_mask, 0.0)

            slate_items = torch.multinomial(scores.double(), slate_size, replacement=False, generator=gen)

            if include_no_choice:
                slate = torch.cat(
                    [torch.full((B, 1), no_choice_id, dtype=torch.long), slate_items], dim=1
                )
            else:
                slate = slate_items

            if slate.shape[1] > slate_width:
                slate = slate[:, :slate_width]
            elif slate.shape[1] < slate_width:
                pad_cols = torch.full((B, slate_width - slate.shape[1]), padding_id, dtype=torch.long)
                slate = torch.cat([slate, pad_cols], dim=1)
            orig_rows = torch.from_numpy(original_options[obs_idx].astype(np.int64))
            if padding_id is not None:
                slate = torch.where(orig_rows == padding_id, torch.full_like(slate, padding_id), slate)

            num_samples = utils.shape[-1]
            if num_samples > 1:
                col = torch.randint(num_samples, (B,), generator=gen)
                u = utils[torch.arange(B), :, col]
            else:
                u = utils.squeeze(-1)
            slate_utils = u.gather(1, slate)
            if padding_id is not None:
                slate_utils = slate_utils.masked_fill(slate == padding_id, -1e9)
            uniform = torch.rand(slate_utils.shape, generator=gen)
            exponential = -torch.log(uniform.clamp_min(1e-12))
            gumbel = -torch.log(exponential.clamp_min(1e-12))
            noisy_utils = slate_utils + gumbel
            if not torch.isfinite(noisy_utils).all():
                raise ValueError("Non-finite utilities/noise during choice sampling.")
            pos = torch.argmax(noisy_utils, dim=1)
            choice = slate.gather(1, pos.unsqueeze(1)).squeeze(1)

            synthetic_options[obs_idx] = slate.numpy()
            synthetic_choices_pos[obs_idx] = pos.numpy().astype(np.int32)
            for i, g in enumerate(chunk):
                c = int(choice[i])
                nc_total[g] += 1
                if include_no_choice and c == no_choice_id:
                    nc_count[g] += 1
                elif padding_id is not None and c == padding_id:
                    raise ValueError("Chosen item is padding, which should not happen.")
                else:
                    hist_state[g].append(c)
                    if not allow_repeat_items:
                        excluded[g].add(c)
            progress.update(B)
    progress.close()

    return synthetic_options, synthetic_choices_pos


def _generate_id_backbone(
    model,
    original_users: np.ndarray,
    original_options: np.ndarray,
    num_items: int,
    slate_size: int,
    no_choice_id: Optional[int],
    padding_id: Optional[int],
    allow_repeat_items: bool,
    seed: int,
    batch_size: int,
    device: torch.device,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    num_observations = len(original_users)
    include_no_choice = no_choice_id is not None

    unique_users = np.unique(original_users)
    user_id_to_index = {int(uid): idx for idx, uid in enumerate(unique_users)}
    num_unique_users = len(unique_users)

    all_items = torch.arange(num_items, dtype=torch.long, device=device)

    user_utilities = None
    user_probs = np.zeros((num_unique_users, num_items), dtype=np.float32)

    batch_iterator = range(0, num_unique_users, batch_size)
    if verbose:
        print(f"Pre-computing utilities for {num_unique_users} unique users...")
        batch_iterator = tqdm(batch_iterator, desc="Computing user utilities")

    for batch_start in batch_iterator:
        batch_user_ids = unique_users[batch_start:batch_start + batch_size]
        B = len(batch_user_ids)
        inp = {
            "option_id": all_items.expand(B, -1),
            "user_ids": torch.from_numpy(batch_user_ids.astype(np.int64)).to(device),
        }
        with torch.no_grad():
            utils, probs = _forward_scores(model, inp)
        utils = utils.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()

        if user_utilities is None:
            num_samples = utils.shape[-1]
            user_utilities = np.zeros((num_unique_users, num_items, num_samples), dtype=np.float32)
        for i, uid in enumerate(batch_user_ids):
            user_idx = user_id_to_index[int(uid)]
            user_utilities[user_idx] = utils[i]
            user_probs[user_idx] = probs[i]

    is_mxl = user_utilities.shape[-1] > 1

    user_chosen_items = {int(uid): set() for uid in unique_users}

    synthetic_options = np.zeros_like(original_options)
    synthetic_choices_pos = np.zeros(num_observations, dtype=np.int32)

    rng_np = np.random.default_rng(seed)

    iterator = range(num_observations)
    if verbose:
        print(f"Processing {num_observations} observations...")
        iterator = tqdm(iterator, desc="Generating synthetic data")

    for obs_idx in iterator:
        user_id = int(original_users[obs_idx])
        user_idx = user_id_to_index[user_id]
        original_options_row = original_options[obs_idx]

        if not allow_repeat_items and user_chosen_items[user_id]:
            obs_utilities = user_utilities[user_idx].copy()
            chosen_items_array = np.array(list(user_chosen_items[user_id]), dtype=np.int64)
            obs_utilities[chosen_items_array] = -1e4
            shifted = obs_utilities - obs_utilities.max(axis=0, keepdims=True)
            exp_utils = np.exp(shifted)
            obs_probs = (exp_utils / exp_utils.sum(axis=0, keepdims=True)).mean(axis=-1)
        else:
            obs_utilities = user_utilities[user_idx]
            obs_probs = user_probs[user_idx]

        probs_for_slate = obs_probs.copy()
        if padding_id is not None:
            probs_for_slate[padding_id] = 0.0
        if not allow_repeat_items and user_chosen_items[user_id]:
            probs_for_slate[np.array(list(user_chosen_items[user_id]), dtype=np.int64)] = 0.0
        slate = generate_slate_sample_probability(
            probs=probs_for_slate,
            k=slate_size,
            include_no_choice=include_no_choice,
            no_choice_id=no_choice_id,
            rng=rng_np,
        )

        if len(slate) < len(original_options_row):
            padding_needed = len(original_options_row) - len(slate)
            slate = np.concatenate([slate, np.full(padding_needed, padding_id, dtype=slate.dtype)])
        elif len(slate) > len(original_options_row):
            slate = slate[:len(original_options_row)]
        original_padding_mask = (original_options_row == padding_id)
        slate = np.where(original_padding_mask, padding_id, slate)

        choice_pos, choice = _sample_choice_from_slate_numpy(
            slate, obs_utilities.squeeze(-1) if not is_mxl else obs_utilities, rng_np,
        )

        synthetic_options[obs_idx] = slate
        synthetic_choices_pos[obs_idx] = choice_pos

        if not allow_repeat_items:
            if include_no_choice and choice == no_choice_id:
                pass
            elif padding_id is not None and choice == padding_id:
                raise ValueError("Chosen item is padding, which should not happen.")
            else:
                user_chosen_items[user_id].add(choice)

    return synthetic_options, synthetic_choices_pos


def generate_synthetic_data(
    model_dir: str,
    base_dataset: Literal["ContentWise_LastActionSplit", "Stockholm_LastActionSplit"],
    slate_policy: SlatePolicy = "sample_probability",
    seed: int = 42,
    temperature: float = 1.0,
    allow_repeat_items: bool = True,
    no_choice_utility_boost: float = 0.0,
    intercept_scale: float = 1.0,
    verbose: bool = True,
    sample_limit: int = None,
    batch_size: int = 128,
    num_prior_samples: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    from src.utils.load_existing_model import load_trained_model, load_results_metadata

    original_data = _load_base_dataset(base_dataset, sample_limit, verbose)

    original_users = np.concatenate([
        original_data["train"]["user"],
        original_data["val"]["user"],
        original_data["test"]["user"]
    ])
    original_options = np.concatenate([
        original_data["train"]["options"],
        original_data["val"]["options"],
        original_data["test"]["options"]
    ])

    num_observations = len(original_users)
    num_items = original_data["max_item_id"] + 1
    no_choice_id = original_data["no_choice_id"]
    padding_id = original_data["padding_id"]

    if no_choice_id is not None:
        slate_size = original_options.shape[1] - 1
    else:
        slate_size = original_options.shape[1]

    if verbose:
        print(f"Original dataset ({base_dataset}): {num_observations} observations, {num_items} items")
        print(f"Slate size: {slate_size}" + (" (excluding no-choice)" if no_choice_id is not None else ""))
        print(f"No-choice ID: {no_choice_id}, Padding ID: {padding_id}")

    if verbose:
        print(f"Loading trained model from {model_dir}...")

    metadata = load_results_metadata(model_dir)
    model_name = metadata["model_name"]

    model, _, _ = load_trained_model(
        model_dir,
        no_choice_id=no_choice_id,
        padding_id=padding_id,
        never_chosen_id=original_data["never_chosen_id"],
    )

    device = torch.device(device)
    model.to(device)
    model.eval()

    model.temperature = temperature
    model.no_choice_boost = no_choice_utility_boost
    model.intercept_scale = intercept_scale
    if num_prior_samples is not None and hasattr(model, "num_prior_samples"):
        model.num_prior_samples = num_prior_samples

    user_backbone = getattr(model, "user_backbone", "id")

    if verbose:
        print(f"Model type: {model_name} (user backbone: {user_backbone})")
        print(f"Slate generation policy: {slate_policy}")
        print(f"Allow repeat items: {allow_repeat_items}")
        print(f"Temperature: {temperature}")
        print(f"No-choice utility boost: {no_choice_utility_boost}")
        print(f"Intercept scale: {intercept_scale}")

    torch.manual_seed(seed)

    if user_backbone != "id":
        synthetic_options, synthetic_choices_pos = _rollout_seq(
            model=model,
            original_users=original_users, original_options=original_options,
            num_items=num_items, slate_size=slate_size,
            no_choice_id=no_choice_id, padding_id=padding_id,
            allow_repeat_items=allow_repeat_items,
            seed=seed, batch_size=batch_size, device=device, verbose=verbose,
        )
    else:
        synthetic_options, synthetic_choices_pos = _generate_id_backbone(
            model=model,
            original_users=original_users, original_options=original_options,
            num_items=num_items, slate_size=slate_size,
            no_choice_id=no_choice_id, padding_id=padding_id,
            allow_repeat_items=allow_repeat_items,
            seed=seed, batch_size=batch_size, device=device, verbose=verbose,
        )

    synthetic_choices = synthetic_options[np.arange(len(synthetic_choices_pos)), synthetic_choices_pos]

    if verbose:
        print(f"Generated {len(synthetic_choices)} synthetic observations")
        if no_choice_id is not None:
            print(f"No-choice rate: {(synthetic_choices == no_choice_id).mean():.2%}")

    train_size = len(original_data["train"]["user"])
    val_size = len(original_data["val"]["user"])

    train_end = train_size
    val_end = train_size + val_size

    result = {
        "max_user_id": original_data["max_user_id"],
        "max_item_id": original_data["max_item_id"],
        "no_choice_id": no_choice_id,
        "padding_id": padding_id,
        "never_chosen_id": original_data["never_chosen_id"],
        "train": {
            "user": original_users[:train_end],
            "choices_pos": synthetic_choices_pos[:train_end],
            "options": synthetic_options[:train_end],
        },
        "val": {
            "user": original_users[train_end:val_end],
            "choices_pos": synthetic_choices_pos[train_end:val_end],
            "options": synthetic_options[train_end:val_end],
        },
        "test": {
            "user": original_users[val_end:],
            "choices_pos": synthetic_choices_pos[val_end:],
            "options": synthetic_options[val_end:],
        },
        "metadata": {
            "source_model_dir": model_dir,
            "source_model_name": model_name,
            "source_user_backbone": user_backbone,
            "slate_policy": slate_policy,
            "temperature": temperature,
            "seed": seed,
            "allow_repeat_items": allow_repeat_items,
            "no_choice_utility_boost": no_choice_utility_boost,
            "intercept_scale": intercept_scale,
            "batch_size": batch_size,
            "num_prior_samples": num_prior_samples,
            "sample_limit": sample_limit,
            "generation_type": "synthetic_from_model",
            "base_dataset": base_dataset,
        }
    }

    if base_dataset == "ContentWise_LastActionSplit":
        result["item_features"] = original_data.get("item_features")
    elif base_dataset == "Stockholm_LastActionSplit":
        if "metadata" in original_data:
            result["metadata"]["original_metadata"] = original_data["metadata"]

    result["_model"] = model

    return result


def save_synthetic_dataset(
    data: Dict[str, Any],
    output_path: str,
    verbose: bool = True,
) -> None:
    import shutil
    from src.utils.checkpoints import store_model

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_dir = os.path.dirname(output_path)

    model = data.get("_model")
    if model is not None:
        source_model_dir = os.path.abspath(os.path.join(output_dir, "source_model"))
        os.makedirs(source_model_dir, exist_ok=True)

        store_model(model, dir=source_model_dir, filename_prefix="best_model_state")

        original_source_model_dir = data.get("metadata", {}).get("source_model_dir")
        if original_source_model_dir:
            original_results_json = os.path.join(original_source_model_dir, "results.json")
            if os.path.exists(original_results_json):
                dest_results_json = os.path.join(source_model_dir, "results.json")
                shutil.copy(original_results_json, dest_results_json)
                if verbose:
                    print(f"Copied results.json from {original_results_json}")

                with open(dest_results_json, 'r') as f:
                    results_data = json.load(f)

                if "hyperparams" in results_data:
                    results_data["hyperparams"]["temperature"] = data.get("metadata", {}).get("temperature", 1.0)
                    results_data["hyperparams"]["no_choice_boost"] = data.get("metadata", {}).get("no_choice_utility_boost", 0.0)
                    results_data["hyperparams"]["intercept_scale"] = data.get("metadata", {}).get("intercept_scale", 1.0)

                with open(dest_results_json, 'w') as f:
                    json.dump(results_data, f, indent=2)

                if verbose:
                    print(f"Updated results.json with temperature={results_data['hyperparams']['temperature']}, "
                          f"no_choice_boost={results_data['hyperparams']['no_choice_boost']}, "
                          f"intercept_scale={results_data['hyperparams']['intercept_scale']}")

        if "metadata" in data:
            data["metadata"]["source_model_saved_path"] = source_model_dir
        if verbose:
            print(f"Saved source model to {source_model_dir}")

    all_users = np.concatenate([
        data["train"]["user"],
        data["val"]["user"],
        data["test"]["user"]
    ])
    all_choices_pos = np.concatenate([
        data["train"]["choices_pos"],
        data["val"]["choices_pos"],
        data["test"]["choices_pos"]
    ])
    all_options = np.concatenate([
        data["train"]["options"],
        data["val"]["options"],
        data["test"]["options"]
    ])

    all_choices = all_options[np.arange(len(all_choices_pos)), all_choices_pos]

    train_size = len(data["train"]["user"])
    val_size = len(data["val"]["user"])
    test_size = len(data["test"]["user"])

    np.savez_compressed(
        output_path,
        users=all_users,
        choices=all_choices,
        choices_pos=all_choices_pos,
        options=all_options,
        item_features=data.get("item_features"),
        max_user_id=data["max_user_id"],
        max_item_id=data["max_item_id"],
        no_choice_id=data["no_choice_id"] if data["no_choice_id"] is not None else -999,
        padding_id=data["padding_id"] if data["padding_id"] is not None else -999,
        never_chosen_id=data["never_chosen_id"] if data["never_chosen_id"] is not None else -999,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        metadata_json=json.dumps(data.get("metadata", {})),
    )

    if verbose:
        print(f"Saved synthetic dataset to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic data from a trained model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--slate_policy", type=str, default="sample_probability",
                        choices=["sample_probability"],
                        help="Policy for generating slates")
    parser.add_argument("--base_dataset", type=str, required=True,
                        help="Base name for saving the generated dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Logit scale used during generation")
    parser.add_argument("--allow_repeat_items", type=bool, default=True,
                        help="Whether to allow users to choose the same item multiple times",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--no_choice_utility_boost", type=float, default=0.0,
                        help="Constant added to no-choice utility during sampling (default: 0.0)")
    parser.add_argument("--intercept_scale", type=float, default=1.0,
                        help="Scale factor for item intercepts (default: 1.0)")
    parser.add_argument("--store", type=bool, default=True,
                        help="Whether to store the generated dataset to disk",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Limit the number of observations to load from the base dataset (for testing)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of users per model forward")
    parser.add_argument("--num_prior_samples", type=int, default=None,
                        help="Generation-time override of the MXL Monte-Carlo sample count "
                             "(None keeps the model's stored value)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device for the model forward ('cpu' or 'cuda')")

    args = parser.parse_args()

    synthetic_data = generate_synthetic_data(
        model_dir=args.model_dir,
        base_dataset=args.base_dataset,
        slate_policy=args.slate_policy,
        seed=args.seed,
        temperature=args.temperature,
        allow_repeat_items=args.allow_repeat_items,
        no_choice_utility_boost=args.no_choice_utility_boost,
        intercept_scale=args.intercept_scale,
        verbose=True,
        sample_limit=args.sample_limit,
        batch_size=args.batch_size,
        num_prior_samples=args.num_prior_samples,
        device=args.device,
    )

    model_dir_name = os.path.basename(os.path.normpath(args.model_dir))
    limit_suffix = f"_limit{args.sample_limit}" if args.sample_limit is not None else ""
    output_filepath = ("data/"
        + f"{args.base_dataset}_synthetic_{model_dir_name}_{args.slate_policy}_{args.seed}_{args.temperature}_{args.allow_repeat_items}_ncboost{args.no_choice_utility_boost}_iscale{args.intercept_scale}{limit_suffix}"
        + "/pre-processed.npz"
    )

    if args.store:
        save_synthetic_dataset(synthetic_data, output_filepath)

    print("\nGenerated data structure:")
    print(f"Train: {len(synthetic_data['train']['user'])} observations")
    print(f"Val: {len(synthetic_data['val']['user'])} observations")
    print(f"Test: {len(synthetic_data['test']['user'])} observations")
