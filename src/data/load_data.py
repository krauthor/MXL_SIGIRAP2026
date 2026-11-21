import os
from typing import Dict, Any

import numpy as np

from src.data.dataset import build_user_histories, build_no_choice_rate

def load_data(
        dataset_name='ContentWise_LastActionSplit',
        seed=42,
        build_history=False,
        max_seq_len=50,
        build_rate=False,
):
    np.random.seed(seed)

    if dataset_name == 'ContentWise_LastActionSplit':
        return load_contentwise_last_action_split_data(
            build_history=build_history,
            max_seq_len=max_seq_len,
            build_rate=build_rate,
        )
    elif dataset_name == 'Stockholm_LastActionSplit':
        return load_stockholm_last_action_split_data(
            build_history=build_history,
            max_seq_len=max_seq_len,
        )
    elif '_synthetic_' in dataset_name:
        return load_semisynthetic_dataset(
            directory="data/" + dataset_name,
            build_history=build_history,
            max_seq_len=max_seq_len,
            build_rate=build_rate,
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Supported datasets are: 'ContentWise_LastActionSplit', 'Stockholm_LastActionSplit' and the semi-synthetic datasets ('*_synthetic_*').")


def load_semisynthetic_dataset(
    directory: str,
    verbose: bool = True,
    build_history: bool = False,
    max_seq_len: int = 50,
    build_rate: bool = False,
) -> Dict[str, Any]:
    import json

    file_path = directory + "/pre-processed.npz"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Synthetic dataset file not found at {file_path}")

    npzfile = np.load(file_path, allow_pickle=True)

    users = npzfile["users"]
    choices_pos = npzfile["choices_pos"]
    options = npzfile["options"]

    train_size = int(npzfile["train_size"])
    val_size = int(npzfile["val_size"])

    train_end = train_size
    val_end = train_size + val_size

    no_choice_id = int(npzfile["no_choice_id"])
    no_choice_id = None if no_choice_id == -999 else no_choice_id

    padding_id = int(npzfile["padding_id"])
    padding_id = None if padding_id == -999 else padding_id

    never_chosen_id = int(npzfile["never_chosen_id"])
    never_chosen_id = None if never_chosen_id == -999 else never_chosen_id

    metadata_json = str(npzfile["metadata_json"])
    metadata = json.loads(metadata_json) if metadata_json else {}

    if verbose:
        print(f"Loaded synthetic dataset from {file_path}")
        print(f"Total observations: {len(users)}")
        print(f"Split sizes: train={train_size}, val={val_size}, test={len(users) - val_end}")
        if metadata:
            print(f"Generated from model: {metadata.get('source_model_name', 'unknown')}")
            print(f"Slate policy: {metadata.get('slate_policy', 'unknown')}")

    data = {
        "max_user_id": int(npzfile["max_user_id"]),
        "max_item_id": int(npzfile["max_item_id"]),
        "no_choice_id": no_choice_id,
        "padding_id": padding_id,
        "never_chosen_id": never_chosen_id,
        "train": {
            "user": users[:train_end],
            "choices_pos": choices_pos[:train_end],
            "options": options[:train_end],
        },
        "val": {
            "user": users[train_end:val_end],
            "choices_pos": choices_pos[train_end:val_end],
            "options": options[train_end:val_end],
        },
        "test": {
            "user": users[val_end:],
            "choices_pos": choices_pos[val_end:],
            "options": options[val_end:],
        },
        "metadata": metadata,
    }

    if build_history or build_rate:
        if "choices" in npzfile.files:
            choices = npzfile["choices"]
        else:
            choices = options[np.arange(len(choices_pos)), choices_pos]
        if build_history:
            history = build_user_histories(
                users, choices, max_seq_len,
                exclude_ids=(no_choice_id,), pad_id=padding_id if padding_id is not None else 0,
            )
            data["train"]["history"] = history[:train_end]
            data["val"]["history"] = history[train_end:val_end]
            data["test"]["history"] = history[val_end:]
        if build_rate and no_choice_id is not None:
            rate = build_no_choice_rate(users, choices, no_choice_id)
            data["train"]["no_choice_rate"] = rate[:train_end]
            data["val"]["no_choice_rate"] = rate[train_end:val_end]
            data["test"]["no_choice_rate"] = rate[val_end:]

    return data

def load_contentwise_last_action_split_data(
        sample_limit=None,
        build_history=False,
        max_seq_len=50,
        build_rate=False,
):
    print("Loading ContentWise data (last-action split)...")
    npzfile = np.load("data/ContentWiseImpressions/pre-processed/pre-processed_CW10M.npz")
    sample_limit = npzfile["users"].shape[0] if sample_limit is None else sample_limit

    users = npzfile["users"][:sample_limit]
    choices = npzfile["choices"][:sample_limit]
    options = npzfile["options"][:sample_limit]

    choices_pos = np.argmax(options == np.expand_dims(choices, -1), axis=1)

    print("Compiling train, validation, and test sets...")

    unique_users, inverse_indices, user_counts = np.unique(
        users, return_inverse=True, return_counts=True
    )

    user_boundaries = np.concatenate([[1], np.diff(users) != 0]).astype(np.int32)
    cumcount_from_start = np.arange(len(users)) - np.maximum.accumulate(
        np.arange(len(users)) * user_boundaries
    )

    position_from_end = user_counts[inverse_indices] - 1 - cumcount_from_start

    train_mask = position_from_end >= 2
    val_mask = position_from_end == 1
    test_mask = position_from_end == 0

    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    train = {"user": users[train_mask], "choices_pos": choices_pos[train_mask], "options": options[train_mask]}
    val = {"user": users[val_mask], "choices_pos": choices_pos[val_mask], "options": options[val_mask]}
    test = {"user": users[test_mask], "choices_pos": choices_pos[test_mask], "options": options[test_mask]}

    if build_history:
        print(f"Building user histories (max_seq_len={max_seq_len})...")
        history = build_user_histories(users, choices, max_seq_len, exclude_ids=(1,), pad_id=0)
        train["history"] = history[train_mask]
        val["history"] = history[val_mask]
        test["history"] = history[test_mask]

    if build_rate:
        print("Building per-observation no-choice rate...")
        rate = build_no_choice_rate(users, choices, no_choice_id=1)
        train["no_choice_rate"] = rate[train_mask]
        val["no_choice_rate"] = rate[val_mask]
        test["no_choice_rate"] = rate[test_mask]

    return {
        "max_user_id": int(users.max()),
        "max_item_id": int(options.max()),
        "no_choice_id": 1,
        "padding_id": 0,
        "never_chosen_id": None,
        "train": train,
        "val": val,
        "test": test,
        "item_features": npzfile["item_types"]
    }


def load_stockholm_last_action_split_data(
    sample_limit=None,
    build_history=False,
    max_seq_len=50,
):
    print("Loading Stockholm restaurant data (last-action split)...")

    try:
        npzfile = np.load("data/DHRD/pre-processed/stockholm_restaurants.npz")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Stockholm restaurant data not found. Please run the preprocessing notebook "
            "(15_stockholm_data_preprocessing.ipynb) first to generate the processed data."
        )

    sample_limit = npzfile["users"].shape[0] if sample_limit is None else sample_limit

    users = npzfile["users"][:sample_limit]
    choices = npzfile["choices"][:sample_limit]
    options = npzfile["options"][:sample_limit]

    padding_id = int(npzfile["padding_id"])
    no_choice_id = npzfile["no_choice_id"]
    no_choice_id = None if no_choice_id == -999 else int(no_choice_id)

    max_options = int(npzfile["max_options"])
    num_users = int(npzfile["num_users"])
    num_vendors = int(npzfile["num_vendors"])

    choices_pos = np.argmax(options == np.expand_dims(choices, -1), axis=1)

    print(f"Dataset info:")
    print(f"- Total choice situations: {len(users):,}")
    print(f"- Unique users: {len(np.unique(users)):,}")

    print("Compiling train, validation, and test sets (temporal per-user split)...")

    unique_users, inverse_indices, user_counts = np.unique(
        users, return_inverse=True, return_counts=True
    )

    user_boundaries = np.concatenate([[1], np.diff(users) != 0]).astype(np.int32)
    cumcount_from_start = np.arange(len(users)) - np.maximum.accumulate(
        np.arange(len(users)) * user_boundaries
    )

    position_from_end = user_counts[inverse_indices] - 1 - cumcount_from_start

    train_mask = position_from_end >= 2
    val_mask = position_from_end == 1
    test_mask = position_from_end == 0

    print(f"Split sizes:")
    print(f"- Train: {train_mask.sum():,} ({train_mask.sum()/len(users)*100:.1f}%)")
    print(f"- Validation: {val_mask.sum():,} ({val_mask.sum()/len(users)*100:.1f}%)")
    print(f"- Test: {test_mask.sum():,} ({test_mask.sum()/len(users)*100:.1f}%)")

    train_dict = {
        "user": users[train_mask],
        "choices_pos": choices_pos[train_mask],
        "options": options[train_mask],
    }
    val_dict = {
        "user": users[val_mask],
        "choices_pos": choices_pos[val_mask],
        "options": options[val_mask],
    }
    test_dict = {
        "user": users[test_mask],
        "choices_pos": choices_pos[test_mask],
        "options": options[test_mask],
    }

    if build_history:
        print(f"Building user histories (max_seq_len={max_seq_len})...")
        exclude = (no_choice_id,) if no_choice_id is not None else ()
        history = build_user_histories(users, choices, max_seq_len, exclude_ids=exclude, pad_id=padding_id)
        train_dict["history"] = history[train_mask]
        val_dict["history"] = history[val_mask]
        test_dict["history"] = history[test_mask]

    return {
        "max_user_id": int(users.max()),
        "max_item_id": int(options[options >= 0].max()),
        "no_choice_id": no_choice_id,
        "padding_id": padding_id,
        "never_chosen_id": None,
        "train": train_dict,
        "val": val_dict,
        "test": test_dict,
        "metadata": {
            "total_users": num_users,
            "total_vendors": num_vendors,
            "max_options_per_choice": max_options,
            "dataset_name": "Stockholm Restaurants (LastActionSplit)",
            "description": "Stockholm restaurant choice data with temporal per-user split"
        }
    }
