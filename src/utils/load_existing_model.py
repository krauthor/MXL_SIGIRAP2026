import os
import json

from src.models.create_model import create_model
from src.utils.checkpoints import load_state_dict


def load_results_metadata(model_dir):
    results_path = os.path.join(model_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"results.json not found in {model_dir}")

    with open(results_path, 'r') as f:
        results = json.load(f)

    for field in ["model_name", "seed", "hyperparams"]:
        if field not in results:
            raise ValueError(f"Required field '{field}' not found in results.json")

    checkpoint_path = os.path.join(model_dir, "best_model_state.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"best_model_state.pt not found in {model_dir}")

    return results


def load_trained_model(model_dir, no_choice_id=None, padding_id=None, never_chosen_id=None):
    metadata = load_results_metadata(model_dir)
    model_name = metadata["model_name"]
    hyperparams = metadata["hyperparams"]

    state_dict = load_state_dict(model_dir, "best_model_state", map_location="cpu")

    if "user_mean" in state_dict:
        num_users = state_dict["user_mean"].shape[0]
    else:
        num_users = int(hyperparams.get("num_users", 1))
    if "item_intercepts" in state_dict:
        num_items = state_dict["item_intercepts"].shape[0]
    elif "item_embs" in state_dict:
        num_items = state_dict["item_embs"].shape[0]
    elif "popularity_scores" in state_dict:
        num_items = state_dict["popularity_scores"].shape[0]
    else:
        raise ValueError("Could not infer num_items from checkpoint.")

    if padding_id is None:
        print("WARNING: padding_id not provided, using computed default (may be incorrect!)")
        padding_id = num_items + 1

    print(f"Reconstructing model: {model_name} with {num_users} users and {num_items} items")

    model = create_model(
        model_name=model_name,
        num_users=num_users,
        num_items=num_items,
        hyperparams=hyperparams,
        no_choice_id=no_choice_id,
        padding_id=padding_id,
        never_chosen_id=never_chosen_id,
    )

    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")

    return model, model, metadata
