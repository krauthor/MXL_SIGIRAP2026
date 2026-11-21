import sys
import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.utils.data as torch_data

sys.path.append(os.path.dirname(os.path.abspath(__file__)).split("src")[0])

from src.data.load_data import load_data
from src.data.dataset import ChoiceDataset, choice_collate
from src.models.create_model import create_model, uses_sequence_backbone
from src.evaluation.evaluate import evaluate
from src.utils.store_results import store_results
from src.utils.checkpoints import store_model
from src.utils.load_existing_model import load_trained_model, load_results_metadata
from src.pipelines.unified_training import (
    unified_train_model, move_batch_to_device, MXL_MODELS, compute_forward_loss,
)
from src.utils.git_utils import get_git_commit_info, print_git_info


def run_experiment(
    model_name,
    hyperparams,
    seed=0,
    optimize_hyperparams=False,
    store_trained_model=False,
    load_model_dir=None,
    results_dir=None,
    compile_model=True,
    compile_backend="inductor",
    disable_cpu_fallback=True,
):
    print("------------------ Setup ------------------")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif disable_cpu_fallback:
        raise RuntimeError(
            "CUDA is not available and CPU fallback is disabled. This usually means a broken "
            "cluster node (CUDA init failure) — aborting instead of silently training on CPU for "
            "hours. Pass --no-disable_cpu_fallback to allow CPU training (e.g. local runs)."
        )
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if load_model_dir:
        print(f"Loading existing model from: {load_model_dir}")
        loaded_metadata = load_results_metadata(load_model_dir)

        cli_eval_params = {
            "dataset_name": hyperparams.get("dataset_name"),
            "semisynth_exclude_seen": hyperparams.get("semisynth_exclude_seen"),
        }
        model_name = loaded_metadata["model_name"]
        hyperparams = loaded_metadata["hyperparams"]
        for key, value in cli_eval_params.items():
            if value is not None:
                hyperparams[key] = value
                print(f"Using CLI value for evaluation parameter '{key}': {value}")

        optimize_hyperparams = hyperparams.get("optimize_hyperparams", False)
        results_dir = load_model_dir if results_dir is None else results_dir
        start_timestamp = loaded_metadata["start_timestamp"]
        git_info = loaded_metadata.get("git_info", {})
        print(f"Loaded model metadata: {model_name}, seed: {seed}")
        hyperparams["results_dir"] = results_dir
        is_loaded_model = True
    else:
        is_loaded_model = False

    if not is_loaded_model:
        git_info = get_git_commit_info()
        print_git_info(git_info)

        start_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        run_id = f"{model_name}_{start_timestamp}_seed{seed}"
        if results_dir is None:
            results_dir = (
                f"data/results/{hyperparams['dataset_name']}/{run_id}"
                if not optimize_hyperparams
                else f"data/hyperparameter_optimization_results/{hyperparams['dataset_name']}/{run_id}"
            )
        os.makedirs(results_dir, exist_ok=True)
        hyperparams["run_id"] = run_id
        hyperparams["results_dir"] = results_dir

    torch.manual_seed(seed)
    np.random.seed(seed)

    use_seq = uses_sequence_backbone(model_name)
    max_seq_len = hyperparams.get("max_seq_len", 50)
    build_rate = use_seq and hyperparams.get("use_no_choice_rate", False)

    data = load_data(
        dataset_name=hyperparams["dataset_name"],
        seed=42,
        build_history=use_seq,
        max_seq_len=max_seq_len,
        build_rate=build_rate,
    )

    print("Creating PyTorch data loaders...")
    train_data_loader = torch_data.DataLoader(
        dataset=ChoiceDataset(data["train"]),
        batch_size=hyperparams["batch_size"], shuffle=True, collate_fn=choice_collate,
        drop_last=True,
    )
    val_data_loader = torch_data.DataLoader(
        dataset=ChoiceDataset(data["val"]),
        batch_size=hyperparams["batch_size"], collate_fn=choice_collate, drop_last=True,
    )
    test_data_loader = torch_data.DataLoader(
        dataset=ChoiceDataset(data["test"]) if not optimize_hyperparams else ChoiceDataset(data["val"]),
        batch_size=hyperparams["batch_size"], collate_fn=choice_collate,
        drop_last=compile_model,
    )

    num_users = int(data["max_user_id"]) + 1
    num_items = int(data["max_item_id"]) + 1
    no_choice_id = data["no_choice_id"]
    padding_id = data["padding_id"]
    never_chosen_id = data["never_chosen_id"]

    if is_loaded_model:
        print("Loading model with dataset-specific special IDs...")
        model, _, _ = load_trained_model(
            load_model_dir, no_choice_id=no_choice_id, padding_id=padding_id,
            never_chosen_id=never_chosen_id,
        )
        model.to(device)

        synth_meta = data.get("metadata") or {}
        source_dirs = {
            os.path.normpath(p) for p in [
                synth_meta.get("source_model_dir"),
                synth_meta.get("source_model_saved_path"),
                os.path.join("data", hyperparams.get("dataset_name", ""), "source_model"),
            ] if p
        }
        if os.path.normpath(load_model_dir) in source_dirs:
            for attr, key in [("temperature", "temperature"),
                              ("no_choice_boost", "no_choice_utility_boost"),
                              ("intercept_scale", "intercept_scale")]:
                val = synth_meta.get(key)
                if val is not None:
                    setattr(model, attr, float(val))
            print(f"Loaded model is the synthetic dataset's SOURCE model -> applied "
                  f"generation params: temperature={getattr(model, 'temperature', None)}, "
                  f"no_choice_boost={getattr(model, 'no_choice_boost', None)}, "
                  f"intercept_scale={getattr(model, 'intercept_scale', None)}")
    else:
        print(f"Creating model object for {model_name}...")
        model = create_model(
            model_name=model_name,
            num_users=num_users,
            num_items=num_items,
            hyperparams=hyperparams,
            no_choice_id=no_choice_id, padding_id=padding_id, never_chosen_id=never_chosen_id,
        )
        model.to(device)

    if compile_model and model_name not in ["MostPop", "Random"]:
        print(f"Compiling model (backend={compile_backend}, dynamic=False)...")
        model.compile(backend=compile_backend, dynamic=False)

    if not is_loaded_model:
        optimizer = None
        if model_name not in ["MostPop", "Random"]:
            print(f"Creating optimizer ({hyperparams['optimizer']})...")
            if hyperparams["optimizer"] == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams["learning_rate"])
            elif hyperparams["optimizer"] == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
            else:
                raise ValueError(f"Unsupported optimizer: {hyperparams['optimizer']}")

        print("------------------ Pre-training performance ------------------")
        if model_name not in ["MostPop", "Random"]:
            _print_train_performance(model, model_name, train_data_loader, device)

        pre_train_results = evaluate(
            model_name=model_name, model=model, test_data_loader=test_data_loader,
            hyperparams=hyperparams, device=device,
        )
        print(f"Pre-train test performance for model {model_name}:")
        for key, val in pre_train_results.items():
            print(f'{key}: {val:.4f}')

        print("------------------ Training ------------------")
        best_trained_model_state, _ = unified_train_model(
            model=model, model_name=model_name, optimizer=optimizer,
            train_data_loader=train_data_loader, val_data_loader=val_data_loader,
            num_epochs=hyperparams["num_epochs"], num_early_stopping=hyperparams["num_early_stopping"],
            results_dir=results_dir,
            device=device,
        )

        print("------------------ Post-training performance ------------------")
        if model_name not in ["MostPop", "Random"]:
            _print_train_performance(best_trained_model_state, model_name, train_data_loader, device)
    else:
        print("Skipping training - using loaded model for evaluation")
        best_trained_model_state = model

    results = evaluate(
        model_name=model_name, model=best_trained_model_state, test_data_loader=test_data_loader,
        hyperparams=hyperparams, device=device,
    )

    if "_synthetic_" in hyperparams.get("dataset_name", "") and not optimize_hyperparams:
        from src.evaluation.semisynthetic_eval import evaluate_semisynthetic_metrics
        print("\n------------------ Semi-synthetic slate evaluation ------------------")
        try:
            semi_results = evaluate_semisynthetic_metrics(
                model=best_trained_model_state,
                model_name=model_name,
                data=data,
                synthetic_metadata=data.get("metadata", {}),
                device=device,
                no_choice_id=no_choice_id,
                padding_id=padding_id,
                never_chosen_id=never_chosen_id,
                seed=seed,
                exclude_seen=hyperparams.get("semisynth_exclude_seen", False),
                fallback_source_dir=os.path.join("data", hyperparams["dataset_name"], "source_model"),
            )
            for key, val in semi_results.items():
                results[f"semisynthetic_{key}"] = val
        except Exception as e:
            print(f"Semi-synthetic slate evaluation failed: {e}")

    print(f"Test performance for model {model_name}:")
    for key, val in results.items():
        if isinstance(val, dict):
            print(f'{key}: [grouped statistics computed]')
        else:
            print(f'{key}: {val:.4f}')

    if not is_loaded_model and store_trained_model and model_name in [
        "MNL", "MXL", "MNL_SASRec", "MXL_SASRec",
    ]:
        print("Storing model parameters...")
        store_dir = os.path.join(os.getcwd(), results_dir)
        store_model(model, dir=store_dir, filename_prefix="best_model_state")

    results_filename = "updated_results.json" if is_loaded_model else "results.json"
    store_results(
        results={
            "model_name": model_name, "seed": seed,
            "start_timestamp": start_timestamp,
            "end_timestamp": datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f'),
            "device": device.type, "trained_on_gpu": device.type == "cuda",
            "git_info": git_info, "hyperparams": hyperparams, "results": results,
        },
        filepath=f"{results_dir}/{results_filename}",
    )
    if is_loaded_model:
        print(f"\nResults saved to {results_dir}/{results_filename}")

    print("--------------------------------")
    return


def _print_train_performance(model, model_name, train_data_loader, device):
    model.train()
    batch = move_batch_to_device(next(iter(train_data_loader)), device)
    components, _ = compute_forward_loss(
        model, model_name, batch[0], batch[1]["choice_pos_onehot"],
        beta=getattr(model, "beta_max", 0.0), l2_weight=getattr(model, "l2_weight", 0),
        return_component_values=True,
    )
    if model_name in MXL_MODELS:
        elbo_val, ll, kl = components[0], components[1], components[2]
        print("ELBO:", float(elbo_val))
        print("Log-likelihood:", float(ll.mean()))
        print("KL divergence:", float(kl.mean()))
    else:
        print("NLL value:", float(components[0]))


def float_to_int(value):
    try:
        return int(float(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid float or int")


def build_parser():
    parser = argparse.ArgumentParser(description="Run an experiment with a given model and hyperparameters.")
    parser.add_argument("--model_name", type=str, default="MNL_Max", help="Model type to use.")
    parser.add_argument("--dataset_name", type=str, default="ContentWise_LastActionSplit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store_trained_model", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--optimize_hyperparams", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--load_model_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--compile", dest="compile_model", type=bool, default=True,
                        action=argparse.BooleanOptionalAction, help="Compile the model with torch.compile.")
    parser.add_argument("--compile_backend", type=str, default="inductor",
                        help="torch.compile backend (e.g. 'inductor', 'aot_eager').")
    parser.add_argument("--disable_cpu_fallback", type=bool, default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Abort instead of silently training on CPU when CUDA is unavailable "
                             "(e.g. broken cluster node). Default True (no fallback); pass "
                             "--no-disable_cpu_fallback to allow CPU training on machines without a GPU.")

    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--num_early_stopping", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--max_training_time_sec", type=float, default=None)
    parser.add_argument("--semisynth_exclude_seen", default=False,
                        action=argparse.BooleanOptionalAction,
                        help="In the semi-synthetic slate eval, drop items the user already "
                             "consumed from the slate (no-repeat, matching the "
                             "--no-allow_repeat_items data generation).")

    parser.add_argument("--out_dim", type=int, default=4)
    parser.add_argument("--has_intercept", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--l2_weight", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Logit temperature (scale). Useful with the sequence backbones, where "
                             "unit-normalized embeddings make the dot-product a cosine in [-1, 1].")

    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--seq_num_heads", type=int, default=1)
    parser.add_argument("--seq_num_blocks", type=int, default=2)
    parser.add_argument("--seq_dropout", type=float, default=0.0)
    parser.add_argument("--use_no_choice_rate", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Append the per-observation no-choice rate to the encoded user features "
                             "(sequence-backbone models only).")
    parser.add_argument("--use_sequence_length", type=bool, default=False, action=argparse.BooleanOptionalAction,
                        help="Append log1p(history length) to the encoded user features "
                             "(sequence-backbone models only).")

    parser.add_argument("--num_embs", type=float_to_int, default=3)

    parser.add_argument("--num_prior_samples", type=int, default=32)
    parser.add_argument("--beta_max", type=float, default=0.1)
    parser.add_argument("--l2_num_prior_samples", type=int, default=128)
    parser.add_argument("--use_output_layer_norm", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cov_type", type=str, default="diag", choices=["diag", "aat", "aat_shared"])
    parser.add_argument("--cov_rank", type=lambda x: int(float(x)), default=0)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    print("------------------ Parameters ------------------")
    print("\n".join([f"{k}: {v}" for k, v in args._get_kwargs()]))

    run_experiment(
        model_name=args.model_name,
        hyperparams={k: v for k, v in args._get_kwargs()},
        seed=args.seed, store_trained_model=args.store_trained_model,
        optimize_hyperparams=args.optimize_hyperparams, load_model_dir=args.load_model_dir,
        results_dir=args.results_dir,
        compile_model=args.compile_model, compile_backend=args.compile_backend,
        disable_cpu_fallback=args.disable_cpu_fallback,
    )
