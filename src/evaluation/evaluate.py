from tqdm.auto import tqdm

import numpy as np

import torch

from src.pipelines.unified_training import (
    move_batch_to_device, MXL_MODELS, MNL_LOSS_FNS, compute_forward_loss,
)


def evaluate(
    model_name,
    model,
    test_data_loader,
    hyperparams,
    device=None,
):
    print("Evaluating...")
    model.eval()
    beta = hyperparams["beta_max"]

    batch_results = []

    is_mxl = model_name in MXL_MODELS

    pbar_batch = tqdm(test_data_loader, total=len(test_data_loader), ascii=True, leave=False)
    for batch in pbar_batch:
        batch = move_batch_to_device(batch, device)

        batch_result = {}
        batch_result["num_obs"] = batch[0]["user_ids"].shape[0]

        choice_pos_onehot = batch[1]["choice_pos_onehot"].detach().cpu().numpy()
        choice_pos = batch[1]["choice_pos"].detach().cpu().numpy()

        if is_mxl:
            with torch.no_grad():
                components, probs = compute_forward_loss(
                    model, model_name, batch[0], batch[1]["choice_pos_onehot"],
                    beta=beta, l2_weight=0, return_component_values=True,
                )
            elbo_val, ll, kl = components[0], components[1], components[2]
            batch_result["elbo"] = float(elbo_val)
            batch_result["nll"] = (-ll).detach().cpu().numpy()
            batch_result["kl"] = float(kl.mean())
            probs = probs.detach().cpu().numpy()
        elif model_name in MNL_LOSS_FNS:
            with torch.no_grad():
                components, probs = compute_forward_loss(
                    model, model_name, batch[0], batch[1]["choice_pos_onehot"],
                    l2_weight=0, return_component_values=True,
                )
            nll = components[0]
            probs = probs.detach().cpu().numpy()
            batch_result["nll"] = float(nll)
        elif model_name in ["MostPop", "Random"]:
            _, probs = model.test_step(batch)
            probs = probs.detach().cpu().numpy()
            batch_result["nll"] = np.nan

        if probs.ndim == 1:
            probs = probs.reshape(1, probs.shape[0])
        ranking = np.argsort(-probs, axis=1)
        ranks = np.zeros_like(ranking)
        ranks[np.arange(probs.shape[0])[:, None], ranking] = 1 + np.arange(ranking.shape[1])[None, :]

        dcg = np.sum(choice_pos_onehot / np.log2(ranks + 1), axis=-1)
        idcg = 1 / np.log2(2)
        batch_result["ndcg"] = dcg / idcg

        if model_name not in ["MostPop", "Random"]:
            choice_label = choice_pos != 0
            choice_prob = probs[:, 1:].sum(axis=1)
            batch_result["slate_nll"] = -(
                choice_label * np.log(choice_prob + 1e-5)
                + (1 - choice_label) * np.log(1 - choice_prob + 1e-5)
            )
        else:
            batch_result["slate_nll"] = np.nan

        batch_results.append(batch_result)

    def isnan_all(v):
        return np.all(np.isnan(v)) if np.ndim(v) > 0 else np.isnan(v)

    results = {}
    for key in batch_results[0].keys():
        if key == "num_obs":
            continue
        all_values = [b[key] for b in batch_results]
        if all(isnan_all(v) for v in all_values):
            continue
        valid = [(v, b["num_obs"]) for v, b in zip(all_values, batch_results) if not isnan_all(v)]
        if not valid:
            continue
        if np.ndim(valid[0][0]) > 0:
            results[key] = float(np.concatenate([np.asarray(v) for v, _ in valid]).mean())
        else:
            total_w = sum(w for _, w in valid)
            results[key] = float(sum(float(v) * w for v, w in valid) / total_w)

    return results
