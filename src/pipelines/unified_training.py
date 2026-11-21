import os
import copy
import time

import torch
import numpy as np
from tqdm.auto import tqdm

from src.models.elbo import elbo
from src.models.mnl_model import softmax_loss as mnl_softmax_loss
from src.models.mnl_multi_emb_model import mnl_multi_loss
from src.models.mnl_max_emb_model import mnl_max_loss
from src.models.intercept_model import softmax_loss as intercept_softmax_loss
from src.utils.server_utils import running_on_server


MXL_MODELS = (
    "MXL", "MXL_Corr", "MXL_Corr_Shared", "MXL_SASRec",
    "MXL_Corr_SASRec", "MXL_Corr_Shared_SASRec",
)

MNL_LOSS_FNS = {
    "MNL": mnl_softmax_loss,
    "MNL_Multi": mnl_multi_loss,
    "MNL_Max": mnl_max_loss,
    "Intercept": intercept_softmax_loss,
    "MNL_SASRec": mnl_softmax_loss,
    "MNL_Multi_SASRec": mnl_multi_loss,
    "MNL_Max_SASRec": mnl_max_loss,
}


def compute_forward_loss(
    model, model_name, data_input, labels_onehot,
    beta=0.0, l2_weight=0, return_component_values=False, kl_over_batch=False,
):
    if model_name in MXL_MODELS:
        if getattr(model, "user_backbone", "id") != "id":
            logits, probs, kl = model(data_input)
            out = elbo(
                logits, probs, None, model, labels_onehot,
                beta=beta, l2_weight=l2_weight,
                precomputed_kl=kl, return_component_values=return_component_values,
            )
        else:
            if l2_weight > 0:
                logits, probs, user_embs = model(data_input, return_user_embs_no_id=True)
            else:
                logits, probs = model(data_input)
                user_embs = None
            kl = model.kl_divergence_for_users(data_input["user_ids"]) if kl_over_batch else None
            out = elbo(
                logits, probs, user_embs, model, labels_onehot,
                beta=beta, l2_weight=l2_weight,
                precomputed_kl=kl, return_component_values=return_component_values,
            )
    else:
        logits, probs = model(data_input)
        out = MNL_LOSS_FNS[model_name](
            logits, probs, model, labels_onehot,
            l2_weight=l2_weight, return_component_values=return_component_values,
        )
    return out, probs


def move_batch_to_device(batch, device):
    if device is None:
        return batch
    data_input, labels = batch
    data_input = {k: v.to(device) for k, v in data_input.items()}
    labels = {k: v.to(device) for k, v in labels.items()}
    return data_input, labels


def unified_train_model(
    model,
    model_name,
    optimizer,
    train_data_loader,
    val_data_loader,
    num_epochs=100,
    num_early_stopping=10,
    results_dir="data/results/ContentWise",
    device=None,
):
    if model_name in ["MostPop", "Random"]:
        start_time = time.time()
        best_model, loss = model.train_model(
            train_data_loader, val_data_loader, num_epochs, num_early_stopping
        )
        training_duration_sec = time.time() - start_time
        print(f"Training duration: {training_duration_sec:.2f} seconds ({training_duration_sec/60:.2f} minutes)")
        training_stats = {"train_losses": [loss], "val_losses": [], "val_metrics": []}
        return best_model, training_stats

    is_mxl = model_name in MXL_MODELS

    pbar = tqdm(range(num_epochs), total=num_epochs, ascii=True)
    training_stats = {"train_losses": [], "val_losses": [], "val_metrics": []}

    num_epochs_since_last_improvement = 0
    best_state_dict = None
    best_epoch = None
    training_cancelled_due_to_time = False
    l2_weight = getattr(model, 'l2_weight', 0)

    start_time = time.time()
    for epoch in pbar:
        max_training_time_sec = getattr(model, 'max_training_time_sec', None)
        if max_training_time_sec is not None and (time.time() - start_time) > max_training_time_sec:
            print(f"Max training time of {max_training_time_sec} seconds exceeded. Stopping at epoch {epoch}.")
            training_cancelled_due_to_time = True
            break

        beta = getattr(model, 'beta_max', 0.1) if is_mxl else None

        model.train()
        num_batches = len(train_data_loader)
        pbar_batch = tqdm(train_data_loader, total=num_batches, ascii=True, leave=False, disable=running_on_server())
        train_loss_this_epoch = 0
        moving_average_weight = np.max([1 / num_batches, 0.05])
        for batch_idx, batch in enumerate(pbar_batch):
            data_input, labels = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss, _ = compute_forward_loss(
                model, model_name, data_input, labels["choice_pos_onehot"],
                beta=beta, l2_weight=l2_weight,
            )
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            if train_loss_this_epoch == 0:
                train_loss_this_epoch = loss_val
            else:
                train_loss_this_epoch = (
                    (1 - moving_average_weight) * train_loss_this_epoch + moving_average_weight * loss_val
                )

        training_stats["train_losses"].append(train_loss_this_epoch)

        if val_data_loader is not None:
            model.eval()
            val_losses = []
            val_metrics = []
            pbar_batch = tqdm(val_data_loader, total=len(val_data_loader), ascii=True, leave=False, disable=running_on_server())
            with torch.no_grad():
                for batch in pbar_batch:
                    data_input, labels = move_batch_to_device(batch, device)
                    if is_mxl:
                        components, _ = compute_forward_loss(
                            model, model_name, data_input, labels["choice_pos_onehot"],
                            beta=beta, l2_weight=0, return_component_values=True,
                        )
                        elbo_val, ll, kl = components[0], components[1], components[2]
                        val_losses.append(float((-ll.mean())))
                        val_metrics.append(float(elbo_val))
                    else:
                        loss, _ = compute_forward_loss(
                            model, model_name, data_input, labels["choice_pos_onehot"], l2_weight=0,
                        )
                        val_losses.append(float(loss))

            training_stats["val_losses"].append(np.mean(val_losses))
            if is_mxl:
                training_stats["val_metrics"].append(np.mean(val_metrics))

        if epoch == 0 or val_data_loader is None:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if val_data_loader is not None and len(training_stats["val_losses"]) > 1:
            if training_stats["val_losses"][-1] < np.min(training_stats["val_losses"][:-1]):
                num_epochs_since_last_improvement = 0
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            else:
                num_epochs_since_last_improvement += 1
            if num_epochs_since_last_improvement >= num_early_stopping:
                print(f" Early stopping at epoch {epoch}. Taking best model from epoch {best_epoch}.")
                break

        if val_data_loader is not None:
            pbar.set_description(
                f"Train: {training_stats['train_losses'][-1]:.4f}, "
                f"Val: {training_stats['val_losses'][-1]:.4f}, "
                f"Non-improving: {num_epochs_since_last_improvement}/{num_early_stopping}"
            )
        else:
            pbar.set_description(f"Train: {training_stats['train_losses'][-1]:.4f}")

    training_duration_sec = time.time() - start_time
    print(f"Training duration: {training_duration_sec:.2f} seconds ({training_duration_sec/60:.2f} minutes)")
    print(f"Training cancelled due to time limit: {training_cancelled_due_to_time}")

    os.makedirs(results_dir, exist_ok=True)

    if best_epoch is not None:
        print(f"Best Epoch {best_epoch}: Train: {training_stats['train_losses'][best_epoch]:.4f}")
        if val_data_loader is not None:
            print(f"Best Epoch {best_epoch}: Validation: {training_stats['val_losses'][best_epoch]:.4f}")
    else:
        print(f"Final Epoch: Train: {training_stats['train_losses'][-1]:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, training_stats
