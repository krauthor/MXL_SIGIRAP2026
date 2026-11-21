# The Triton kernels in this project were developed with the assistance of
# Claude Opus and Claude Fable (Anthropic).

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.data.create_synthetic_data import create_synthetic_data
from src.models.create_model import create_model
from src.models.elbo import elbo, unbias_loglikelihood

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
NUM_USERS, NUM_ITEMS, NUM_OPTIONS, NUM_TYPES = 5, 20, 10, 5
PADDING_ID = NUM_ITEMS + 1
BATCH = 512
EPOCHS = 80
LR = 0.05

HP = dict(out_dim=2, num_prior_samples=32, cov_type="diag", has_intercept=True, beta_max=0.0,
          l2_weight=0.0, temperature=1.3, no_choice_boost=0.4, optimizer="adam")
BETA = 0.0


def to_dev(batch):
    data_input, labels = batch
    di = {k: (v.to(DEV).long() if k in ("user_ids", "option_id") else v.to(DEV).float())
          for k, v in data_input.items()}
    lb = {k: v.to(DEV).float() for k, v in labels.items()}
    return di, lb


def truncate_slates(di, y):
    opt = di["option_id"]
    O = opt.shape[1]
    tgt = y["choice_pos_onehot"].argmax(dim=-1)
    h = (di["user_ids"].long() * 7919 + opt.long().sum(dim=1) * 104729 + tgt * 31) % (O - 2)
    valid = torch.maximum(tgt + 1, h + 2)
    pos = torch.arange(O, device=opt.device)[None, :]
    di["option_id"] = torch.where(pos < valid[:, None], opt, torch.full_like(opt, PADDING_ID))
    return di, y


def make_model():
    torch.manual_seed(0)
    return create_model("MXL", hyperparams=dict(HP), num_users=NUM_USERS,
                        num_items=NUM_ITEMS + 2, no_choice_id=0,
                        padding_id=PADDING_ID, never_chosen_id=None).to(DEV)


def ref_loss(model, di, y):
    logits, probs = model(di)
    return elbo(logits, probs, None, model, y["choice_pos_onehot"], beta=BETA, l2_weight=0.0)


def triton_loss(model, di, y, chunk_samples=None):
    target_idx = y["choice_pos_onehot"].argmax(dim=-1)
    log_g = model.triton_logprobs(di, target_idx, chunk_samples=chunk_samples)
    g = torch.exp(log_g)
    ll = unbias_loglikelihood(torch.log(g.mean(dim=1) + 1e-10), g, g.shape[1])
    kl = model.kl_divergence()
    return -(ll.mean() - BETA * kl.mean())


def triton_chunked_loss(model, di, y):
    return triton_loss(model, di, y, chunk_samples=16)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    nll, correct, n = 0.0, 0, 0
    for batch in loader:
        di, y = to_dev(batch)
        di, y = truncate_slates(di, y)
        _, probs = model(di)
        tgt = y["choice_pos_onehot"].argmax(dim=-1)
        p_tgt = probs[torch.arange(probs.shape[0], device=DEV), tgt]
        nll += -torch.log(p_tgt + 1e-10).sum().item()
        correct += (probs.argmax(dim=-1) == tgt).sum().item()
        n += probs.shape[0]
    return nll / n, correct / n


def procrustes_corr(A, B):
    A0 = A - A.mean(0); B0 = B - B.mean(0)
    A0 = A0 / (np.linalg.norm(A0) + 1e-12); B0 = B0 / (np.linalg.norm(B0) + 1e-12)
    U, _, Vt = np.linalg.svd(A0.T @ B0)
    A_aligned = A0 @ (U @ Vt)
    return float(np.corrcoef(A_aligned.ravel(), B0.ravel())[0, 1])


def train(name, loss_fn, train_loader):
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    curve = []
    for ep in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            di, y = to_dev(batch)
            di, y = truncate_slates(di, y)
            opt.zero_grad()
            loss = loss_fn(model, di, y)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        curve.append(tot / nb)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"  [{name}] epoch {ep:3d}  train_loss={curve[-1]:.4f}")
    return model, curve


def main():
    print(f"Device: {DEV}  ({torch.cuda.get_device_name(0) if DEV=='cuda' else 'cpu'})")
    train_loader, _, test_loader, *_ = create_synthetic_data(
        NUM_USERS, NUM_ITEMS, NUM_OPTIONS, NUM_TYPES, BATCH, seed=SEED)

    true_item_embs = np.random.RandomState(SEED).multivariate_normal(
        np.zeros(2), np.eye(2), size=NUM_ITEMS)

    rand_nll = np.log(NUM_OPTIONS + 1)
    print(f"\nRandom-baseline NLL = {rand_nll:.4f}, random acc = {1/(NUM_OPTIONS+1):.3f}\n")

    results = {}
    variants = [("ref", ref_loss), ("triton", triton_loss), ("triton-chunk16", triton_chunked_loss)]
    for name, loss_fn in variants:
        print(f"=== training: {name} ===")
        model, curve = train(name, loss_fn, train_loader)
        nll, acc = evaluate(model, test_loader)
        learned = model.item_embs.detach().cpu().numpy()[1:NUM_ITEMS + 1]
        rec = procrustes_corr(learned, true_item_embs)
        results[name] = dict(curve=curve, nll=nll, acc=acc, rec=rec)
        print(f"  -> test NLL={nll:.4f}  test acc={acc:.3f}  item-emb recovery r={rec:.3f}\n")

    names = [n for n, _ in variants]
    print("=" * (22 + 15 * len(names)))
    print(f"{'metric':<22}" + "".join(f"{n:>15}" for n in names))
    print("-" * (22 + 15 * len(names)))
    for k, lbl in [("curve", "loss start->end"), ("nll", "test NLL"),
                   ("acc", "test top-1 acc"), ("rec", "item recovery r")]:
        if k == "curve":
            print(f"{lbl:<22}" + "".join(
                f"{results[n][k][0]:>8.2f}->{results[n][k][-1]:<5.2f}" for n in names))
        else:
            print(f"{lbl:<22}" + "".join(f"{results[n][k]:>15.4f}" for n in names))
    print("=" * (22 + 15 * len(names)))

    r = results["ref"]
    rand_acc = 1 / (NUM_OPTIONS + 1)
    for name in names[1:]:
        t = results[name]
        assert t["curve"][-1] < t["curve"][0] - 0.1, f"{name} training loss did not decrease"
        assert t["acc"] > 3 * rand_acc, f"{name} test accuracy not above random ({t['acc']:.3f})"
        assert t["rec"] > 0.8, f"{name} item-emb recovery too low ({t['rec']:.3f})"
        assert t["nll"] < rand_nll - 0.3, f"{name} test NLL not better than random baseline"
        assert abs(t["acc"] - r["acc"]) < 0.05, f"{name} acc far from reference"
        assert abs(t["nll"] - r["nll"]) < 0.1, f"{name} NLL far from reference"
    print("\n✓ Triton-fused MXL trains correctly (loss down, choices learned, embeddings "
          "recovered) — persist AND chunked noise modes.")


if __name__ == "__main__":
    main()
