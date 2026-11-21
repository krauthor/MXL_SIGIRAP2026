import os

import torch


def store_model(model, dir, filename_prefix):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, filename_prefix + ".pt")
    print(f"Storing model state under {path}...")
    torch.save(model.state_dict(), path)


def load_state_dict(dir, filename_prefix, map_location="cpu"):
    path = os.path.join(dir, filename_prefix + ".pt")
    print(f"Retrieving model state from {path}...")
    return torch.load(path, map_location=map_location)
