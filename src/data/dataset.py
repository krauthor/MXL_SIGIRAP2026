import numpy as np
import torch.utils.data as data
from torch.utils.data import default_collate
from collections import defaultdict


def build_user_histories(users, choices, max_seq_len, exclude_ids=(), pad_id=0):
    n = len(users)
    history = np.full((n, max_seq_len), pad_id, dtype=np.int64)
    exclude = set(int(x) for x in exclude_ids if x is not None)
    exclude.add(int(pad_id))
    seqs = defaultdict(list)
    for j in range(n):
        u = int(users[j])
        s = seqs[u]
        if s:
            h = s[-max_seq_len:]
            history[j, :len(h)] = h
        c = int(choices[j])
        if c not in exclude:
            s.append(c)
    return history


def build_no_choice_rate(users, choices, no_choice_id):
    n = len(users)
    rate = np.zeros(n, dtype=np.float32)
    counts = defaultdict(lambda: [0, 0])
    for j in range(n):
        u = int(users[j])
        total, nc = counts[u]
        if total > 0:
            rate[j] = nc / total
        counts[u] = [total + 1, nc + (1 if int(choices[j]) == no_choice_id else 0)]
    return rate


def choice_collate(batch):
    data_input, labels = default_collate(batch)

    if "user_ids" in data_input:
        data_input["user_ids"] = data_input["user_ids"].long()
    if "option_id" in data_input:
        data_input["option_id"] = data_input["option_id"].long()
    if "history" in data_input:
        data_input["history"] = data_input["history"].long()
    if "no_choice_rate" in data_input:
        data_input["no_choice_rate"] = data_input["no_choice_rate"].float()
    if "item_embs" in data_input:
        data_input["item_embs"] = data_input["item_embs"].float()

    if "choice_pos" in labels:
        labels["choice_pos"] = labels["choice_pos"].long()
    if "choice_pos_onehot" in labels:
        labels["choice_pos_onehot"] = labels["choice_pos_onehot"].float()

    return data_input, labels


class ChoiceDataset(data.Dataset):

    def __init__(self, data):
        super().__init__()
        self.size = data["user"].shape[0]

        self.user_ids = data["user"].ravel() if data["user"].ndim > 1 else data["user"]
        self.choice_pos = data["choices_pos"].ravel() if data["choices_pos"].ndim > 1 else data["choices_pos"]
        self.option_ids = data["options"]
        self.num_options = self.option_ids.shape[1]

        self.history = data.get("history", None)
        self.no_choice_rate = data.get("no_choice_rate", None)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        option_id = self.option_ids[idx]
        choice_pos = self.choice_pos[idx]
        choice_pos_onehot = np.zeros(self.num_options, dtype=np.float32)
        choice_pos_onehot[choice_pos] = 1.0

        data_input = {
            "user_ids": user_id,
            "option_id": option_id,
        }
        if self.history is not None:
            data_input["history"] = self.history[idx]
        if self.no_choice_rate is not None:
            data_input["no_choice_rate"] = self.no_choice_rate[idx]

        return (
            data_input,
            {
                "choice_pos": choice_pos,
                "choice_pos_onehot": choice_pos_onehot,
            }
        )
