import numpy as np
import torch.utils.data as data

from src.data.dataset import choice_collate


def create_synthetic_data(
    num_users,
    num_items,
    num_options,
    num_user_types,
    batch_size,
    seed=0,
):
    print("Creating synthetic dataset...")

    rng = np.random.RandomState(seed)

    init_params = {
        "user_mean": np.array([[2, -2], [2, 2]], dtype=np.float64),
        "user_cov": np.array([[[0.15, 0], [0, 0.15]], [[0.15, 0], [0, 0.15]]], dtype=np.float64),
    }

    true_params = {
        "item_embs": rng.multivariate_normal(np.zeros(2), np.eye(2), size=num_items),
        "num_options": num_options,
        "num_users": num_users,
    }

    true_params["user_mean"] = np.stack([
        init_params["user_mean"] @ np.array([
            [np.cos(2 * np.pi * user_id / num_user_types), -np.sin(2 * np.pi * user_id / num_user_types)],
            [np.sin(2 * np.pi * user_id / num_user_types), np.cos(2 * np.pi * user_id / num_user_types)],
        ])
        for user_id in range(num_users)
    ])

    true_params["user_cov"] = np.stack([
        init_params["user_cov"] for _ in range(num_users)
    ])

    # The dataset class is adapted from the UvA Deep Learning tutorial 2, which is released under
    # the MIT License (Copyright (c) 2020 Phillip Lippe):
    # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html
    class ChoiceDataset(data.Dataset):

        def __init__(self, size, seed, true_params, std=0.1):
            super().__init__()
            self.size = size
            self.np_rng = np.random.RandomState(seed=seed)
            self.std = std
            self.true_params = true_params
            self.generate_continuous_choices()

        def generate_continuous_choices(self):
            self.user_ids = self.np_rng.choice(
                a=np.arange(self.true_params["num_users"]),
                size=(self.size,),
                replace=True,
            )

            embs = [
                [
                    self.np_rng.multivariate_normal(user_mean, user_cov, size=1)
                    for user_mean, user_cov in zip(
                        self.true_params["user_mean"][user_id], self.true_params["user_cov"][user_id]
                    )
                ]
                for user_id in self.user_ids
            ]
            self.user_embs = np.asarray(embs)

            user_emb_mix_idxs = self.np_rng.choice(
                a=np.arange(len(self.true_params["user_mean"][0])),
                size=(self.size,),
                replace=True,
            )
            self.user_embs = self.user_embs[np.arange(self.size), user_emb_mix_idxs]

            all_item_embs = np.concatenate(
                [np.zeros((1, self.true_params["item_embs"].shape[1])), self.true_params["item_embs"]], axis=0
            )

            self.option_ids = 1 + self.np_rng.choice(
                len(self.true_params["item_embs"]), size=(self.size, self.true_params["num_options"])
            )
            self.option_ids = np.concatenate(
                [np.zeros((self.option_ids.shape[0], 1)).astype(int), self.option_ids], axis=1
            )
            self.item_embs = all_item_embs[self.option_ids]

            det_utils = np.sum(np.multiply(self.user_embs, self.item_embs), axis=-1)
            rand_utils = det_utils + self.np_rng.gumbel(loc=0, scale=1, size=det_utils.shape)
            self.choice_pos = np.argmax(rand_utils, axis=-1)

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            user_id = self.user_ids[idx]
            option_id = self.option_ids[idx]
            choice_pos = self.choice_pos[idx]
            choice_pos_onehot = np.eye(self.true_params["num_options"] + 1)[choice_pos]
            data_input = {
                "user_ids": user_id,
                "option_id": option_id,
            }
            return (
                data_input,
                {
                    "choice_pos": choice_pos,
                    "choice_pos_onehot": choice_pos_onehot,
                }
            )

    print("Creating training data...")
    train_dataset = ChoiceDataset(size=10000, seed=42, true_params=true_params)
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=choice_collate, drop_last=True)
    print("Creating validation data...")
    val_dataset = ChoiceDataset(size=2500, seed=42, true_params=true_params)
    val_data_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=choice_collate, drop_last=True)
    print("Creating test data...")
    test_dataset = ChoiceDataset(size=2500, seed=42, true_params=true_params)
    test_data_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=choice_collate, drop_last=True)

    no_choice_id = 0
    padding_id = None
    never_chosen_id = None

    print("Data created successfully.")

    return (
        train_data_loader,
        val_data_loader,
        test_data_loader,
        no_choice_id,
        padding_id,
        never_chosen_id,
    )
