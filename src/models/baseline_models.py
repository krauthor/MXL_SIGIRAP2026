import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


class MostPop_Model(nn.Module):

    def __init__(self, num_users=10, num_items=10, no_choice_id=None, padding_id=None,
                 never_chosen_id=None, l2_weight=0, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.no_choice_id = no_choice_id
        self.padding_id = padding_id
        self.never_chosen_id = never_chosen_id
        self.has_intercept = False
        self.l2_weight = l2_weight
        self.max_training_time_sec = None
        self.popularity_scores = nn.Parameter(torch.zeros(num_items), requires_grad=False)

    def forward(self, input):
        option_id = input["option_id"]
        option_ids_flat = option_id.reshape(-1)
        option_popularities = self.popularity_scores[option_ids_flat].reshape(option_id.shape[0], -1)
        utils = option_popularities.unsqueeze(-1)

        if self.padding_id is not None:
            padding_mask = (option_id == self.padding_id).unsqueeze(-1)
            utils = utils.masked_fill(padding_mask, -1e6)
        if self.never_chosen_id is not None:
            never_mask = (option_id == self.never_chosen_id).unsqueeze(-1)
            utils = utils.masked_fill(never_mask, -1e6)

        probs = F.softmax(utils.squeeze(-1), dim=-1)
        return utils, probs

    def train_model(self, train_data_loader, val_data_loader=None, num_epochs=1, num_early_stopping=10):
        print("Training MostPop model by counting item popularity...")
        item_counts = torch.zeros(self.num_items)
        total_choices = 0

        for batch in tqdm(train_data_loader, desc="Counting item popularity"):
            data_input, labels = batch
            chosen_items = torch.argmax(labels["choice_pos_onehot"], dim=1)
            option_ids = data_input["option_id"]
            batch_indices = torch.arange(option_ids.shape[0])
            chosen_item_ids = option_ids[batch_indices, chosen_items]

            for item_id in chosen_item_ids.tolist():
                if item_id not in (self.padding_id, self.never_chosen_id):
                    item_counts[item_id] += 1
                    total_choices += 1

        popularity_scores = torch.log(item_counts + 1e-10)
        self.popularity_scores.data = popularity_scores.to(self.popularity_scores.device)

        print(f"MostPop training completed. Total choices counted: {total_choices}")
        return self, 0.0

    def test_step(self, batch):
        self.eval()
        with torch.no_grad():
            data_input, labels = batch
            logits, probs = self(data_input)
            nll = -(labels["choice_pos_onehot"] * F.log_softmax(logits.squeeze(-1), dim=-1)).sum(dim=-1)
        return nll, probs


class Random_Model(nn.Module):

    def __init__(self, num_users=10, num_items=10, no_choice_id=None, padding_id=None,
                 never_chosen_id=None, l2_weight=0, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.no_choice_id = no_choice_id
        self.padding_id = padding_id
        self.never_chosen_id = never_chosen_id
        self.has_intercept = False
        self.l2_weight = l2_weight
        self.max_training_time_sec = None
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, input):
        option_id = input["option_id"]
        batch_size, num_options = option_id.shape
        utils = torch.randn(batch_size, num_options, 1, device=option_id.device)

        if self.padding_id is not None:
            padding_mask = (option_id == self.padding_id).unsqueeze(-1)
            utils = utils.masked_fill(padding_mask, -1e6)
        if self.never_chosen_id is not None:
            never_mask = (option_id == self.never_chosen_id).unsqueeze(-1)
            utils = utils.masked_fill(never_mask, -1e6)

        probs = F.softmax(utils.squeeze(-1), dim=-1)
        return utils, probs

    def train_model(self, train_data_loader, val_data_loader=None, num_epochs=1, num_early_stopping=10):
        print("Random model doesn't require training.")
        return self, 0.0

    def test_step(self, batch):
        self.eval()
        with torch.no_grad():
            data_input, labels = batch
            logits, probs = self(data_input)
            nll = -(labels["choice_pos_onehot"] * F.log_softmax(logits.squeeze(-1), dim=-1)).sum(dim=-1)
        return nll, probs
