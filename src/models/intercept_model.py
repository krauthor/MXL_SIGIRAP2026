import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.var_inf_model import Model


def softmax_loss(logits, probs, model, labels_onehot, l2_weight=0, return_component_values=False):
    logits = logits.squeeze(-1)

    nll = -(labels_onehot * F.log_softmax(logits, dim=-1)).sum(dim=-1)

    if not return_component_values:
        return nll.mean()
    else:
        return nll.mean(), probs, 0, 0


class Intercept_Model(Model):

    def __init__(
        self, num_users, num_items, no_choice_id=None, padding_id=None, never_chosen_id=None,
        max_training_time_sec=None,
    ):
        nn.Module.__init__(self)
        self.num_users = num_users
        self.num_items = num_items
        self.no_choice_id = no_choice_id
        self.padding_id = padding_id
        self.never_chosen_id = never_chosen_id
        self.max_training_time_sec = max_training_time_sec

        self.item_intercepts = nn.Parameter(torch.zeros(num_items, 1))

    def get_item_intercepts(self, item_list):
        return self.item_intercepts[item_list]

    def forward(self, input):
        option_id = input["option_id"]
        batch_size = option_id.shape[0]
        option_ids_flat = option_id.reshape(-1)

        utils = self.get_item_intercepts(option_ids_flat).reshape(batch_size, -1, 1)

        if self.no_choice_id is not None:
            no_choice_mask = (option_id == self.no_choice_id).unsqueeze(-1)
            utils = utils.masked_fill(no_choice_mask, 0.0)

        if self.padding_id is not None:
            padding_mask = (option_id == self.padding_id).unsqueeze(-1)
            utils = utils.masked_fill(padding_mask, -99.0)

        if self.never_chosen_id is not None:
            never_mask = (option_id == self.never_chosen_id).unsqueeze(-1)
            utils = utils.masked_fill(never_mask, -99.0)

        probs = F.softmax(utils.squeeze(-1), dim=-1)

        return utils, probs
