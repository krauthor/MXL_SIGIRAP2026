import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.var_inf_model import Model, glorot_normal_


def softmax_loss(logits, probs, model, labels_onehot, l2_weight=0, return_component_values=False):
    logits = logits.squeeze(-1)

    nll = -(labels_onehot * F.log_softmax(logits, dim=-1)).sum(dim=-1)

    if model.user_backbone != "id":
        item_reg = torch.zeros(1, device=logits.device)
        user_reg = torch.zeros(1, device=logits.device)
        nll = nll + l2_weight * model.model_param_l2()
    else:
        item_reg = l2_weight * torch.square(model.item_embs).sum(dim=-1)
        user_reg = l2_weight * torch.square(model.user_mean).sum(dim=-1)
        nll = nll + item_reg.sum() + user_reg.sum()

    if not return_component_values:
        return nll.mean()
    else:
        return nll.mean(), probs, user_reg, item_reg


class MNL_Model(Model):

    def __init__(
        self, out_dim, num_users, num_items, no_choice_id=None, padding_id=None,
        never_chosen_id=None, has_intercept=False, l2_weight=0, max_training_time_sec=None,
        temperature=1.0, no_choice_boost=0.0, intercept_scale=1.0,
        max_seq_len=50, seq_num_heads=1, seq_num_blocks=2, seq_dropout=0.0,
        use_no_choice_rate=False,
        use_sequence_length=False,
    ):
        nn.Module.__init__(self)
        self.out_dim = out_dim
        self.num_users = num_users
        self.num_items = num_items
        self.no_choice_id = no_choice_id
        self.padding_id = padding_id
        self.never_chosen_id = never_chosen_id
        self.has_intercept = has_intercept
        self.l2_weight = l2_weight
        self.max_training_time_sec = max_training_time_sec
        self.temperature = temperature
        self.no_choice_boost = no_choice_boost
        self.intercept_scale = intercept_scale
        self._set_backbone_config(max_seq_len, seq_num_heads, seq_num_blocks,
                                  seq_dropout, use_no_choice_rate, use_sequence_length)

        self._build_item_params()
        if self.user_backbone != "id":
            self._build_user_backbone()
            self._build_user_proj(num_slots=1)
        else:
            self.user_mean = nn.Parameter(glorot_normal_(torch.empty(num_users, out_dim)))

    def get_item_embeddings(self, item_list):
        return self.item_embs[item_list]

    def draw_user_embedding_sample(self, input):
        return self._deterministic_user_embs(input, num_slots=1)

    def forward(self, input):
        option_id = input["option_id"]
        batch_size = option_id.shape[0]
        option_ids_flat = option_id.reshape(-1)
        item_embs = self.get_item_embeddings(option_ids_flat).reshape(batch_size, -1, self.out_dim)

        user_embs = self.draw_user_embedding_sample(input)[:, None, :]

        utils = torch.matmul(item_embs, user_embs.transpose(1, 2))

        if self.has_intercept:
            utils = utils + self.intercept_scale * self.get_item_intercepts(option_ids_flat).reshape(
                batch_size, -1, 1
            )

        if self.no_choice_id is not None:
            no_choice_mask = (option_id == self.no_choice_id).unsqueeze(-1)
            utils = utils.masked_fill(no_choice_mask, float(self.no_choice_boost))

        if self.padding_id is not None:
            padding_mask = (option_id == self.padding_id).unsqueeze(-1)
            utils = utils.masked_fill(padding_mask, -99.0)

        if self.never_chosen_id is not None:
            never_mask = (option_id == self.never_chosen_id).unsqueeze(-1)
            utils = utils.masked_fill(never_mask, -99.0)

        utils = utils * self.temperature

        e = torch.exp(utils.squeeze(-1))
        probs = e / torch.sum(e + 1e-10, dim=-1, keepdim=True)

        return utils, probs
