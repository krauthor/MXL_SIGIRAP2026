import math

import torch
import torch.nn as nn

from src.models.elbo import (
    kl_gaussian,
    kl_gaussian_aat,
)
from src.models.sasrec_backbone import SASRecBackbone


SEQUENCE_BACKBONES = ("sasrec",)


def he_normal_(tensor):
    fan_in = tensor.shape[-2] if tensor.ndim >= 2 else tensor.shape[-1]
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        tensor.normal_(0.0, std)
    return tensor


def glorot_normal_(tensor):
    if tensor.ndim >= 2:
        fan_in, fan_out = tensor.shape[-2], tensor.shape[-1]
    else:
        fan_in = fan_out = tensor.shape[-1]
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.normal_(0.0, std)
    return tensor


class Model(nn.Module):

    _force_user_backbone = None

    def __init__(
        self,
        out_dim,
        num_prior_samples,
        num_users,
        num_items,
        no_choice_id=None,
        padding_id=None,
        never_chosen_id=None,
        beta_max=0.1,
        has_intercept=False,
        l2_weight=0,
        l2_num_prior_samples=4,
        use_output_layer_norm=False,
        cov_type="diag",
        cov_rank=0,
        max_training_time_sec=None,
        temperature=1.0,
        no_choice_boost=0.0,
        intercept_scale=1.0,
        max_seq_len=50,
        seq_num_heads=1,
        seq_num_blocks=2,
        seq_dropout=0.0,
        use_no_choice_rate=False,
        use_sequence_length=False,
    ):
        super().__init__()

        if cov_rank > out_dim:
            print(f"cov_rank ({cov_rank}) > out_dim ({out_dim}). Clamping to out_dim.")
            cov_rank = out_dim

        self.out_dim = out_dim
        self.num_prior_samples = num_prior_samples
        self.num_users = num_users
        self.num_items = num_items
        self.no_choice_id = no_choice_id
        self.padding_id = padding_id
        self.never_chosen_id = never_chosen_id
        self.beta_max = beta_max
        self.has_intercept = has_intercept
        self.l2_weight = l2_weight
        self.l2_num_prior_samples = l2_num_prior_samples
        self.use_output_layer_norm = use_output_layer_norm
        self.cov_type = cov_type
        self.cov_rank = cov_rank
        self.max_training_time_sec = max_training_time_sec
        self.temperature = temperature
        self.no_choice_boost = no_choice_boost
        self.intercept_scale = intercept_scale
        self.user_backbone = self._force_user_backbone if self._force_user_backbone is not None else "id"
        self.max_seq_len = max_seq_len
        self.seq_num_heads = seq_num_heads
        self.seq_num_blocks = seq_num_blocks
        self.seq_dropout = seq_dropout
        self.history_pad_id = self.padding_id if self.padding_id is not None else 0
        self.use_no_choice_rate = use_no_choice_rate
        self.use_sequence_length = use_sequence_length
        self.user_feat_dim = out_dim + (1 if use_no_choice_rate else 0) + (1 if use_sequence_length else 0)

        self._build_user_params()
        self._build_item_params()
        self.output_layer_norm = nn.LayerNorm(self.out_dim) if self.use_output_layer_norm else None

    def _build_user_params(self):
        ld = self.out_dim

        if self.user_backbone != "id":
            self._build_user_backbone()
            fd = self.user_feat_dim
            self.user_mean_proj = nn.Linear(fd, ld)
            r = self.cov_rank if self.cov_rank > 0 else ld
            if self.cov_type == "aat":
                self.aat_raw_proj = nn.Linear(fd, ld * r)
                self.aat_log_sigma_proj = nn.Linear(fd, ld)
            elif self.cov_type == "aat_shared":
                self.shared_aat_raw = nn.Parameter(torch.zeros(1, 1, ld, r))
                self.shared_aat_log_sigma = nn.Parameter(torch.zeros(1, 1, ld))
                self.log_d_proj = nn.Linear(fd, ld)
            else:
                self.log_std_proj = nn.Linear(fd, ld)
            if self.out_dim == 1:
                self.sample_bias = nn.Parameter(torch.zeros(1))
            return

        self.user_mean = nn.Parameter(he_normal_(torch.empty(self.num_users, 1, ld)))

        self.log_user_std = None
        self.user_aat_raw = None
        self.user_aat_log_sigma = None
        self.shared_aat_raw = None
        self.shared_aat_log_sigma = None
        self.user_log_d = None

        if self.cov_type == "aat":
            r = self.cov_rank if self.cov_rank > 0 else ld
            self.user_aat_raw = nn.Parameter(torch.zeros(self.num_users, 1, ld, r))
            self.user_aat_log_sigma = nn.Parameter(torch.zeros(self.num_users, 1, ld))
        elif self.cov_type == "aat_shared":
            r = self.cov_rank if self.cov_rank > 0 else ld
            self.shared_aat_raw = nn.Parameter(torch.zeros(1, 1, ld, r))
            self.shared_aat_log_sigma = nn.Parameter(torch.zeros(1, 1, ld))
            self.user_log_d = nn.Parameter(torch.zeros(self.num_users, 1, ld))
        else:
            self.log_user_std = nn.Parameter(torch.full((self.num_users, 1, ld), float(math.log(1.0))))

        if self.out_dim == 1:
            self.sample_bias = nn.Parameter(torch.zeros(1))

    def _build_item_params(self):
        self.item_emb_scale = nn.Parameter(torch.tensor(1.0))
        self.item_intercepts = nn.Parameter(torch.zeros(self.num_items, 1))
        self.item_embs = nn.Parameter(he_normal_(torch.empty(self.num_items, self.out_dim)))

    def get_item_embeddings(self, item_list):
        return self.item_embs[item_list] * self.item_emb_scale

    def get_item_intercepts(self, item_list):
        return self.item_intercepts[item_list]

    def _sample_and_reshape(
        self, user_mean, scale_param, num_samples, num_prior_samples,
        residual_log_sigma=None,
    ):
        device = user_mean.device
        dim = self.out_dim
        if self.cov_type in ("aat", "aat_shared"):
            r = scale_param.shape[-1]
            eps1 = torch.randn(num_samples, num_prior_samples, r, device=device)
            eps2 = torch.randn(num_samples, num_prior_samples, dim, device=device)
            d = (
                user_mean
                + torch.einsum('...ij,...j->...i', scale_param, eps1)
                + torch.exp(residual_log_sigma) * eps2
            )
        else:
            eps = torch.randn(num_samples, num_prior_samples, dim, device=device)
            d = user_mean + torch.exp(scale_param) * eps
        return d.reshape(-1, dim)

    def _user_id_params(self, user_ids):
        user_mean = self.user_mean[user_ids]
        residual_log_sigma = None
        if self.cov_type == "aat":
            scale_param = self.user_aat_raw[user_ids]
            residual_log_sigma = self.user_aat_log_sigma[user_ids]
        elif self.cov_type == "aat_shared":
            user_log_d = self.user_log_d[user_ids]
            d = torch.exp(user_log_d)
            scale_param = d[..., None] * self.shared_aat_raw
            residual_log_sigma = user_log_d + self.shared_aat_log_sigma
        else:
            scale_param = self.log_user_std[user_ids]
        return user_mean, scale_param, residual_log_sigma

    def _user_seq_params(self, input):
        h = self._encode_history(input)
        ld = self.out_dim
        mean = self.user_mean_proj(h).unsqueeze(1)
        residual_log_sigma = None
        if self.cov_type == "aat":
            scale_param = self.aat_raw_proj(h).reshape(mean.shape[0], 1, ld, -1)
            residual_log_sigma = self.aat_log_sigma_proj(h).unsqueeze(1)
        elif self.cov_type == "aat_shared":
            log_d = self.log_d_proj(h).unsqueeze(1)
            d = torch.exp(log_d)
            scale_param = d[..., None] * self.shared_aat_raw
            residual_log_sigma = log_d + self.shared_aat_log_sigma
        else:
            scale_param = self.log_std_proj(h).unsqueeze(1)
        return mean, scale_param, residual_log_sigma

    def _kl_from_params(self, mean, scale_param, residual_log_sigma):
        if self.cov_type in ("aat", "aat_shared"):
            return kl_gaussian_aat(mean, scale_param, residual_log_sigma)
        else:
            return kl_gaussian(mean, torch.exp(scale_param))

    def draw_user_embedding_sample(self, user_ids, num_samples, num_prior_samples, ignore_user_id=False):
        user_mean, scale_param, residual_log_sigma = self._user_id_params(user_ids)

        if ignore_user_id:
            user_mean = 0.0 * user_mean
            scale_param = 0.0 * scale_param
            if residual_log_sigma is not None:
                residual_log_sigma = 0.0 * residual_log_sigma

        s = self._sample_and_reshape(
            user_mean, scale_param, num_samples, num_prior_samples,
            residual_log_sigma=residual_log_sigma,
        )

        return self._decode_and_reshape(s, num_samples, num_prior_samples)

    def _decode_and_reshape(self, s, num_samples, num_prior_samples):
        out = s.reshape(num_samples, num_prior_samples, -1)

        if self.out_dim == 1:
            mean = out.mean(dim=1, keepdim=True)
            std = out.std(dim=1, keepdim=True, unbiased=False)
            out = (out - mean) / (std + 1e-6)
            out = out + self.sample_bias

        return out

    def _build_user_backbone(self):
        if self.user_backbone == "sasrec":
            backbone_cls = SASRecBackbone
        else:
            raise ValueError(
                f"Unknown sequence backbone {self.user_backbone!r}; "
                f"expected one of {SEQUENCE_BACKBONES}."
            )
        self.user_backbone_module = backbone_cls(
            emb_dim=self.out_dim, max_seq_len=self.max_seq_len,
            num_heads=self.seq_num_heads, num_blocks=self.seq_num_blocks,
            dropout=self.seq_dropout,
        )

    def _encode_history(self, input):
        history = input["history"]
        batch_size, seq_len = history.shape
        hist_embs = self.get_item_embeddings(history.reshape(-1)).reshape(batch_size, seq_len, self.out_dim)
        valid_mask = history != self.history_pad_id
        h = self.user_backbone_module(hist_embs, valid_mask)
        if self.use_no_choice_rate:
            h = torch.cat([h, input["no_choice_rate"].unsqueeze(-1).to(h.dtype)], dim=-1)
        if self.use_sequence_length:
            lengths = valid_mask.sum(dim=1, keepdim=True).to(h.dtype)
            h = torch.cat([h, torch.log1p(lengths)], dim=-1)
        return h

    def model_param_l2(self):
        reg = torch.zeros((), device=self.item_embs.device)
        for name, p in self.named_parameters():
            if p.dim() >= 2 and "item_embs" not in name and "position_embedding" not in name:
                reg = reg + torch.mean(torch.square(p))
        return reg

    def _set_backbone_config(self, max_seq_len=50, seq_num_heads=1, seq_num_blocks=2, seq_dropout=0.0,
                             use_no_choice_rate=False, use_sequence_length=False):
        self.user_backbone = self._force_user_backbone if self._force_user_backbone is not None else "id"
        self.max_seq_len = max_seq_len
        self.seq_num_heads = seq_num_heads
        self.seq_num_blocks = seq_num_blocks
        self.seq_dropout = seq_dropout
        self.history_pad_id = self.padding_id if self.padding_id is not None else 0
        self.use_no_choice_rate = use_no_choice_rate
        self.use_sequence_length = use_sequence_length
        self.user_feat_dim = self.out_dim + (1 if use_no_choice_rate else 0) + (1 if use_sequence_length else 0)

    def _build_user_proj(self, num_slots=1):
        self.user_proj = nn.Linear(self.user_feat_dim, num_slots * self.out_dim)

    def _deterministic_user_embs(self, input, num_slots=1):
        if self.user_backbone != "id":
            u = self.user_proj(self._encode_history(input))
            if num_slots > 1:
                u = u.reshape(-1, num_slots, self.out_dim)
            return u
        return self.user_mean[input["user_ids"]]

    def get_all_user_out_vectors(self, num_samples_per_user):
        return self.draw_user_embedding_sample(
            user_ids=torch.arange(self.num_users, device=self.user_mean.device),
            num_samples=self.num_users,
            num_prior_samples=num_samples_per_user,
        )

    def forward(self, input, return_user_embs_no_id=False):
        option_id = input["option_id"]
        batch_size = option_id.shape[0]
        option_ids_flat = option_id.reshape(-1)
        item_embs = self.get_item_embeddings(option_ids_flat).reshape(batch_size, -1, self.out_dim)

        if self.user_backbone != "id":
            return self._forward_seq(input, item_embs, option_id, option_ids_flat, batch_size)

        user_embs = self.draw_user_embedding_sample(
            num_samples=batch_size,
            user_ids=input["user_ids"],
            num_prior_samples=self.num_prior_samples,
        )
        utils = torch.matmul(item_embs, user_embs.transpose(1, 2))

        if self.has_intercept:
            utils = utils + self.intercept_scale * self.get_item_intercepts(option_ids_flat).reshape(
                batch_size, -1, 1
            )

        if self.no_choice_id is not None and self.no_choice_boost != 0.0:
            no_choice_mask = (option_id == self.no_choice_id).unsqueeze(-1)
            utils = torch.where(no_choice_mask, utils + self.no_choice_boost, utils)

        if self.padding_id is not None:
            padding_mask = (option_id == self.padding_id).unsqueeze(-1)
            utils = utils.masked_fill(padding_mask, -99.0)

        if self.never_chosen_id is not None:
            never_mask = (option_id == self.never_chosen_id).unsqueeze(-1)
            utils = utils.masked_fill(never_mask, -99.0)

        utils = utils * self.temperature

        c = torch.max(utils, dim=-2, keepdim=True).values
        probs = torch.exp(
            utils - (c + torch.logsumexp(utils - c, dim=-2, keepdim=True))
        ).mean(dim=-1)
        probs = probs * (1 - 1e-10) + 1e-10

        if return_user_embs_no_id:
            all_user_embs = self.draw_user_embedding_sample(
                num_samples=1,
                user_ids=torch.arange(1, device=utils.device),
                num_prior_samples=self.l2_num_prior_samples,
                ignore_user_id=True,
            )
            return utils, probs, all_user_embs
        return utils, probs

    def _forward_seq(self, input, item_embs, option_id, option_ids_flat, batch_size):
        mean, scale_param, residual_log_sigma = self._user_seq_params(input)
        s = self._sample_and_reshape(
            mean, scale_param, batch_size, self.num_prior_samples,
            residual_log_sigma=residual_log_sigma,
        )
        user_embs = self._decode_and_reshape(s, batch_size, self.num_prior_samples)

        utils = torch.matmul(item_embs, user_embs.transpose(1, 2))

        if self.has_intercept:
            utils = utils + self.intercept_scale * self.get_item_intercepts(option_ids_flat).reshape(
                batch_size, -1, 1
            )
        if self.no_choice_id is not None and self.no_choice_boost != 0.0:
            no_choice_mask = (option_id == self.no_choice_id).unsqueeze(-1)
            utils = torch.where(no_choice_mask, utils + self.no_choice_boost, utils)
        if self.padding_id is not None:
            utils = utils.masked_fill((option_id == self.padding_id).unsqueeze(-1), -99.0)
        if self.never_chosen_id is not None:
            utils = utils.masked_fill((option_id == self.never_chosen_id).unsqueeze(-1), -99.0)

        utils = utils * self.temperature

        c = torch.max(utils, dim=-2, keepdim=True).values
        probs = torch.exp(utils - (c + torch.logsumexp(utils - c, dim=-2, keepdim=True))).mean(dim=-1)
        probs = probs * (1 - 1e-10) + 1e-10

        kl = self._kl_from_params(mean, scale_param, residual_log_sigma)
        return utils, probs, kl

    def kl_divergence(self):
        user_ids = torch.arange(self.num_users, device=self.user_mean.device)
        mean, scale_param, residual_log_sigma = self._user_id_params(user_ids)
        return self._kl_from_params(mean, scale_param, residual_log_sigma)

    def kl_divergence_for_users(self, user_ids):
        mean, scale_param, residual_log_sigma = self._user_id_params(user_ids)
        return self._kl_from_params(mean, scale_param, residual_log_sigma)

    def triton_logprobs(self, input, target_idx, chunk_samples=None):
        from src.models.fused_mxl_triton import triton_mxl_logprobs

        assert self.user_backbone == "id", "Triton path supports the id user backbone only"
        assert self.cov_type == "diag", "Triton path supports diagonal covariance only"
        assert self.never_chosen_id is None, "Triton path does not apply never-chosen masking"
        assert float(self.temperature) > 0.0, "Triton path needs a positive temperature"
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            assert capability >= (8, 0), (
                f"Triton path needs compute capability >= 8.0 (bf16 MMA); got {capability}"
            )

        user_ids = input["user_ids"]
        option_id = input["option_id"]
        B = option_id.shape[0]
        mu = self.user_mean[user_ids].reshape(B, self.out_dim)
        log_sigma = self.log_user_std[user_ids].reshape(B, self.out_dim)
        item_table = self.item_emb_scale * self.item_embs

        valid_count = None
        if self.padding_id is not None:
            valid_count = (option_id != self.padding_id).sum(dim=1).to(torch.int32)

        bias = None
        if self.has_intercept:
            bias = self.intercept_scale * self.get_item_intercepts(option_id.reshape(-1)).reshape(B, -1)
        if self.no_choice_id is not None and float(self.no_choice_boost) != 0.0:
            boost = self.no_choice_boost * (option_id == self.no_choice_id).to(mu.dtype)
            bias = boost if bias is None else bias + boost

        temperature = float(self.temperature)
        if temperature != 1.0:
            mu = mu * temperature
            log_sigma = log_sigma + math.log(temperature)
            if bias is not None:
                bias = bias * temperature

        if self.out_dim < 16:
            pad = 16 - self.out_dim
            mu = torch.nn.functional.pad(mu, (0, pad))
            log_sigma = torch.nn.functional.pad(log_sigma, (0, pad))
            item_table = torch.nn.functional.pad(item_table, (0, pad))

        return triton_mxl_logprobs(
            mu, log_sigma, item_table, option_id, target_idx.to(torch.long),
            self.num_prior_samples,
            bias=bias, valid_count=valid_count, chunk_samples=chunk_samples,
        )
