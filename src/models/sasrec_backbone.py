# The transformer internals (MultiHeadAttention, TransformerBlock) and the encoder forward are
# copied near-verbatim from Aleksandr Petrov's gSASRec-PyTorch reference implementation
# (Petrov & Macdonald, "gSASRec: Reducing Overconfidence in Sequential Recommendation with
# Negative Sampling", RecSys 2023):
#   https://github.com/asash/gSASRec-pytorch/blob/main/transformer_decoder.py
#   https://github.com/asash/gSASRec-pytorch/blob/main/gsasrec.py
# gSASRec-PyTorch is licensed under the Apache License, Version 2.0. This file contains modified
# portions of that code and is therefore licensed under the Apache License, Version 2.0, not under
# the MIT License of the rest of this repository. A copy of the Apache License is in
# LICENSE-Apache-2.0.txt at the root of this repository.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate=0.5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, causality=False):
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.val_proj(keys)

        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)

        outputs = torch.matmul(Q_, K_.transpose(1, 2))
        outputs = outputs / (K_.size(-1) ** 0.5)

        key_masks = torch.sign(torch.sum(torch.abs(keys), dim=-1))
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)
        outputs = outputs.masked_fill(key_masks == 0, float('-inf'))

        if causality:
            diag_vals = torch.ones_like(outputs[0])
            tril = torch.tril(diag_vals)
            masks = tril[None, :, :].repeat(outputs.size(0), 1, 1)
            outputs = outputs.masked_fill(masks == 0, float('-inf'))

        outputs = F.softmax(outputs, dim=-1)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)

        query_masks = torch.sign(torch.sum(torch.abs(queries), dim=-1))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))
        outputs *= query_masks

        attention_chunks = outputs.chunk(self.num_heads, dim=0)
        attention_weights = torch.stack(attention_chunks, dim=1)

        outputs = self.dropout(outputs)
        outputs = torch.matmul(outputs, V_)
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=2)
        return outputs, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout_rate=0.5, causality=True):
        super(TransformerBlock, self).__init__()

        self.first_norm = nn.LayerNorm(dim)
        self.second_norm = nn.LayerNorm(dim)

        self.multihead_attention = MultiHeadAttention(dim, num_heads, dropout_rate)

        self.dense1 = nn.Linear(dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.causality = causality

    def forward(self, seq, mask=None):
        x = self.first_norm(seq)
        queries = x
        keys = seq
        x, attentions = self.multihead_attention(queries, keys, self.causality)

        x = x + queries
        x = self.second_norm(x)

        residual = x
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        x = x + residual

        if mask is not None:
            x *= mask

        return x, attentions


class SASRecBackbone(nn.Module):

    def __init__(self, emb_dim, max_seq_len, num_heads=1, num_blocks=2, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq_len = max_seq_len
        self.position_embedding = nn.Embedding(max_seq_len, emb_dim)
        self.embeddings_dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, num_heads, emb_dim, dropout) for _ in range(num_blocks)]
        )
        self.seq_norm = nn.LayerNorm(emb_dim)
        self.empty_user_emb = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, item_emb_seq, valid_mask):
        batch, seq_len, _ = item_emb_seq.shape
        device = item_emb_seq.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        seq = item_emb_seq + self.position_embedding(positions)
        seq = self.embeddings_dropout(seq)
        mask = valid_mask.unsqueeze(-1).to(seq.dtype)
        seq = seq * mask

        for block in self.transformer_blocks:
            seq, _ = block(seq, mask)
        seq = self.seq_norm(seq)

        lengths = valid_mask.sum(dim=1)
        last_idx = (lengths - 1).clamp(min=0)
        h = seq[torch.arange(batch, device=device), last_idx]
        h = torch.where((lengths == 0).unsqueeze(-1), self.empty_user_emb.to(seq.dtype), h)
        return h
