"""UIE (Universal Information Extraction) model implementation in PyTorch.

Faithful reimplementation of the PaddleNLP UIE model.  All submodule names
mirror PaddleNLP's original naming so converted weights load with **zero** key
remapping.

Reference: https://github.com/PaddlePaddle/PaddleNLP
Original paper: Lu et al., "Unified Structure Generation for Universal
Information Extraction", ACL 2022.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig


# ---------------------------------------------------------------------------
# Embeddings – word + position + token-type + task-type (ERNIE specific)
# ---------------------------------------------------------------------------
class ErnieEmbeddings(nn.Module):
    """ERNIE embedding layer.

    Combines word, position, token-type and **task-type** embeddings.
    The LayerNorm is named ``layer_norm`` (lower-case) to match PaddleNLP.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # ERNIE-specific task type embeddings (task_type_vocab_size default 3)
        task_type_vocab_size = getattr(config, "task_type_vocab_size", 3)
        self.task_type_embeddings = nn.Embedding(task_type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if task_ids is None:
            task_ids = torch.zeros_like(input_ids)

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
            + self.task_type_embeddings(task_ids)
        )
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# ---------------------------------------------------------------------------
# Multi-head attention – ``self_attn`` module with q/k/v/out_proj children
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Standard multi-head attention matching PaddleNLP's ``MultiHeadAttention``.

    The module exposes ``q_proj``, ``k_proj``, ``v_proj`` and ``out_proj``
    as direct children so the state-dict keys are
    ``self_attn.q_proj.weight`` etc.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(self, query: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(query))
        v = self._split_heads(self.v_proj(query))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.embed_dim,))
        return self.out_proj(context)


class TransformerEncoderLayer(nn.Module):
    """A single transformer encoder layer with PaddleNLP naming:

    * ``self_attn`` – :class:`MultiHeadAttention` (children q/k/v/out_proj)
    * ``norm1`` – post-attention LayerNorm
    * ``linear1`` / ``linear2`` – feed-forward network
    * ``norm2`` – post-FFN LayerNorm

    ERNIE uses **post-LayerNorm**: residual → add → norm.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention sub-layer (residual inside attention via out_proj only)
        attn_out = self.self_attn(hidden_states, attention_mask)
        attn_out = self.dropout(attn_out)
        h = self.norm1(hidden_states + attn_out)

        # FFN sub-layer
        ffn_out = self.activation(self.linear1(h))
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.dropout(ffn_out)
        out = self.norm2(h + ffn_out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class ErniePooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(hidden_states[:, 0]))


class ErnieModel(nn.Module):
    """Minimal ERNIE encoder that mirrors ``paddlenlp.transformers.ErnieModel``.

    All submodule names match PaddleNLP so converted ``.pdparams`` load
    without any key remapping.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ErnieEmbeddings(config)
        self.encoder = TransformerEncoder(config)
        self.pooler = ErniePooler(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # Build additive mask: (batch, 1, 1, seq_len)
        extended = attention_mask[:, None, None, :].to(dtype=self.embeddings.word_embeddings.weight.dtype)
        extended = (1.0 - extended) * torch.finfo(extended.dtype).min

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        sequence_output = self.encoder(embedding_output, extended)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class UIE(nn.Module):
    """ERNIE + two linear heads for universal information extraction.

    Returns ``(start_prob, end_prob)`` of shape ``(batch, seq_len)`` via
    sigmoid (positions are treated independently).

    Args:
        config: a :class:`~transformers.BertConfig` (ERNIE-compatible).
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.ernie = ErnieModel(config)
        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass, returning ``(start_prob, end_prob)``."""
        sequence_output, _ = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        start_prob = self.sigmoid(self.linear_start(sequence_output).squeeze(-1))
        end_prob = self.sigmoid(self.linear_end(sequence_output).squeeze(-1))
        return start_prob, end_prob
