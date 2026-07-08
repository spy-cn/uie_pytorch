"""Test UIE model architecture: forward pass, shapes, parameter count."""

import os
import sys

import pytest
import torch
from transformers import BertConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uie_pytorch.model import UIE, ErnieModel, ErnieEmbeddings


@pytest.fixture
def config():
    """A small test config."""
    return BertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
        type_vocab_size=4,
        task_type_vocab_size=3,
        layer_norm_eps=1e-12,
        pad_token_id=0,
    )


class TestErnieEmbeddings:
    def test_forward_shape(self, config):
        emb = ErnieEmbeddings(config)
        input_ids = torch.randint(0, 1000, (4, 32))
        output = emb(input_ids)
        assert output.shape == (4, 32, 64)

    def test_has_task_type_embeddings(self, config):
        emb = ErnieEmbeddings(config)
        assert hasattr(emb, "task_type_embeddings")
        assert emb.task_type_embeddings.num_embeddings == 3


class TestErnieModel:
    def test_forward_returns_sequence_and_pooled(self, config):
        model = ErnieModel(config)
        input_ids = torch.randint(0, 1000, (4, 32))
        attention_mask = torch.ones(4, 32, dtype=torch.long)
        seq_out, pool_out = model(input_ids, attention_mask=attention_mask)
        assert seq_out.shape == (4, 32, 64)
        assert pool_out.shape == (4, 64)


class TestUIE:
    def test_forward_returns_start_end_probs(self, config):
        model = UIE(config)
        input_ids = torch.randint(0, 1000, (4, 32))
        attention_mask = torch.ones(4, 32, dtype=torch.long)
        start_prob, end_prob = model(input_ids=input_ids, attention_mask=attention_mask)
        assert start_prob.shape == (4, 32)
        assert end_prob.shape == (4, 32)

    def test_probs_are_in_0_1_range(self, config):
        """Sigmoid output must be between 0 and 1."""
        model = UIE(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        start_prob, end_prob = model(input_ids=input_ids, attention_mask=attention_mask)
        assert (start_prob >= 0).all() and (start_prob <= 1).all()
        assert (end_prob >= 0).all() and (end_prob <= 1).all()

    def test_param_naming_matches_paddlenlp(self, config):
        """Verify key names match PaddleNLP so converted weights load without remapping."""
        model = UIE(config)
        keys = dict(model.named_parameters()).keys()
        key_set = set(keys)

        # Embedding keys
        assert "ernie.embeddings.word_embeddings.weight" in key_set
        assert "ernie.embeddings.task_type_embeddings.weight" in key_set
        assert "ernie.embeddings.layer_norm.weight" in key_set

        # Encoder layer keys (layer 0)
        assert "ernie.encoder.layers.0.self_attn.q_proj.weight" in key_set
        assert "ernie.encoder.layers.0.self_attn.k_proj.weight" in key_set
        assert "ernie.encoder.layers.0.self_attn.v_proj.weight" in key_set
        assert "ernie.encoder.layers.0.self_attn.out_proj.weight" in key_set
        assert "ernie.encoder.layers.0.norm1.weight" in key_set
        assert "ernie.encoder.layers.0.linear1.weight" in key_set
        assert "ernie.encoder.layers.0.linear2.weight" in key_set
        assert "ernie.encoder.layers.0.norm2.weight" in key_set

        # Extraction heads
        assert "linear_start.weight" in key_set
        assert "linear_end.weight" in key_set

    def test_gradient_flows(self, config):
        """Ensure backprop works without errors."""
        model = UIE(config)
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        start_prob, end_prob = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = start_prob.sum() + end_prob.sum()
        loss.backward()
        # Check that at least one gradient is non-zero
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.parameters()
        )
        assert has_grad
