"""Convert PaddleNLP UIE weights (``model_state.pdparams``) to PyTorch.

PaddleNLP's UIE stores parameters in ``.pdparams`` (pickled dicts of numpy
arrays).  Because our PyTorch :class:`~uie_pytorch.model.UIE` is a faithful
structural clone, the conversion is **pure key-name remapping** – no tensor
transposition is required (both frameworks use the same layout for ``Linear``
and ``Embedding`` layers).

Usage::

    python -m uie_pytorch.converter --model uie-base --output_dir ./weights/uie-base

After conversion, ``output_dir`` contains ``pytorch_model.bin``,
``config.json``, ``vocab.txt`` and tokenizer metadata files, ready to be
loaded with :class:`uie_pytorch.extractor.UIEExtractor`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Original PaddleNLP download URLs.
# ---------------------------------------------------------------------------
MODEL_URLS: Dict[str, Dict[str, str]] = {
    "uie-base": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-medium": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-mini": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-micro": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-nano": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-base-en": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en_v1.2/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/vocab.txt",
    },
    "uie-medical-base": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medical_base_v0.2/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
}


def _download(url: str, dest: str):
    """Download *url* to *dest*."""
    import urllib.request

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> saved to {dest} ({os.path.getsize(dest) / 1024:.0f} KB)")


def _load_pdparams(path: str) -> "OrderedDict":
    """Load a ``.pdparams`` file (pickled dict of numpy arrays)."""
    with open(path, "rb") as f:
        state = pickle.load(f)
    clean = OrderedDict()
    for key, value in state.items():
        if key.startswith("StructuredToParameterName"):
            continue  # PaddlePaddle internal metadata key
        if hasattr(value, "numpy"):
            value = value.numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array(value)
        clean[key] = value
    return clean


# ---------------------------------------------------------------------------
# Key remapping
# ---------------------------------------------------------------------------
def _convert_key(key: str) -> str:
    """Map a PaddleNLP UIE parameter key to our PyTorch UIE key.

    Our ``UIE`` model keeps the exact same layer names as PaddleNLP
    (``ernie.encoder.layers.N...``, ``linear_start``, ``linear_end``) so the
    remapping is essentially the identity function – we just need to preserve
    the keys as-is.
    """
    return key


def convert_state_dict(pdparams: dict) -> "OrderedDict":
    """Convert a PaddleNLP UIE state dict to our PyTorch UIE state dict.

    PaddlePaddle's ``nn.Linear`` stores weight with shape ``(in_features,
    out_features)`` whereas PyTorch uses ``(out_features, in_features)``.
    We transpose every ``*.weight`` tensor whose name ends with a Linear
    projection (``q_proj``, ``k_proj``, ``v_proj``, ``out_proj``,
    ``linear1``, ``linear2``, ``linear_start``, ``linear_end``, ``dense``)
    to correct this.  Embedding weights are 2-D but stored identically
    (vocab x hidden) in both frameworks, so they are left untouched.
    """
    # Weight suffixes that correspond to nn.Linear (not nn.Embedding).
    LINEAR_SUFFIXES = (
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight",
        "linear1.weight", "linear2.weight",
        "linear_start.weight", "linear_end.weight",
        "pooler.dense.weight",
    )

    new_state = OrderedDict()
    skipped = []
    transposed = 0
    for old_key, array in pdparams.items():
        if old_key.startswith("StructuredToParameterName"):
            skipped.append(old_key)
            continue
        new_key = _convert_key(old_key)
        tensor = torch.from_numpy(array.copy())
        # Detect Linear weights by suffix and shape (2-D).
        if any(new_key.endswith(suf) for suf in LINEAR_SUFFIXES) and tensor.dim() == 2:
            # PaddleNLP (in, out) -> PyTorch (out, in)
            tensor = tensor.t().contiguous()
            transposed += 1
        new_state[new_key] = tensor
    print(f"  (transposed {transposed} Linear weights)")
    if skipped:
        print(f"  (skipped {len(skipped)} metadata keys)")
    return new_state


def paddle_config_to_bert(config: dict) -> dict:
    """Map a PaddleNLP ErnieConfig dict to HuggingFace BertConfig fields,
    preserving ERNIE-specific ``task_type_vocab_size``."""
    bert_config = {
        "architectures": ["UIE"],
        "model_type": "bert",
        "hidden_size": config.get("hidden_size", 768),
        "num_hidden_layers": config.get("num_hidden_layers", 12),
        "num_attention_heads": config.get("num_attention_heads", 12),
        "intermediate_size": config.get("intermediate_size", 3072),
        "vocab_size": config.get("vocab_size", 40000),
        "max_position_embeddings": config.get("max_position_embeddings", 2048),
        "hidden_act": config.get("hidden_act", "gelu"),
        "hidden_dropout_prob": config.get("hidden_dropout_prob", 0.1),
        "attention_probs_dropout_prob": config.get("attention_probs_dropout_prob", 0.1),
        "type_vocab_size": config.get("type_vocab_size", 4),
        "task_type_vocab_size": config.get("task_type_vocab_size", 3),
        "layer_norm_eps": config.get("layer_norm_eps", 1e-12),
        "pad_token_id": config.get("pad_token_id", 0),
        "initializer_range": 0.02,
    }
    return bert_config


def convert_model(model_name: str, output_dir: str):
    """Download, convert and save a UIE model to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)

    urls = MODEL_URLS[model_name]
    pdparams_path = os.path.join(output_dir, "model_state.pdparams")
    config_path = os.path.join(output_dir, "source_config.json")
    vocab_path = os.path.join(output_dir, "vocab.txt")

    if not os.path.exists(pdparams_path):
        _download(urls["model_state"], pdparams_path)
    if not os.path.exists(config_path):
        _download(urls["config"], config_path)
    if not os.path.exists(vocab_path):
        _download(urls["vocab_file"], vocab_path)

    # Load and convert ---------------------------------------------------
    pd_state = _load_pdparams(pdparams_path)
    pt_state = convert_state_dict(pd_state)

    with open(config_path) as f:
        paddle_cfg = json.load(f)
    bert_cfg = paddle_config_to_bert(paddle_cfg)

    torch.save(pt_state, os.path.join(output_dir, "pytorch_model.bin"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(bert_cfg, f, indent=2)

    # Tokenizer metadata
    special = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special, f, indent=2)
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"do_lower_case": True, "model_max_length": 512}, f, indent=2)

    print(f"\n✅ Converted {model_name} -> {output_dir}")
    print(f"   pytorch_model.bin  ({len(pt_state)} tensors)")
    print(f"   config.json")
    print(f"   vocab.txt")


def main():
    parser = argparse.ArgumentParser(description="Convert PaddleNLP UIE to PyTorch")
    parser.add_argument("--model", default="uie-base", choices=list(MODEL_URLS.keys()))
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
