"""Zero-shot / few-shot information extraction powered by a PyTorch UIE model.

``UIEExtractor`` is a self-contained replacement for the PaddleNLP
``Taskflow("information_extraction", ...)`` API.  It loads converted weights
(via :mod:`converter`) and exposes a simple ``__call__`` interface that
supports **entity, relation, and event extraction** through a nested schema
tree – the same design used by the original PaddleNLP implementation.
"""

from __future__ import annotations

import os
from typing import List, Union

import numpy as np
import torch
from transformers import BertTokenizerFast, BertConfig

from .model import UIE
from .utils import (
    SchemaTree,
    build_schema_tree,
    dbc2sbc,
    auto_splitter,
    get_bool_ids_greater_than,
    get_span,
    get_id_and_prob,
)


# Default HuggingFace Hub repo that hosts the converted PyTorch weights.
# It mirrors every PaddleNLP UIE checkpoint (uie-base, uie-medium, ...).
DEFAULT_REPO = "PaddlePaddle/uie-base"
SUPPORTED_MODELS = {
    "uie-base": "PaddlePaddle/uie-base",
    "uie-medium": "PaddlePaddle/uie-medium",
    "uie-mini": "PaddlePaddle/uie-mini",
    "uie-micro": "PaddlePaddle/uie-micro",
    "uie-nano": "PaddlePaddle/uie-nano",
    "uie-base-en": "PaddlePaddle/uie-base-en",
    "uie-medical-base": "PaddlePaddle/uie-medical-base",
    "uie-m-base": "PaddlePaddle/uie-m-base",
    "uie-m-large": "PaddlePaddle/uie-m-large",
}


class UIEExtractor:
    """Zero-shot information extraction pipeline.

    Args:
        model: Either a local directory path or one of the keys in
            :data:`SUPPORTED_MODELS`.
        schema: Extraction schema, e.g. ``["时间", "选手", "赛事名称"]`` for
            entity extraction or ``[{"人物": ["出生地"]}]`` for relation
            extraction.  Can be set later via :meth:`set_schema`.
        position_prob: Threshold for start / end probability filtering.
        max_seq_len: Maximum sequence length fed to the model.
        batch_size: Inference batch size.
        device: ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``, or
            ``"auto"`` (auto-detect best available device).
        split_sentence: Whether to split long texts by sentence delimiters.
    """

    def __init__(
        self,
        model: str = "uie-base",
        schema=None,
        position_prob: float = 0.5,
        max_seq_len: int = 512,
        batch_size: int = 16,
        device: str = "auto",
        split_sentence: bool = False,
    ):
        # Resolve model path ------------------------------------------------
        model_path = SUPPORTED_MODELS.get(model, model)

        # Load tokenizer and config from HF hub / local dir
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        config = BertConfig.from_pretrained(model_path)
        self.model = UIE(config)
        self._load_weights(model_path)

        # Inference settings ------------------------------------------------
        self._position_prob = position_prob
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._split_sentence = split_sentence
        self._summary_token_num = 3  # [CLS] prompt [SEP] text [SEP]

        # Detect English model for prompt construction
        self._is_en = "en" in model_path.lower() or "base-en" in model.lower()

        # Device – supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

        # Schema
        self._schema_tree: SchemaTree | None = None
        if schema is not None:
            self.set_schema(schema)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve a device string to a ``torch.device``.

        Priority when *device* is ``"auto"``:
        1. CUDA (NVIDIA GPU)
        2. MPS  (Apple Silicon unified memory / GPU)
        3. CPU
        """
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS requires torch >= 1.12 and Apple Silicon (M1/M2/M3/M4)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_weights(self, model_path: str):
        """Load ``pytorch_model.bin`` if present."""
        pt_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pt_file):
            # weights_only=True was introduced in PyTorch 2.0 as a security
            # measure.  Fall back gracefully for older versions.
            try:
                state_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(pt_file, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            # ``position_ids`` is a non-persistent buffer that we can ignore.
            real_missing = [k for k in missing if not k.endswith("position_ids")]
            if real_missing:
                print(f"Warning: {len(real_missing)} keys missing when loading weights: {real_missing[:5]}")
            if unexpected:
                print(f"Warning: {len(unexpected)} unexpected keys: {unexpected[:5]}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_schema(self, schema):
        """Set or replace the extraction schema."""
        self._schema_tree = build_schema_tree(schema)

    def __call__(self, inputs: Union[str, List[str]]) -> List[dict]:
        """Run extraction on ``inputs``.

        Args:
            inputs: A single string or a list of strings.

        Returns:
            A list (one element per input) of ``{field: [...]}`` dicts.
        """
        if self._schema_tree is None:
            raise ValueError("Schema has not been set. Call set_schema() first.")
        if isinstance(inputs, str):
            inputs = [inputs]
        return self._multi_stage_predict(inputs)

    # ------------------------------------------------------------------
    # Multi-stage prediction (schema tree traversal)
    # ------------------------------------------------------------------
    def _multi_stage_predict(self, data: List[str]) -> List[dict]:
        results = [{} for _ in range(len(data))]
        if len(data) < 1 or self._schema_tree is None:
            return results

        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                # Root-level fields: prompt = node.name
                for one_data in data:
                    examples.append({"text": one_data, "prompt": dbc2sbc(node.name)})
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                # Nested fields: prompt = "<parent_value>的<node.name>"
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            if self._is_en:
                                if node.name.startswith("[") or "[" in node.name:
                                    prompt = f"{node.name} {p}"
                                else:
                                    prompt = node.name + " of " + p
                            else:
                                prompt = p + node.name
                            examples.append({"text": one_data, "prompt": dbc2sbc(prompt)})
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1

            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            # Merge results ---------------------------------------------------
            if not node.parent_relations:
                relations = [[] for _ in range(len(data))]
                for k, v in input_map.items():
                    for index in v:
                        if len(result_list[index]) == 0:
                            continue
                        if node.name not in results[k]:
                            results[k][node.name] = result_list[index]
                        else:
                            results[k][node.name].extend(result_list[index])
                    if node.name in results[k]:
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i]:
                            relations[k][i]["relations"] = {node.name: result_list[v[i]]}
                        elif node.name not in relations[k][i]["relations"]:
                            relations[k][i]["relations"][node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(result_list[v[i]])
                # Flatten for next level
                new_relations = [[] for _ in range(len(data))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j] and node.name in relations[i][j]["relations"]:
                            for item in relations[i][j]["relations"][node.name]:
                                new_relations[i].append(item)
                relations = new_relations

            # Build prefix for children
            prefix = [[] for _ in range(len(data))]
            for k, v in input_map.items():
                for index in v:
                    for i in range(len(result_list[index])):
                        if self._is_en:
                            prefix[k].append(" of " + result_list[index][i]["text"])
                        else:
                            prefix[k].append(result_list[index][i]["text"] + "的")

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)

        return results

    # ------------------------------------------------------------------
    # Single-stage prediction (one pass over the schema tree)
    # ------------------------------------------------------------------
    def _single_stage_predict(self, inputs: List[dict]) -> List[dict]:
        input_texts = [d["text"] for d in inputs]
        prompts = [d["prompt"] for d in inputs]
        max_prompt_len = max(len(p) for p in prompts)
        max_predict_len = self._max_seq_len - max_prompt_len - self._summary_token_num

        short_input_texts, input_mapping = auto_splitter(
            input_texts, max_predict_len, split_sentence=self._split_sentence
        )
        short_texts_prompts = []
        for k, v in input_mapping.items():
            short_texts_prompts.extend([prompts[k] for _ in range(len(v))])

        short_inputs = [
            {"text": short_input_texts[i], "prompt": short_texts_prompts[i]}
            for i in range(len(short_input_texts))
        ]

        # Batch encode ------------------------------------------------------
        sentence_ids = []
        probs = []
        for batch_start in range(0, len(short_inputs), self._batch_size):
            batch = short_inputs[batch_start: batch_start + self._batch_size]
            batch_start_probs = []
            batch_end_probs = []
            batch_offset_maps = []
            for example in batch:
                encoded = self._encode_example(example)
                batch_start_probs.append(encoded["start_prob"])
                batch_end_probs.append(encoded["end_prob"])
                batch_offset_maps.append(encoded["offset_mapping"])

            for start_prob, end_prob, offset_map in zip(
                batch_start_probs, batch_end_probs, batch_offset_maps
            ):
                start_ids_list = get_bool_ids_greater_than(
                    start_prob, limit=self._position_prob, return_prob=True
                )
                end_ids_list = get_bool_ids_greater_than(
                    end_prob, limit=self._position_prob, return_prob=True
                )
                span_set = get_span(start_ids_list, end_ids_list, with_prob=True)
                sentence_id, prob = get_id_and_prob(span_set, offset_map)
                sentence_ids.append(sentence_id)
                probs.append(prob)

        results = self._convert_ids_to_results(short_inputs, sentence_ids, probs)
        results = self._auto_joiner(results, short_input_texts, input_mapping)
        return results

    # ------------------------------------------------------------------
    # Tokenisation helper
    # ------------------------------------------------------------------
    def _encode_example(self, example: dict) -> dict:
        """Encode a single ``(prompt, text)`` pair and run the model."""
        prompt = example["prompt"]
        text = example["text"]
        # Use the tokenizer's encode_pair pattern: text=prompt, text_pair=text
        encoding = self.tokenizer(
            prompt,
            text,
            truncation=True,
            max_length=self._max_seq_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        token_type_ids = encoding["token_type_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            start_prob, end_prob = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
        start_prob = start_prob.squeeze(0).cpu().numpy()
        end_prob = end_prob.squeeze(0).cpu().numpy()
        offset_mapping = encoding["offset_mapping"].squeeze(0).cpu().numpy().tolist()
        return {
            "start_prob": start_prob,
            "end_prob": end_prob,
            "offset_mapping": offset_mapping,
        }

    # ------------------------------------------------------------------
    # Result conversion helpers
    # ------------------------------------------------------------------
    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    # Result came from the prompt itself
                    start += len(prompt) + 1
                    end += len(prompt) + 1
                    result = {"text": prompt[start:end], "probability": prob[i]}
                    result_list.append(result)
                else:
                    result = {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "probability": prob[i],
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        concat_results = []
        for k, vs in input_mapping.items():
            offset = 0
            single_results = []
            for v in vs:
                if v == 0:
                    single_results = short_results[v]
                    offset += len(short_inputs[v])
                else:
                    for i in range(len(short_results[v])):
                        if "start" not in short_results[v][i] or "end" not in short_results[v][i]:
                            continue
                        short_results[v][i]["start"] += offset
                        short_results[v][i]["end"] += offset
                    offset += len(short_inputs[v])
                    single_results.extend(short_results[v])
            concat_results.append(single_results)
        return concat_results
