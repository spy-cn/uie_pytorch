"""Utility helpers for UIE zero-shot inference, ported from PaddleNLP.

These functions – ``SchemaTree``, ``get_bool_ids_greater_than``, ``get_span``,
``get_id_and_prob``, ``dbc2sbc`` and the auto-splitter – are direct reimplementations of the helpers used by ``paddlenlp.taskflow.information_extraction``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Schema tree – supports nested schemas for entity / relation / event
# extraction, mirroring ``paddlenlp.taskflow.utils.SchemaTree``.
# ---------------------------------------------------------------------------
class SchemaTree:
    """A node in the extraction schema tree.

    The root node's name is ``"root"``.  Each child represents a field to
    extract.  A child may itself have children, which enables relation /
    event extraction (e.g. ``{"人物": ["出生地", "职业"]}``).
    """

    def __init__(self, name: str = "root", children=None):
        self.name = name
        self.children: List[SchemaTree] = []
        self.prefix = None  # set during multi-stage traversal
        self.parent_relations = None
        self.parent = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node: "SchemaTree"):
        assert isinstance(node, SchemaTree), "children must be SchemaTree instances"
        self.children.append(node)


def build_schema_tree(schema) -> SchemaTree:
    """Build a :class:`SchemaTree` from a nested list / dict / str schema."""
    root = SchemaTree("root")
    if isinstance(schema, dict) or isinstance(schema, str):
        schema = [schema]
    for s in schema:
        root.add_child(_build_node(s))
    return root


def _build_node(s):
    node = SchemaTree(name=s) if isinstance(s, str) else None
    if isinstance(s, str):
        return node
    if isinstance(s, dict):
        for k, v in s.items():
            node = SchemaTree(name=k)
            if isinstance(v, list):
                for child_name in v:
                    if isinstance(child_name, str):
                        node.add_child(SchemaTree(child_name))
                    elif isinstance(child_name, dict):
                        for ck, cv in child_name.items():
                            child_node = SchemaTree(ck)
                            for gc in cv:
                                child_node.add_child(SchemaTree(gc))
                            node.add_child(child_node)
            return node
    raise TypeError(f"Unsupported schema element: {s!r}")


# ---------------------------------------------------------------------------
# Decoding helpers – direct ports from paddlenlp.utils.tools / ie_utils
# ---------------------------------------------------------------------------
def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """Return indices of ``probs`` that exceed ``limit``."""
    probs = np.array(probs)
    if probs.ndim > 1:
        return [get_bool_ids_greater_than(p, limit, return_prob) for p in probs]
    result = []
    for i, p in enumerate(probs):
        if p > limit:
            result.append((i, float(p)) if return_prob else i)
    return result


def get_span(start_ids, end_ids, with_prob=False):
    """Pair start / end indices into non-overlapping spans.

    Returns a ``set`` of ``(start, end)`` tuples.  When ``with_prob`` is
    ``True`` each index element is itself a ``(index, prob)`` tuple.
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]
        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        # start_id > end_id
        end_pointer += 1
    return {(couple_dict[end], end) for end in couple_dict}


def get_id_and_prob(span_set, offset_mapping):
    """Map token-level spans back to character-level (start, end) offsets."""
    offset_mapping = [list(x) for x in offset_mapping]
    # The prompt occupies tokens [1, prompt_end_token_id).
    prompt_end_token_id = offset_mapping[1:].index([0, 0])
    bias = offset_mapping[prompt_end_token_id][1] + 1
    for idx in range(1, prompt_end_token_id + 1):
        offset_mapping[idx][0] -= bias
        offset_mapping[idx][1] -= bias

    sentence_id = []
    prob = []
    for start, end in span_set:
        prob.append(start[1] * end[1])
        start_id = offset_mapping[start[0]][0]
        end_id = offset_mapping[end[0]][1]
        sentence_id.append((start_id, end_id))
    return sentence_id, prob


def dbc2sbc(text: str) -> str:
    """Convert full-width (DBCS) characters to half-width (SBCS)."""
    rs = ""
    for char in text:
        code = ord(char)
        if code == 0x3000:
            code = 0x0020
        else:
            code -= 0xFEE0
        if not (0x0021 <= code <= 0x7E):
            rs += char
            continue
        rs += chr(code)
    return rs


# ---------------------------------------------------------------------------
# Text auto-splitter – handles inputs longer than ``max_predict_len``.
# ---------------------------------------------------------------------------
def auto_splitter(input_texts: List[str], max_predict_len: int, split_sentence: bool = False):
    """Split long texts so each chunk fits within the model's window.

    Returns ``(short_texts, input_mapping)`` where ``input_mapping[i]`` is a
    list of indices into ``short_texts`` that together form the *i*-th original
    input.
    """
    short_input_texts = []
    input_mapping = {}
    cnt = 0
    for text in input_texts:
        if len(text) <= max_predict_len:
            short_input_texts.append(text)
            input_mapping[cnt] = [cnt]
            cnt += 1
        elif split_sentence:
            # Split by common Chinese / English sentence delimiters
            import re
            sentences = re.split(r"(?<=[。！？!?\n])", text)
            sentences = [s for s in sentences if s]
            temp_text = ""
            temp_list = []
            for s in sentences:
                if len(temp_text) + len(s) <= max_predict_len:
                    temp_text += s
                else:
                    if temp_text:
                        temp_list.append(temp_text)
                    temp_text = s
            if temp_text:
                temp_list.append(temp_text)
            input_mapping[cnt] = [len(short_input_texts) + i for i in range(len(temp_list))]
            short_input_texts.extend(temp_list)
            cnt += 1
        else:
            # Hard split
            parts = [text[i:i + max_predict_len] for i in range(0, len(text), max_predict_len)]
            input_mapping[cnt] = [len(short_input_texts) + i for i in range(len(parts))]
            short_input_texts.extend(parts)
            cnt += 1
    return short_input_texts, input_mapping
