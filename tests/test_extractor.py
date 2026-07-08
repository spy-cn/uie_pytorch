"""Test UIE utility functions and end-to-end extraction.

These tests use the real uie-base weights if available; otherwise they
are skipped with a warning.
"""

import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uie_pytorch.utils import (
    SchemaTree,
    build_schema_tree,
    get_bool_ids_greater_than,
    get_span,
    get_id_and_prob,
    dbc2sbc,
    auto_splitter,
)


WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "uie-base")
HAS_WEIGHTS = os.path.exists(os.path.join(WEIGHTS_DIR, "pytorch_model.bin"))


# ---------------------------------------------------------------------------
# SchemaTree tests
# ---------------------------------------------------------------------------
class TestSchemaTree:
    def test_flat_schema(self):
        tree = build_schema_tree(["时间", "地点", "人物"])
        assert tree.name == "root"
        assert len(tree.children) == 3
        assert [c.name for c in tree.children] == ["时间", "地点", "人物"]

    def test_nested_schema(self):
        tree = build_schema_tree([{"人物": ["出生地", "职业"]}])
        assert len(tree.children) == 1
        node = tree.children[0]
        assert node.name == "人物"
        assert len(node.children) == 2
        assert [c.name for c in node.children] == ["出生地", "职业"]

    def test_add_child(self):
        tree = SchemaTree("root")
        child = SchemaTree("child1")
        tree.add_child(child)
        assert len(tree.children) == 1
        assert tree.children[0].name == "child1"


# ---------------------------------------------------------------------------
# Decoding utility tests
# ---------------------------------------------------------------------------
class TestGetBoolIdsGreaterThan:
    def test_basic(self):
        probs = [0.1, 0.6, 0.3, 0.8]
        result = get_bool_ids_greater_than(probs, limit=0.5)
        assert result == [1, 3]

    def test_with_prob(self):
        probs = [0.1, 0.6, 0.3, 0.8]
        result = get_bool_ids_greater_than(probs, limit=0.5, return_prob=True)
        assert result == [(1, 0.6), (3, 0.8)]

    def test_empty(self):
        probs = [0.1, 0.2, 0.3]
        result = get_bool_ids_greater_than(probs, limit=0.5)
        assert result == []


class TestGetSpan:
    def test_single_span(self):
        start_ids = [(2, 0.9)]
        end_ids = [(5, 0.8)]
        spans = get_span(start_ids, end_ids, with_prob=True)
        assert ((2, 0.9), (5, 0.8)) in spans

    def test_multiple_spans(self):
        start_ids = [(1, 0.9), (10, 0.8)]
        end_ids = [(3, 0.7), (12, 0.6)]
        spans = get_span(start_ids, end_ids, with_prob=True)
        assert len(spans) == 2


class TestDbc2sbc:
    def test_full_width_to_half_width(self):
        assert dbc2sbc("ＡＢＣ１２３") == "ABC123"

    def test_chinese_unchanged(self):
        assert dbc2sbc("中文文本") == "中文文本"

    def test_space_conversion(self):
        assert dbc2sbc("　") == " "


class TestAutoSplitter:
    def test_short_text(self):
        texts = ["short text"]
        short_texts, mapping = auto_splitter(texts, max_predict_len=100)
        assert len(short_texts) == 1
        assert mapping == {0: [0]}

    def test_long_text_hard_split(self):
        text = "a" * 250
        short_texts, mapping = auto_splitter([text], max_predict_len=100)
        assert len(short_texts) == 3  # 100 + 100 + 50
        assert len(mapping[0]) == 3

    def test_sentence_split(self):
        text = "第一句话。第二句话。第三句话。第四句话。" * 20
        short_texts, mapping = auto_splitter([text], max_predict_len=100, split_sentence=True)
        assert len(short_texts) >= 1
        assert mapping[0] is not None


# ---------------------------------------------------------------------------
# End-to-end extraction tests (require real weights)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_WEIGHTS, reason="uie-base weights not converted yet")
class TestExtractionE2E:
    """Full pipeline tests that require the converted uie-base model."""

    @pytest.fixture(scope="class")
    def extractor(self):
        from uie_pytorch import UIEExtractor
        return UIEExtractor(model=WEIGHTS_DIR, device="cpu")

    def test_entity_extraction(self, extractor):
        extractor.set_schema(["时间", "选手", "赛事名称"])
        text = "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"
        results = extractor(text)

        assert len(results) == 1
        result = results[0]
        # Should find at least one entity
        total = sum(len(v) for v in result.values())
        assert total > 0, "No entities extracted"

        # Verify extracted text values
        all_texts = []
        for field, items in result.items():
            for item in items:
                all_texts.append(item["text"])
                assert "probability" in item
                assert item["probability"] > 0.5

        # Check that at least some expected entities are found
        expected_any = ["谷爱凌", "2月8日上午", "北京冬奥会"]
        found_any = any(exp in t for t in all_texts for exp in expected_any)
        assert found_any, f"Expected entities not found. Got: {all_texts}"

    def test_relation_extraction(self, extractor):
        extractor.set_schema([{"歌曲名称": ["歌手"]}])
        text = "《告别了》是孙耀威在专辑爱的故事里面的歌曲"
        results = extractor(text)

        assert len(results) == 1
        result = results[0]
        assert "歌曲名称" in result
        assert len(result["歌曲名称"]) > 0

        # Check nested relations exist
        for song in result["歌曲名称"]:
            assert "text" in song

    def test_batch_extraction(self, extractor):
        extractor.set_schema(["时间"])
        texts = [
            "2024年1月1日新年快乐",
            "昨天下午三点开会",
            "明天上午十点出发",
        ]
        results = extractor(texts)
        assert len(results) == 3

    def test_schema_change(self, extractor):
        """Verify that set_schema can be called multiple times."""
        extractor.set_schema(["时间"])
        r1 = extractor("2024年1月1日")

        extractor.set_schema(["地点"])
        r2 = extractor("北京天安门广场")

        assert "时间" in r1[0]
        assert "地点" in r2[0]
