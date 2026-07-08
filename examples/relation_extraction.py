"""Example: Chinese relation extraction with nested schema.

Usage:
    python examples/relation_extraction.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uie_pytorch import UIEExtractor


def main():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "uie-base")
    model_dir = os.path.abspath(model_dir)

    ie = UIEExtractor(model=model_dir, device="auto")

    # Example 1: Song → singer + album
    print("=" * 70)
    print("关系抽取示例 1: 歌曲 → 歌手 + 专辑")
    print("=" * 70)
    ie.set_schema([{"歌曲名称": ["歌手", "所属专辑"]}])
    text = "《告别了》是孙耀威在专辑爱的故事里面的歌曲"
    print(f"\n文本: {text}")
    result = ie(text)
    print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # Example 2: Person → birthplace + occupation
    print("\n" + "=" * 70)
    print("关系抽取示例 2: 人物 → 出生地 + 职业")
    print("=" * 70)
    ie.set_schema([{"人物": ["出生地", "职业"]}])
    text = "姚明1980年出生于上海，前中国职业篮球运动员，司职中锋"
    print(f"\n文本: {text}")
    result = ie(text)
    print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # Example 3: Company → founder + headquarter
    print("\n" + "=" * 70)
    print("关系抽取示例 3: 公司 → 创始人 + 总部")
    print("=" * 70)
    ie.set_schema([{"公司": ["创始人", "总部地点"]}])
    text = "华为由任正非于1987年在深圳创立，总部位于广东省深圳市龙岗区"
    print(f"\n文本: {text}")
    result = ie(text)
    print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
