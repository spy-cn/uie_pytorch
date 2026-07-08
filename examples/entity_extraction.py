"""Example: Chinese entity extraction with UIE-PyTorch.

Usage:
    python examples/entity_extraction.py
"""

from __future__ import annotations

import json
import os
import sys

# Add src to path for direct script execution (without pip install)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uie_pytorch import UIEExtractor, convert_model


def main():
    # Step 1: Ensure weights are converted
    model_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "uie-base")
    model_dir = os.path.abspath(model_dir)
    if not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        print("Converting uie-base weights to PyTorch format...")
        convert_model("uie-base", model_dir)
        print("Done!\n")

    # Step 2: Initialize extractor
    ie = UIEExtractor(
        model=model_dir,
        schema=["时间", "选手", "赛事名称", "奖项"],
        device="auto",
    )
    print(f"Model loaded from: {model_dir}\n")

    # Step 3: Entity extraction
    texts = [
        "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！",
        "2024年巴黎奥运会乒乓球男单决赛，樊振东战胜特鲁尔斯·莫雷加德夺得金牌。",
    ]

    print("=" * 70)
    print("中文实体抽取")
    print("=" * 70)

    for text in texts:
        print(f"\n文本: {text}")
        result = ie(text)
        print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # Step 4: Custom schema
    print("\n" + "=" * 70)
    print("自定义 Schema")
    print("=" * 70)

    ie.set_schema(["城市", "天气", "温度"])
    text = "北京今天晴，最高温度35度；上海多云，28度；广州大雨，30度"
    print(f"\n文本: {text}")
    result = ie(text)
    print(f"结果: {json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
