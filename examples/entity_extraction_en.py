"""Example: English entity extraction with uie-base-en.

Requires converting the English model first:
    python -m uie_pytorch.converter --model uie-base-en --output_dir weights/uie-base-en

Usage:
    python examples/entity_extraction_en.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uie_pytorch import UIEExtractor, convert_model


def main():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "uie-base-en")
    model_dir = os.path.abspath(model_dir)

    if not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        print("Converting uie-base-en weights to PyTorch format...")
        convert_model("uie-base-en", model_dir)
        print("Done!\n")

    ie = UIEExtractor(model=model_dir, schema=["person", "organization", "location"])

    texts = [
        "Steve Jobs was the CEO of Apple Inc. in Cupertino, California.",
        "Barack Obama was born in Hawaii and served as president of the United States.",
        "The United Nations headquarters is located in New York City.",
    ]

    print("=" * 70)
    print("English Entity Extraction")
    print("=" * 70)

    for text in texts:
        print(f"\nText: {text}")
        result = ie(text)
        print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
