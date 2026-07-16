"""英文实体抽取 (NER) 示例脚本。

运行方法:
    python examples/entity_extraction_en.py

前置条件：
    在运行前，脚本会自动检查并转换英文基础模型权重。您也可以手动在终端执行：
    python -m uie_pytorch.converter --model uie-base-en --output_dir weights/uie-base-en

说明：
    本脚本演示了如何加载专为英文优化的 `uie-base-en` 权重进行零样本实体抽取。
    英文 UIE 模型在处理多语言或纯英文语料时，具备比中文模型更好的英文子词（Subword）分词适配度以及语法泛化能力。
"""

from __future__ import annotations

import json
import os
import sys

# 💡 技巧：将项目根目录下的 src 目录添加到 Python 顺址路径中。
# 确保在未运行 pip install 时，脚本也能直接识别并导入 uie_pytorch 模块。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uie_pytorch import UIEExtractor, convert_model


def main():
    # ------------------------------------------------------------------
    # 步骤 1: 自动检查并转换英文 UIE 权重格式
    # ------------------------------------------------------------------
    # 定义 PyTorch 版 uie-base-en 权重的本地存放路径
    model_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "uie-base-en")
    model_dir = os.path.abspath(model_dir)

    # 检查本地是否已经存在转换好的 PyTorch 英文权重文件 (pytorch_model.bin)
    if not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        print("正在将 uie-base-en 权重转换为 PyTorch 格式（此操作仅在首次运行时执行）...")
        convert_model("uie-base-en", model_dir)
        print("权重转换完成！\n")

    # ------------------------------------------------------------------
    # 步骤 2: 初始化英文 UIE 提取器 Pipeline
    # ------------------------------------------------------------------
    # 架构微调说明：
    # 当 UIEExtractor 识别到权重目录中包含 "en" 或 "base-en" 时，
    # 内部的 `_is_en` 标志会自动设为 True。这会使多阶段关系抽取时拼接的 Prompt 自动调整为：
    # "relation_name of entity_name" （如： "founder of Apple"）而非中文的 "张三的创始人"。
    ie = UIEExtractor(
        model=model_dir,
        schema=["person", "organization", "location"],
        device="auto"  # 自动选择 CUDA、MPS (Mac 芯片加速) 或 CPU
    )

    # ------------------------------------------------------------------
    # 步骤 3: 运行英文实体抽取测试
    # ------------------------------------------------------------------
    texts = [
        "Steve Jobs was the CEO of Apple Inc. in Cupertino, California.",
        "Barack Obama was born in Hawaii and served as president of the United States.",
        "The United Nations headquarters is located in New York City.",
    ]

    print("=" * 70)
    print("英文命名实体识别 (English Entity Extraction)")
    print("=" * 70)

    for text in texts:
        print(f"\n待抽取文本 (Text): {text}")

        # 执行前向推理与解码后处理
        result = ie(text)

        # 以格式化 JSON 格式输出匹配到的实体边界、置信度和文本切片
        print(f"抽取结果 (Result):\n{json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
