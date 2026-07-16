"""命名实体识别 (NER) / 实体抽取示例脚本。

运行方法:
    python examples/entity_extraction.py

本脚本演示了如何利用 UIE-PyTorch 进行零样本（Zero-shot）中文实体抽取：
  1. 脚本会自动检查并把官方的 PaddlePaddle 权重转成 PyTorch 格式。
  2. 使用简单的扁平 Schema 列表（如 ["时间", "选手", "赛事名称"]）对多个文本段落并行抽取多类实体。
  3. 演示了如何在同一个提取器实例中，通过 ``set_schema`` 动态切换抽取目标。
"""

from __future__ import annotations

import json
import os
import sys

# 技巧：将项目根目录下的 src 目录添加到 Python 寻址路径中。
# 这样即使没有使用 `pip install -e .` 安装这个包，也可以直接在 examples 目录下运行此脚本。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uie_pytorch import UIEExtractor, convert_model


def main():
    # ------------------------------------------------------------------
    # 步骤 1: 自动检查并转换权重格式
    # ------------------------------------------------------------------
    # 定义 PyTorch 版 uie-base 权重的本地存放路径
    model_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "uie-base")
    model_dir = os.path.abspath(model_dir)

    # 检查本地是否已经存在转换好的 PyTorch 权重文件 (pytorch_model.bin)
    # 如果没有，则调用内部的 `convert_model` 自动从 HuggingFace 下载官方 Paddle 权重并无损转换为 PyTorch 格式
    if not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        print("正在将 uie-base 权重转换为 PyTorch 格式（此操作仅在首次运行时执行）...")
        convert_model("uie-base", model_dir)
        print("权重转换完成！\n")

    # ------------------------------------------------------------------
    # 步骤 2: 初始化 UIEExtractor 抽取器
    # ------------------------------------------------------------------
    # 实例化抽取器：
    #   - schema: 传入我们需要抽取的实体类别（扁平列表表示实体抽取）。
    #   - device: "auto" 会自动检测当前的硬件环境，优先使用 GPU (CUDA) 或 Apple M 系列芯片 (MPS)。
    ie = UIEExtractor(
        model=model_dir,
        schema=["时间", "选手", "赛事名称", "奖项"],
        device="auto",
    )
    print(f"模型成功载入自: {model_dir}\n")

    # ------------------------------------------------------------------
    # 步骤 3: 批量文本实体抽取
    # ------------------------------------------------------------------
    texts = [
        "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！",
        "2024年巴黎奥运会乒乓球男单决赛，樊振东战胜特鲁尔斯·莫雷加德夺得金牌。",
    ]

    print("=" * 70)
    print("中文实体抽取示例")
    print("=" * 70)

    for text in texts:
        print(f"\n当前文本: {text}")
        # 执行抽取：模型会根据初始化时设定的 4 个 Schema 独立进行扫描
        result = ie(text)
        # 格式化打印输出 JSON 结果
        print(f"抽取结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # ------------------------------------------------------------------
    # 步骤 4: 动态切换自定义 Schema 任务
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("动态切换自定义 Schema")
    print("=" * 70)

    # 优势：UIE 具有强大的 Zero-shot（零样本）泛化能力，
    # 我们可以随时调用 `set_schema` 改变抽取目标，而不需要重新加载笨重的模型。
    ie.set_schema(["城市", "天气", "温度"])
    text = "北京今天晴，最高温度35度；上海多云，28度；广州大雨，30度"

    print(f"\n当前文本: {text}")
    result = ie(text)
    print(f"抽取结果: {json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()