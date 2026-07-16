"""示例脚本：微调 UIE 模型并使用微调后的权重进行推理测试。

运行方法:
    python examples/finetune_example.py

本脚本展示了信息抽取（IE）的完整工程闭环流：
    1. 前置条件：确保您已提前将 PaddleNLP 权重转换为了 PyTorch 格式（参见 converter 转换工具）。
    2. 使用 ``data/train.jsonl`` 训练集对 UIE 模型进行快速微调。
    3. 加载微调后生成的 Checkpoint，实例化抽取器，在测试文本上进行实时推理并打印输出。
"""

from __future__ import annotations

import json
import os
import sys

# 💡 技巧：将项目根目录下的 src 添加到 Python 寻址路径中，
# 确保在未将项目执行 pip install -e . 安装时，依然能正确引入 uie_pytorch 模块。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uie_pytorch import UIEExtractor, train_uie


def main():
    # ---- 0. 定义相关的输入输出绝对路径 ---------------------------------------
    # 基础模型路径（转换后生成的 PyTorch 基础 UIE 权重目录）
    base_model = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "weights", "uie-base")
    )
    # 训练集路径
    train_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "train.jsonl")
    )
    # 验证集路径
    dev_data = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "dev.jsonl")
    )
    # 训练完成后，微调模型的保存路径
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "weights", "uie-finetuned")
    )

    # ---- 步骤 1: 启动模型微调 ------------------------------------------------
    print("=" * 70)
    print(" 步骤 1: 正在对 UIE 模型进行微调训练...")
    print("=" * 70)

    train_uie(
        model_path=base_model,      # 基础权重路径
        train_data=train_data,      # 训练集文件路径
        dev_data=dev_data,          # 验证集文件路径
        output_dir=output_dir,      # 输出保存目录
        epochs=3,                   # 训练轮数（对于演示/快速测试，3轮即可）
        batch_size=4,               # 批次大小（单卡跑 demo 设为 4，显存富裕时可调整为 16 或 32）
        lr=2e-5,                    # 学习率
        max_seq_len=128,            # 最大序列截断长度（因为演示数据通常较短，设为 128 可大幅节省训练时间）
        negative_ratio=0.2,         # 负样本采样比例（混入 20% 的无答案 Schema 样本，以增强模型对“幻觉”的抵抗能力）
        device="auto",              # 自动检测可用硬件（有显卡用 CUDA，Mac 用 MPS，无卡退回 CPU）
        log_interval=5,             # 每隔 5 个 step 打印一次 Loss，方便实时观察 Loss 下降曲线
    )

    # ---- 步骤 2: 使用微调后的模型进行推理（Inference） ----------------------
    print("\n" + "=" * 70)
    print(" 步骤 2: 使用微调后的模型执行实时信息抽取推理")
    print("=" * 70)

    # 💡 核心：实例化 UIE 抽取器。
    # 您可以通过直接修改 schema 数组，来灵活指定您想要从文本中榨取的信息类别。
    # 这里我们定义了一个并列的实体抽取 Schema。
    ie = UIEExtractor(
        model=output_dir,                # 加载刚才微调好并落地的模型权重
        schema=["人物", "出生年份", "出生地"], # 定义业务抽取 Schema
        device="auto",                   # 自动选择最快硬件
    )

    # 待测试的原始测试文本
    text = "钱七于1992年在杭州出生，后考入清华大学。"
    print(f"\n输入待测文本: {text}")

    # 执行推理抽取
    result = ie(text)

    # 格式化输出 JSON 抽取结果
    print(f"抽取结果: {json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()