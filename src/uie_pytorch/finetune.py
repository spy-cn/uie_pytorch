"""UIE 模型微调的命令行启动入口脚本。

UIE (Universal Information Extraction) 微调特点：
    由于 UIE 采用 Pointer Network (指针网络) 来预测实体/关系的起止位置，
    在微调时，合理的学习率、负样本比例以及梯度累积对模型最终的收敛效果至关重要。

运行示例::

    # 1. 基础微调命令行 (标准多卡/单卡 GPU 训练)
    python -m uie_pytorch.finetune \
        --model weights/uie-base \
        --train_data data/train.jsonl \
        --dev_data data/dev.jsonl \
        --output_dir weights/uie-finetuned \
        --epochs 3 --batch_size 16 --lr 2e-5

    # 2. 快速 CPU 冒烟测试 (用于验证数据格式和代码管道是否畅通)
    python -m uie_pytorch.finetune \
        --model weights/uie-base \
        --train_data data/train.jsonl \
        --output_dir weights/uie-finetuned \
        --epochs 1 --batch_size 4 --device cpu
"""

from __future__ import annotations

import argparse

# 导入核心训练执行器，负责具体的训练循环、Loss 计算及 Checkpoint 保存
from .trainer import train_uie


def parse_args():
    """解析命令行输入参数。"""
    p = argparse.ArgumentParser(description="微调 UIE (Universal Information Extraction) 模型")

    # ------------------ 模型与数据路径相关参数 ------------------
    p.add_argument("--model", type=str, default="weights/uie-base",
                   help="预训练 UIE 模型或转换后的 PyTorch UIE 权重目录。")
    p.add_argument("--train_data", type=str, required=True,
                   help="训练集 JSON-Lines 格式文件路径。")
    p.add_argument("--dev_data", type=str, default=None,
                   help="验证集 JSON-Lines 格式文件路径（可选，若不提供则不进行 Epoch 验证评估）。")
    p.add_argument("--output_dir", type=str, default="weights/uie-finetuned",
                   help="微调后模型的保存输出目录。")

    # ------------------ 核心训练超参数 ------------------
    p.add_argument("--epochs", type=int, default=3,
                   help="总训练轮数（Epoch）。对于 UIE 的少样本微调，通常 3~10 轮即可充分收敛。")
    p.add_argument("--batch_size", type=int, default=16,
                   help="单步训练批次大小（Batch Size）。根据显存大小调整。")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="基础学习率。推荐范围：1e-5 到 5e-5。学习率过大容易导致指针网络收敛崩溃。")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="权重衰减（L2 正则化），用于防止模型过拟合。")
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                   help="线性学习率预热（Warmup）步数占总步数的比例，有助于训练初期模型的平稳收敛。")
    p.add_argument("--max_seq_len", type=int, default=512,
                   help="输入序列的最大截断长度（包括 Prompt 与 Text 的总和），最大支持 512。")

    # ------------------ UIE 鲁棒性关键控制参数 ------------------
    p.add_argument("--negative_ratio", type=float, default=0.2,
                   help="负样本（无答案 Schema）的采样比例。例如 0.2 表示加入 20% 比例的负样本。"
                        "强烈建议：在实际抽取业务中，如果模型在推理时频繁出现『幻觉/瞎编实体』，"
                        "可以尝试将此参数调高至 0.3 到 0.5。")

    # ------------------ 显存优化与训练加速相关参数 ------------------
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="梯度累积步数。在显存受限时有用，例如设置 batch_size=4 且 grad_accum_steps=4，"
                        "可以达到模拟等效 batch_size=16 的效果。")
    p.add_argument("--max_grad_norm", type=float, default=1.0,
                   help="梯度裁剪的最大范数（Gradient Clipping），用以防止梯度爆炸。")

    # ------------------ 评估与日志输出控制 ------------------
    p.add_argument("--save_best", action="store_true", default=True,
                   help="是否自动根据验证集（Dev Loss）的表现，保存效果最优的模型权重（默认：True）。")
    p.add_argument("--log_interval", type=int, default=50,
                   help="每隔多少个迭代步数（Steps）在终端打印/记录一次训练日志（Loss 等）。")

    # ------------------ 运行环境硬件与随机种子 ------------------
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="运行设备的类型：'auto'（自动检测 GPU/CPU）、'cpu'、'cuda'（Nvidia GPU）、'mps'（Mac 芯片加速）。")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子。固定该值可保证训练过程、负样本采样以及数据洗牌的可复现性。")

    return p.parse_args()


def main():
    """微调流程主函数：解析命令行输入并拉起 trainer 执行。"""
    args = parse_args()

    # 将解析出的命令行配置参数优雅地解包并传递给 train_uie 核心训练方法
    train_uie(
        model_path=args.model,
        train_data=args.train_data,
        output_dir=args.output_dir,
        dev_data=args.dev_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_seq_len=args.max_seq_len,
        negative_ratio=args.negative_ratio,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        save_best=args.save_best,
        log_interval=args.log_interval,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
