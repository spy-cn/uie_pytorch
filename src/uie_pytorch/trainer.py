"""UIE 模型微调的核心训练循环与损失函数实现。

UIE 模型的 Loss 计算原理：
    UIE 的 Loss 是【多标签二分类交叉熵损失（Binary Cross Entropy, BCE）】。
    它将实体抽取建模为两个独立的二分类任务：
    1. 预测当前 Token 是否为实体的【起点（Start Token）】
    2. 预测当前 Token 是否为实体的【终点（End Token）】

    对于输入序列中的每一个 Token 位置，起点和终点都会独立输出一个 $[0, 1]$ 之间的概率值。
    如果是标注的区间边界，则 Label 为 1，其余所有背景 Token 的 Label 均为 0。

使用示例::

    from uie_pytorch.trainer import train_uie

    train_uie(
        model_path="weights/uie-base",
        train_data="data/train.jsonl",
        dev_data="data/dev.jsonl",
        output_dir="weights/uie-finetuned",
        epochs=3,
        batch_size=16,
        lr=2e-5,
    )
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast, get_linear_schedule_with_warmup

from uie_pytorch.model import UIE
from uie_pytorch.dataset import UIEDataset


def _resolve_device(device: str) -> torch.device:
    """自动检测并选择最佳的模型运行硬件设备。

    支持自动检测：NVIDIA GPU (CUDA) -> Apple Silicon GPU (MPS) -> CPU。
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 兼容 M1/M2/M3 等 Mac 芯片的硬件加速 (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate_fn(batch):
    """整理函数（Collate Function）：将 Dataset 产生的 UIEExample 列表打包成 Batch 级的 Tensor 字典。

    此步骤会将多条独立样本拼接（Stack）为形状如 (Batch_Size, Sequence_Length) 的二维张量。
    """
    return {
        "input_ids": torch.stack([b.input_ids for b in batch]),
        "token_type_ids": torch.stack([b.token_type_ids for b in batch]),
        "attention_mask": torch.stack([b.attention_mask for b in batch]),
        "start_ids": torch.stack([b.start_ids for b in batch]),
        "end_ids": torch.stack([b.end_ids for b in batch]),
    }


def uie_loss(
        start_prob: torch.Tensor,
        end_prob: torch.Tensor,
        start_ids: torch.Tensor,
        end_ids: torch.Tensor,
) -> torch.Tensor:
    """计算所有 Token 位置上的平均二分类交叉熵损失 (BCE Loss)。

    参数：
        start_prob / end_prob: 模型前向传播输出的概率矩阵（经过 Sigmoid 激活后），形状为 ``(Batch_Size, Seq_Len)``。
        start_ids / end_ids:   真实的 0/1 边界标注，形状为 ``(Batch_Size, Seq_Len)``。

    避坑指南：
        虽然 Sigmoid 的理论输出在 (0, 1) 之间，但在半精度（Float16）或单精度浮点数的边缘计算时，
        极小或极大的值可能会被截断成精确的 0.0 或 1.0。这会导致标准 BCE 内部的 log(0) 计算产生 NaN。
        因此，在这里使用 ``clamp`` 将预测概率强行约束在 $[10^{-7}, 1 - 10^{-7}]$ 之间。
    """
    bce = nn.BCELoss(reduction="mean")
    eps = 1e-7

    # 限制概率边界，防止产生梯度 NaN
    start_loss = bce(start_prob.clamp(eps, 1 - eps), start_ids)
    end_loss = bce(end_prob.clamp(eps, 1 - eps), end_ids)

    # 起点损失和终点损失等权相加，求平均作为最终 Loss
    return (start_loss + end_loss) / 2.0


def train_uie(
        model_path: str,
        train_data: str,
        output_dir: str,
        dev_data: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_seq_len: int = 512,
        negative_ratio: float = 0.2,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_best: bool = True,
        log_interval: int = 50,
        device: str = "auto",
        seed: int = 42,
):
    """UIE 模型微调的主控制训练函数。

    参数说明：
        model_path: 存放预训练模型权重的目录（需包含 ``pytorch_model.bin`` 与 ``config.json``）。
        train_data: 训练集 JSONL 文件的文件路径。
        output_dir: 训练完毕或评估表现最好时的模型保存目录。
        dev_data: 验证集 JSONL 文件路径（可选，若传入，则在每轮 Epoch 结束后进行评估，并根据 Loss 保存最优模型）。
        epochs: 迭代总轮数。
        batch_size: 单卡/单设备上的 Batch 大小。
        lr: AdamW 优化器的最大学习率。
        weight_decay: 权重衰减系数（L2 正则化），避免模型权重数值过大以防过拟合。
        warmup_ratio: 预热步数占总步数的比例，在此阶段学习率从 0 线性爬升到 lr 的设定值。
        max_seq_len: 模型能处理的最大序列长度，超长部分会被截断。
        negative_ratio: 负样本采样比例。
        grad_accum_steps: 梯度累积步数（等效 batch = batch_size * grad_accum_steps）。
        max_grad_norm: 用于梯度裁剪的阈值，防止在优化过程中出现极大的梯度导致模型崩溃。
        save_best: 如果为 True 且提供了验证集，会始终保存验证集 Loss 最低的那一轮模型。
        log_interval: 训练过程中，每隔多少个全局 Step 打印一次平均 Loss 和学习率。
        device: 指定运行设备 ("auto" / "cpu" / "cuda" / "mps")。
        seed: 随机种子，固定随机数以确保结果可复现。
    """
    # 设定 PyTorch 随机种子，确保模型初始状态、数据洗牌和负样本采样行为一致
    torch.manual_seed(seed)
    dev = _resolve_device(device)

    # --- 1. 初始化 Tokenizer、Config 与 Model -----------------------------------
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    config = BertConfig.from_pretrained(model_path)
    model = UIE(config)

    pt_file = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(pt_file):
        try:
            # 优先尝试 PyTorch 2.0+ 推荐的安全加载模式（防止反序列化漏洞）
            state = torch.load(pt_file, map_location="cpu", weights_only=True)
        except TypeError:  # 兼容 PyTorch 1.x 老版本
            state = torch.load(pt_file, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[train] 已成功加载预训练 UIE 权重： {pt_file}")
    else:
        print(f"[train] 警告: 未找到预训练权重文件 {pt_file} —— 将从零初始化开始训练。")

    model.to(dev)

    # --- 2. 构建 Dataset 与 DataLoader -------------------------------------------
    train_ds = UIEDataset(
        train_data, tokenizer,
        max_seq_len=max_seq_len, negative_ratio=negative_ratio,
    )
    print(f"[train] 成功构建训练集。正负样本总实例数: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_fn, drop_last=False,
    )

    dev_loader = None
    if dev_data and os.path.exists(dev_data):
        # 验证集上通常无需设置负样本采样比例（negative_ratio=0.0），保持标准分布进行真实评估
        dev_ds = UIEDataset(dev_data, tokenizer, max_seq_len=max_seq_len, negative_ratio=0.0)
        dev_loader = DataLoader(
            dev_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn,
        )
        print(f"[train] 成功构建验证集。验证实例数: {len(dev_ds)}")

    # --- 3. 优化器 (Optimizer) 与 学习率调度器 (Scheduler) ------------------------
    # 计算总全局更新步数（Global Steps）
    total_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    # 权重衰减过滤策略：偏置 bias、LayerNorm 权重和各种 Norm 层的权重在数学上不需要进行 L2 惩罚
    no_decay = {"bias", "LayerNorm.weight", "norm1.weight", "norm2.weight"}
    named_params = list(model.named_parameters())
    optimizer_grouped = [
        {
            "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    os.makedirs(output_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0  # 记录完成了多少次梯度更新 (即实际调用 optimizer.step 的次数)

    # --- 4. 核心训练循环 (Training Loop) ------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()  # 确保在每个 Epoch 开始前清空累计梯度

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(dev)
            token_type_ids = batch["token_type_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            start_ids = batch["start_ids"].to(dev)
            end_ids = batch["end_ids"].to(dev)

            # 前向传播：模型会直接输出起点和终点的 sigmoid 概率概率图
            start_prob, end_prob = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            loss = uie_loss(start_prob, end_prob, start_ids, end_ids)

            # 梯度累积：如果是累积多步更新，当前单步 Loss 需要除以累积步数以平摊梯度
            loss = loss / grad_accum_steps
            loss.backward()

            # 达到指定的累积步数后，再进行真实的参数更新
            if step % grad_accum_steps == 0:
                # 梯度截断，防止由于硬长输入产生的超大梯度使模型不稳定
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # 还原真实的 Loss 大小以便进行直观的日志打印
            running_loss += loss.item() * grad_accum_steps

            # 打印训练日志
            if global_step > 0 and global_step % log_interval == 0 and step % grad_accum_steps == 0:
                avg = running_loss / (step)
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch} | step {global_step}/{total_steps} | "
                    f"loss {avg:.4f} | lr {cur_lr:.2e}"
                )

        # 整个 Epoch 训练集上的平均 Loss
        epoch_loss = running_loss / len(train_loader)
        msg = f"[train] Epoch {epoch}/{epochs} — 训练 Loss {epoch_loss:.4f}"

        # --- 5. 验证集评估 (Validation Phase) --------------------------------------
        if dev_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():  # 验证阶段禁用梯度计算，节省显存与算力
                for batch in dev_loader:
                    input_ids = batch["input_ids"].to(dev)
                    token_type_ids = batch["token_type_ids"].to(dev)
                    attention_mask = batch["attention_mask"].to(dev)
                    start_ids = batch["start_ids"].to(dev)
                    end_ids = batch["end_ids"].to(dev)

                    sp, ep = model(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                    )
                    val_loss += uie_loss(sp, ep, start_ids, end_ids).item()

            val_loss /= len(dev_loader)
            msg += f" | 验证 Loss {val_loss:.4f}"
            model.train()  # 恢复为训练模式（让 Dropout、LayerNorm 生效）

            # 如果验证集表现提升，则保存当前的 checkpoint
            if save_best and val_loss < best_loss:
                best_loss = val_loss
                _save_checkpoint(model, tokenizer, config, output_dir)
                msg += "  ← [已保存最佳模型]"
        else:
            # 如果未提供验证集，则默认在每个 Epoch 结束时保存当前的最新状态
            _save_checkpoint(model, tokenizer, config, output_dir)

        print(msg)

    # 兜底保存逻辑：如果未开启 save_best 或没有设置验证集，确保最后一轮的模型成功落地
    if not save_best or dev_loader is None:
        _save_checkpoint(model, tokenizer, config, output_dir)

    print(f"[train] 训练完成！最终微调权重与配置均已保存至：{output_dir}")
    return output_dir


def _save_checkpoint(model, tokenizer, config, output_dir: str):
    """保存训练权重、分词器和配置文件，以便 UIEExtractor 或者是 Transformers 可以无缝加载。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存模型参数状态字典 (State Dict)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # 2. 保存分词器词表、特殊 Token 映射以及配置文件
    tokenizer.save_pretrained(output_dir)

    # 3. 保存 BertConfig。其内部的 save_pretrained 方法会自动将其格式化为 config.json 文件
    config.save_pretrained(output_dir)
