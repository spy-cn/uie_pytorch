"""将 PaddleNLP UIE 权重 (``model_state.pdparams``) 转换为 PyTorch 格式。

PaddleNLP 的 UIE 将模型参数保存在 ``.pdparams`` 文件中（本质上是序列化的 numpy 数组字典）。
由于 ``uie_pytorch`` 在结构上是 PaddleNLP UIE 的忠实克隆，
因此转换过程主要是【键名映射（Key Remapping）】和【Linear 层权重转置】。

💡 核心技术细节：
1. 键名映射：PyTorch 版 UIE 保持了与 Paddle 相同的层命名（如 ernie.encoder.layers...），因此键名几乎是一对一直接复制。
2. 维度转置：Paddle 的 ``nn.Linear`` 权重维度是 (in_features, out_features)，而 PyTorch 的
   ``nn.Linear`` 权重维度是 (out_features, in_features)。因此，所有 Linear 层的 weight 张量都需要进行转置（.t()）。
   而 Embedding 层在两个框架中都是 (vocab, hidden)，不需要转置。

使用方法：
    python -m uie_pytorch.converter --model uie-base --output_dir ./weights/uie-base

转换完成后，``output_dir`` 目录下会生成：
  - ``pytorch_model.bin``: PyTorch 格式权重文件
  - ``config.json``: 适配 Hugging Face BERT 格式的模型配置文件
  - ``vocab.txt``: 词表文件
  - 相关的分词器（Tokenizer）元数据文件
之后，您便可以直接使用 ``uie_pytorch.extractor.UIEExtractor`` 来加载它们。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# PaddleNLP 官方 UIE 预训练模型及配置文件的下载链接映射表
# ---------------------------------------------------------------------------
MODEL_URLS: Dict[str, Dict[str, str]] = {
    "uie-base": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-medium": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-mini": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-micro": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-nano": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano_v1.1/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
    "uie-base-en": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en_v1.2/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/vocab.txt",
    },
    "uie-medical-base": {
        "model_state": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medical_base_v0.2/model_state.pdparams",
        "config": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/config.json",
        "vocab_file": "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
    },
}


def _download(url: str, dest: str):
    """从指定的 *url* 下载文件并保存至 *dest* 路径下。"""
    import urllib.request

    print(f"正在下载 {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> 已成功保存至 {dest} (文件大小: {os.path.getsize(dest) / 1024:.0f} KB)")


def _load_pdparams(path: str) -> "OrderedDict":
    """加载 PaddlePaddle 的 ``.pdparams`` 权重文件。

    Paddle 的权重本质上是用 pickle 序列化的【参数名 -> Numpy 数组】字典。
    """
    with open(path, "rb") as f:
        state = pickle.load(f)

    clean = OrderedDict()
    for key, value in state.items():
        # 过滤掉 PaddlePaddle 框架内部自动生成的元数据 Key (例如结构体命名映射)
        if key.startswith("StructuredToParameterName"):
            continue

            # 将 Paddle Tensor 转换为标准的 Numpy Array
        if hasattr(value, "numpy"):
            value = value.numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array(value)

        clean[key] = value
    return clean


# ---------------------------------------------------------------------------
# 权重 Key 映射逻辑
# ---------------------------------------------------------------------------
def _convert_key(key: str) -> str:
    """将 PaddleNLP UIE 的参数 Key 映射为 PyTorch UIE 对应的 Key。

    由于 PyTorch 版的 UIE (``uie_pytorch``) 保持了与 Paddle 相同的底层结构和层级命名
    (例如 ``ernie.encoder.layers...``, ``linear_start``, ``linear_end``)，
    因此这里无需对 Key 进行复杂的正则替换，保持原样（恒等映射）即可。
    """
    return key


def convert_state_dict(pdparams: dict) -> "OrderedDict":
    """核心权重转换逻辑：将 PaddleNLP 的权重字典转换为 PyTorch 格式。

    重要硬件/框架差异说明：
    1. PaddlePaddle 的 ``nn.Linear`` 内部权重张量形状为 (in_features, out_features)。
    2. PyTorch 的 ``nn.Linear`` 内部权重张量形状为 (out_features, in_features)。

    为了保证数学计算结果完全一致，我们必须对所有属于 nn.Linear 层的二维权重矩阵进行「转置（Transpose）」。
    而像 Embedding 层的权重（形状为 vocab x hidden），在两个框架中存储布局一致，则坚决不能进行转置。
    """
    # 定义所有需要被转置的 Linear 层权重的后缀名
    LINEAR_SUFFIXES = (
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight",
        "linear1.weight", "linear2.weight",
        "linear_start.weight", "linear_end.weight",
        "pooler.dense.weight",
    )

    new_state = OrderedDict()
    skipped = []
    transposed = 0

    for old_key, array in pdparams.items():
        # 跳过 Paddle 的元数据
        if old_key.startswith("StructuredToParameterName"):
            skipped.append(old_key)
            continue

        new_key = _convert_key(old_key)
        # 将 numpy 数组深度拷贝，并包装为 PyTorch Tensor
        tensor = torch.from_numpy(array.copy())

        # 匹配当前的 tensor 是否为 Linear 层的权重（必须是 2D 且属于上述 Linear 后缀列表）
        if any(new_key.endswith(suf) for suf in LINEAR_SUFFIXES) and tensor.dim() == 2:
            # 矩阵转置：Paddle (in, out) -> PyTorch (out, in)，并确保内存连续（contiguous）
            tensor = tensor.t().contiguous()
            transposed += 1

        new_state[new_key] = tensor

    print(f"  (成功转置了 {transposed} 个 Linear 层的权重矩阵)")
    if skipped:
        print(f"  (过滤跳过了 {len(skipped)} 个 Paddle 内部元数据 Key)")
    return new_state


def paddle_config_to_bert(config: dict) -> dict:
    """将 Paddle 的 Ernie 配置文件字段，映射兼容至 HuggingFace BertConfig 规范。

    特别保留并适配了 ERNIE 专属的 ``task_type_vocab_size`` 等多任务 Token 嵌入配置，
    这是 UIE 能够正确初始化对应 embedding 层的基础。
    """
    bert_config = {
        "architectures": ["UIE"],  # 声明模型架构
        "model_type": "bert",  # 映射为 BERT 主干
        "hidden_size": config.get("hidden_size", 768),  # 隐层维度
        "num_hidden_layers": config.get("num_hidden_layers", 12),  # Transformer 层数
        "num_attention_heads": config.get("num_attention_heads", 12),  # 多头注意力头数
        "intermediate_size": config.get("intermediate_size", 3072),  # FFN 隐层维度
        "vocab_size": config.get("vocab_size", 40000),  # 词表大小
        "max_position_embeddings": config.get("max_position_embeddings", 2048),  # 最大位置编码长度
        "hidden_act": config.get("hidden_act", "gelu"),  # 激活函数
        "hidden_dropout_prob": config.get("hidden_dropout_prob", 0.1),  # 隐藏层 Dropout 概率
        "attention_probs_dropout_prob": config.get("attention_probs_dropout_prob", 0.1),  # 注意力机制 Dropout 概率
        "type_vocab_size": config.get("type_vocab_size", 4),  # Segment ID 嵌入空间大小
        "task_type_vocab_size": config.get("task_type_vocab_size", 3),  # ERNIE 专属：Task ID 嵌入空间大小
        "layer_norm_eps": config.get("layer_norm_eps", 1e-12),  # 层归一化微小值
        "pad_token_id": config.get("pad_token_id", 0),  # Pad 占位符 ID
        "initializer_range": 0.02,  # 权重初始化范围
    }
    return bert_config


def convert_model(model_name: str, output_dir: str):
    """自动完成模型下载、格式转换，并输出为标准的 PyTorch 加载格式。"""
    os.makedirs(output_dir, exist_ok=True)

    urls = MODEL_URLS[model_name]
    pdparams_path = os.path.join(output_dir, "model_state.pdparams")
    config_path = os.path.join(output_dir, "source_config.json")
    vocab_path = os.path.join(output_dir, "vocab.txt")

    # 1. 如果本地缓存不存在，则从 Paddle 官方拉取相应资源
    if not os.path.exists(pdparams_path):
        _download(urls["model_state"], pdparams_path)
    if not os.path.exists(config_path):
        _download(urls["config"], config_path)
    if not os.path.exists(vocab_path):
        _download(urls["vocab_file"], vocab_path)

    # 2. 读取并开始进行参数矩阵转置与转换
    pd_state = _load_pdparams(pdparams_path)
    pt_state = convert_state_dict(pd_state)

    # 3. 将 Paddle 的 Ernie 配置转换映射为 Hugging Face BERT 配置
    with open(config_path) as f:
        paddle_cfg = json.load(f)
    bert_cfg = paddle_config_to_bert(paddle_cfg)

    # 4. 持久化存储 PyTorch 权重与配置文件
    torch.save(pt_state, os.path.join(output_dir, "pytorch_model.bin"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(bert_cfg, f, indent=2)

    # 5. 生成 PyTorch UIETokenizer 运行所需的元数据，以兼容 HuggingFace Tokenizer 加载机制
    special = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special, f, indent=2)
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"do_lower_case": True, "model_max_length": 512}, f, indent=2)

    print(f"\n✅ 成功将 {model_name} 权重转换为 PyTorch 格式，并输出到 {output_dir}")
    print(f"   - pytorch_model.bin  (共包含 {len(pt_state)} 个参数张量)")
    print(f"   - config.json")
    print(f"   - vocab.txt")


def main():
    parser = argparse.ArgumentParser(description="将 PaddleNLP UIE 权重无损转换为 PyTorch 格式的工具")
    parser.add_argument(
        "--model",
        default="uie-base",
        choices=list(MODEL_URLS.keys()),
        help="待转换的模型名称（默认为 uie-base）"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="转换后文件的保存输出路径"
    )
    args = parser.parse_args()
    convert_model(args.model, args.output_dir)


if __name__ == "__main__":
    main()
