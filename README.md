# UIE-PyTorch

> **PaddleNLP UIE 模型的 PyTorch 复刻版** —— 从 PaddleNLP 原始代码提炼，用纯 PyTorch + HuggingFace Transformers 实现相同的零样本/少样本信息抽取能力，并支持**自定义数据微调**。

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 简介

UIE（Universal Information Extraction）是一种统一的文本信息抽取模型，能够通过自然语言描述的 schema 来完成**实体抽取、关系抽取、事件抽取**等任务，无需训练即可使用（零样本）。

本项目是 [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) 中 UIE 的 **PyTorch 独立实现**，不依赖 PaddlePaddle 框架。模型结构和预训练权重与官方版本完全对齐，抽取结果一致。同时提供了**完整的微调工具链**，可以用少量标注数据快速适配垂直领域。

### 核心特性

- ✅ **完全复刻** PaddleNLP UIE 模型结构（ERNIE 编码器 + 双线性抽取头）
- ✅ **权重转换**：自动下载 PaddlePaddle `.pdparams` 并转换为 PyTorch `.bin`
- ✅ **零样本抽取**：实体抽取、关系抽取、事件抽取（嵌套 schema 树）
- ✅ **自定义微调**：支持实体/关系/事件标注数据，BCE 损失 + AdamW + warmup，少量数据即可适配垂直领域
- ✅ **API 兼容**：与 PaddleNLP `Taskflow("information_extraction")` 用法一致
- ✅ **纯 PyTorch**：不依赖 PaddlePaddle 框架
- ✅ **多设备支持**：CUDA（NVIDIA）、MPS（Apple Silicon）、CPU 自动检测

---

## 安装

### 从源码安装（开发模式）

```bash
git clone https://github.com/spy-cn/uie_pytorch.git
cd uie_pytorch
pip install -e .
```

### 仅安装依赖

```bash
pip install -r requirements.txt
```

### 依赖

- Python >= 3.8
- PyTorch >= 1.10
- Transformers >= 4.20
- NumPy >= 1.21

---

## 快速开始

### 1. 转换权重（首次使用）

UIE 的预训练权重由百度基于 PaddlePaddle 发布。首次使用需要下载并转换为 PyTorch 格式：

```bash
# 转换 uie-base（约 450MB，自动下载）
python -m uie_pytorch.converter --model uie-base --output_dir weights/uie-base
```

支持的模型：

| 模型 | 语言 | 说明 |
|------|------|------|
| `uie-base` | 中文 | 通用基础版（12层，768维） |
| `uie-medium` | 中文 | 中等尺寸（6层，768维） |
| `uie-mini` | 中文 | 小尺寸（6层，384维） |
| `uie-micro` | 中文 | 微型（4层，384维） |
| `uie-nano` | 中文 | 纳米型（4层，312维） |
| `uie-base-en` | 英文 | 英文基础版 |
| `uie-medical-base` | 中文 | 医疗领域专用 |

### 2. 运行抽取

```python
from uie_pytorch import UIEExtractor

# 初始化（指定本地权重路径）
ie = UIEExtractor(
    model="weights/uie-base",
    schema=["时间", "选手", "赛事名称"],
    device="auto",         # 自动检测: CUDA > MPS > CPU
)

# 实体抽取
result = ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
print(result)
```

输出：
```json
[
  {
    "时间": [{"text": "2月8日上午", "probability": 0.986, "start": 0, "end": 5}],
    "选手": [{"text": "谷爱凌", "probability": 0.898, ...}],
    "赛事名称": [{"text": "北京冬奥会自由式滑雪女子大跳台决赛", "probability": 0.850, ...}]
  }
]
```

### 3. 关系抽取（嵌套 Schema）

```python
# 嵌套 schema: 先抽取歌曲名称，再抽取其歌手和所属专辑
ie.set_schema([{"歌曲名称": ["歌手", "所属专辑"]}])
result = ie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")
print(result)
# [{"歌曲名称": [{"text": "告别了", "relations": {"歌手": [...], "所属专辑": [...]}}]}]
```

### 4. 批量抽取

```python
texts = [
    "北京今天气温30度",
    "上海明天有小雨",
    "广州未来三天多云转晴",
]
ie.set_schema(["城市", "天气"])
results = ie(texts)  # 一次处理多条
```

---

## 微调指南

当零样本效果不满足需求时，可以用少量标注数据对 UIE 进行微调，使其适配特定领域（如法律、医疗、金融等）。

### 工作原理

UIE 把所有抽取任务统一为 **span 抽取**，微调时的训练信号也是基于 span 的：

```
训练输入:  [CLS] <schema prompt> [SEP] <text> [SEP]
训练目标:  start_positions[] (0/1)  — span 起始 token 为 1
           end_positions[]   (0/1)  — span 结束 token 为 1
损失函数:  Binary Cross-Entropy（对每个 token 位置独立计算）
```

### 1. 准备标注数据

标注数据使用 **JSON Lines** 格式（每行一个 JSON 对象），支持三种标注类型：

#### 实体抽取

```jsonl
{"text": "张三于1985年出生于北京", "entities": [{"label": "人物", "start": 0, "end": 2}, {"label": "出生地", "start": 9, "end": 11}], "schema": ["人物", "出生地"]}
```

#### 关系抽取

```jsonl
{"text": "《三体》是刘慈欣创作的小说", "relations": [{"subject": {"label": "作品", "start": 1, "end": 3}, "predicate": "作者", "object": {"start": 4, "end": 7}}], "schema": ["作品", "作者"]}
```

#### 事件抽取

```jsonl
{"text": "张三入职百度", "events": [{"label": "入职", "trigger": {"start": 2, "end": 4}, "arguments": [{"role": "人物", "start": 0, "end": 2}, {"role": "公司", "start": 4, "end": 6}]}]}
```

> **字段说明**
> - `text`：原始文本
> - `start` / `end`：**字符级半开区间** `[start, end)`，即 `text[start:end]` 为标注内容
> - `schema`：该文本涉及的 schema 标签列表（用于自动生成负样本，提升鲁棒性，可选）
> - `entities` / `relations` / `events`：三种标注类型，可在同一条数据中混用

项目中提供了示例数据：`data/train.jsonl`、`data/dev.jsonl`、`data/relations_example.jsonl`。

<details>
<summary>📖 标注数据的来源（点击展开）</summary>

标注数据通常来自以下途径：

| 来源 | 说明 |
|------|------|
| **公开数据集** | DuIE 2.0（百度关系抽取）、DuEE 1.0（事件抽取）、CLUE、MSRA NER 等 |
| **人工标注** | 使用 doccano、Label Studio 等标注工具对业务文本进行标注 |
| **数据蒸馏** | 用大模型（如 GPT）自动标注 + 人工校验 |

PaddleNLP 官方提供了从 [doccano](https://github.com/doccano/doccano) 导出数据转换为 UIE 格式的脚本（`doccano.py`），可直接参考。
</details>

### 2. 运行微调

#### 命令行方式

```bash
python -m uie_pytorch.finetune \
    --model weights/uie-base \
    --train_data data/train.jsonl \
    --dev_data data/dev.jsonl \
    --output_dir weights/uie-finetuned \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5
```

完整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `weights/uie-base` | 预训练 UIE 模型目录 |
| `--train_data` | （必填） | 训练集 JSONL 路径 |
| `--dev_data` | `None` | 验证集 JSONL 路径（提供则按 dev loss 保存最优模型） |
| `--output_dir` | `weights/uie-finetuned` | 微调模型输出目录 |
| `--epochs` | `3` | 训练轮数 |
| `--batch_size` | `16` | 每设备 batch size |
| `--lr` | `2e-5` | 峰值学习率（AdamW） |
| `--weight_decay` | `0.01` | 权重衰减 |
| `--warmup_ratio` | `0.1` | warmup 占总步数比例 |
| `--max_seq_len` | `512` | 最大序列长度 |
| `--negative_ratio` | `0.2` | 负样本（无答案）采样比例 |
| `--grad_accum_steps` | `1` | 梯度累积步数（有效 batch = batch_size × grad_accum_steps） |
| `--max_grad_norm` | `1.0` | 最大梯度范数（梯度裁剪） |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `--seed` | `42` | 随机种子 |

#### Python API 方式

```python
from uie_pytorch import train_uie

train_uie(
    model_path="weights/uie-base",
    train_data="data/train.jsonl",
    dev_data="data/dev.jsonl",
    output_dir="weights/uie-finetuned",
    epochs=3,
    batch_size=16,
    lr=2e-5,
    negative_ratio=0.2,
    device="auto",
)
```

### 3. 使用微调后的模型推理

微调保存的权重格式与 `UIEExtractor` 完全兼容，直接加载即可：

```python
from uie_pytorch import UIEExtractor

ie = UIEExtractor(
    model="weights/uie-finetuned",   # ← 指向微调后的目录
    schema=["人物", "出生年份", "出生地"],
    device="auto",
)

result = ie("钱七于1992年在杭州出生，后考入清华大学。")
print(result)
```

### 4. 运行微调示例

```bash
python examples/finetune_example.py
```

该脚本会自动完成：微调 → 加载微调模型 → 执行推理的完整流程。

---

## 运行示例

项目内置了丰富的示例脚本：

```bash
# 中文实体抽取
python examples/entity_extraction.py

# 中文关系抽取
python examples/relation_extraction.py

# 英文实体抽取（需先转换 uie-base-en 权重）
python examples/entity_extraction_en.py

# 微调 + 推理端到端示例
python examples/finetune_example.py
```

---

## 运行测试

```bash
# 安装测试依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v
```

---

## 设备加速

### NVIDIA GPU (CUDA)

```python
ie = UIEExtractor(model="weights/uie-base", schema=["时间"], device="cuda")
# 或 device="auto" 会自动检测
```

### Apple Silicon (MPS)

在 Mac M1/M2/M3/M4 上，PyTorch 通过 MPS（Metal Performance Shaders）利用 Apple GPU 和统一内存加速：

```python
ie = UIEExtractor(model="weights/uie-base", schema=["时间"], device="mps")
# 或 device="auto" 会自动检测
```

> **要求**：PyTorch >= 1.12（MPS 从 1.12 开始引入，2.0+ 后稳定）
```bash
# Mac 上安装带 MPS 支持的 PyTorch
pip install torch transformers numpy
```

```python
# 验证 MPS 是否可用
import torch
print(torch.backends.mps.is_available())  # True 表示可用
```

### CPU

```python
ie = UIEExtractor(model="weights/uie-base", schema=["时间"], device="cpu")
```

> UIE-base 模型仅 118M 参数，CPU 推理单条文本约 50-100ms，日常使用足够。GPU/MPS 加速在大批量抽取时才有明显优势。

---

## 项目结构

```
uie_pytorch/
├── src/uie_pytorch/            # 核心 Python 包
│   ├── __init__.py             # 包入口
│   ├── model.py                # UIE 模型定义（ErnieModel + linear_start/end）
│   ├── extractor.py            # UIEExtractor 推理管线（零样本抽取）
│   ├── utils.py                # SchemaTree、解码工具函数
│   ├── converter.py            # PaddlePaddle → PyTorch 权重转换器
│   ├── dataset.py              # 微调数据集（JSONL → UIE 训练张量）
│   ├── trainer.py              # 微调训练循环（BCE + AdamW + warmup）
│   └── finetune.py             # 微调命令行入口
├── examples/                   # 使用示例
│   ├── entity_extraction.py
│   ├── relation_extraction.py
│   ├── entity_extraction_en.py
│   └── finetune_example.py     # 微调端到端示例
├── data/                       # 示例标注数据
│   ├── train.jsonl             # 实体抽取训练示例
│   ├── dev.jsonl               # 验证集示例
│   └── relations_example.jsonl # 关系抽取示例
├── tests/                      # 测试
│   ├── test_model.py
│   └── test_extractor.py
├── weights/                    # 权重文件（gitignore，转换后生成）
├── pyproject.toml              # 项目配置（setuptools）
├── requirements.txt            # 依赖
├── .gitignore
├── LICENSE
└── README.md
```

---

## 工作原理

UIE 的核心思想是将所有信息抽取任务统一为 **span 抽取**：

```
输入: [CLS] <schema prompt> [SEP] <text> [SEP]
模型: ERNIE 编码 → linear_start → sigmoid → start_prob[]
                       ↓
                linear_end → sigmoid → end_prob[]
解码: 配对 start/end → 文本片段
```

- **实体抽取**：prompt = 字段名（如 `"时间"`）
- **关系抽取**：prompt = `"实体值" + 关系名`（如 `"谷爱凌的出生地"`）
- **事件抽取**：嵌套 schema 树，多阶段预测

### 训练 vs 推理

| 阶段 | 输入 | 目标/输出 | 损失 |
|------|------|-----------|------|
| **训练/微调** | prompt + text | `start_ids[]`, `end_ids[]`（0/1 标签） | Binary Cross-Entropy |
| **推理** | prompt + text | `start_prob[]`, `end_prob[]`（概率） → 解码为文本片段 | — |

### 与 PaddleNLP 的对应关系

| PaddleNLP | 本项目 |
|-----------|--------|
| `paddlenlp.transformers.ernie.modeling.UIE` | `uie_pytorch.model.UIE` |
| `paddlenlp.taskflow.information_extraction.UIETask` | `uie_pytorch.extractor.UIEExtractor` |
| `paddlenlp.taskflow.utils.SchemaTree` | `uie_pytorch.utils.SchemaTree` |
| `.pdparams` | `.bin`（通过 `converter` 转换） |

---

## API 参考

### `UIEExtractor`（推理）

```python
UIEExtractor(
    model: str = "uie-base",       # 本地路径或预定义模型名
    schema=None,                    # 抽取 schema
    position_prob: float = 0.5,     # 起止概率阈值
    max_seq_len: int = 512,         # 最大序列长度
    batch_size: int = 16,           # 推理 batch size
    device: str = "auto",           # "auto" / "cpu" / "cuda" / "mps"
    split_sentence: bool = False,   # 长文本按句子切分
)
```

**设备支持：**

| device | 说明 |
|--------|------|
| `"auto"` | 自动检测，优先级：CUDA → MPS → CPU |
| `"cuda"` | NVIDIA GPU |
| `"mps"` | Apple Silicon（M1/M2/M3/M4）统一内存 GPU |
| `"cpu"` | CPU |

**方法：**
- `set_schema(schema)` — 设置抽取 schema
- `__call__(inputs)` — 执行抽取，返回 `List[dict]`

### `train_uie`（微调）

```python
from uie_pytorch import train_uie

train_uie(
    model_path: str,             # 预训练 UIE 模型目录
    train_data: str,             # 训练集 JSONL 路径
    output_dir: str,             # 输出目录
    dev_data: str = None,        # 验证集 JSONL 路径（可选）
    epochs: int = 3,             # 训练轮数
    batch_size: int = 16,        # batch size
    lr: float = 2e-5,            # 学习率
    weight_decay: float = 0.01,  # 权重衰减
    warmup_ratio: float = 0.1,   # warmup 比例
    max_seq_len: int = 512,      # 最大序列长度
    negative_ratio: float = 0.2, # 负样本采样比例
    grad_accum_steps: int = 1,   # 梯度累积步数
    max_grad_norm: float = 1.0,  # 梯度裁剪
    save_best: bool = True,      # 按 dev loss 保存最优模型
    device: str = "auto",        # 设备
    seed: int = 42,              # 随机种子
)
```

### `UIEDataset`（数据集）

```python
from uie_pytorch import UIEDataset

ds = UIEDataset(
    data_path="data/train.jsonl",  # JSONL 文件路径
    tokenizer=tokenizer,           # BertTokenizerFast 实例
    max_seq_len=512,               # 最大序列长度
    negative_ratio=0.2,            # 负样本采样比例
)
```

---

## 技术细节

### 权重转换

PaddlePaddle 和 PyTorch 的 `nn.Linear` 权重布局不同：
- PaddlePaddle: `(in_features, out_features)`
- PyTorch: `(out_features, in_features)`

转换器自动对所有 Linear 层权重做转置。Embedding 层布局一致，无需处理。

### ERNIE vs BERT

ERNIE 比 BERT 多一个 **task_type_embeddings** 表（`task_type_vocab_size=3`），这是百度 ERNIE 的独有设计。模型中已包含该层。

### 微调损失函数

UIE 微调使用 **Binary Cross-Entropy (BCE)** 对每个 token 位置独立计算损失：

```python
loss = (BCE(start_prob, start_ids) + BCE(end_prob, end_ids)) / 2
```

其中 `start_prob` / `end_prob` 是模型经 sigmoid 后的输出（`(B, L)`），`start_ids` / `end_ids` 是 0/1 标签，span 起止 token 位置为 1。

---

## 致谢

- **PaddleNLP** — 百度飞桨自然语言处理开发套件，本项目的所有模型权重和架构设计均来源于此
- **UIE 论文** — Lu et al., "Unified Structure Generation for Universal Information Extraction", ACL 2022

## 许可

[Apache License 2.0](LICENSE)，继承自 PaddleNLP。
