# UIE-PyTorch

> **PaddleNLP UIE 模型的 PyTorch 复刻版** —— 从 PaddleNLP 原始代码提炼，用纯 PyTorch + HuggingFace Transformers 实现相同的零样本/少样本信息抽取能力。

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 简介

UIE（Universal Information Extraction）是一种统一的文本信息抽取模型，能够通过自然语言描述的 schema 来完成**实体抽取、关系抽取、事件抽取**等任务，无需训练即可使用（零样本）。

本项目是 [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) 中 UIE 的 **PyTorch 独立实现**，不依赖 PaddlePaddle 框架。模型结构和预训练权重与官方版本完全对齐，抽取结果一致。

### 核心特性

- ✅ **完全复刻** PaddleNLP UIE 模型结构（ERNIE 编码器 + 双线性抽取头）
- ✅ **权重转换**：自动下载 PaddlePaddle `.pdparams` 并转换为 PyTorch `.bin`
- ✅ **零样本抽取**：实体抽取、关系抽取、事件抽取（嵌套 schema 树）
- ✅ **API 兼容**：与 PaddleNLP `Taskflow("information_extraction")` 用法一致
- ✅ **纯 PyTorch**：不依赖 PaddlePaddle 框架
- ✅ **多设备支持**：CUDA（NVIDIA）、MPS（Apple Silicon）、CPU 自动检测

---

## 安装

### 从源码安装（开发模式）

```bash
git clone <your-repo-url>
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

## 运行示例

项目内置了丰富的示例脚本：

```bash
# 中文实体抽取
python examples/entity_extraction.py

# 中文关系抽取
python examples/relation_extraction.py

# 英文实体抽取（需先转换 uie-base-en 权重）
python examples/entity_extraction_en.py
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
├── src/uie_pytorch/          # 核心 Python 包
│   ├── __init__.py           # 包入口
│   ├── model.py              # UIE 模型定义（ErnieModel + linear_start/end）
│   ├── extractor.py          # UIEExtractor 推理管线（零样本抽取）
│   ├── utils.py              # SchemaTree、解码工具函数
│   └── converter.py          # PaddlePaddle → PyTorch 权重转换器
├── examples/                  # 使用示例
│   ├── entity_extraction.py
│   ├── relation_extraction.py
│   └── entity_extraction_en.py
├── tests/                     # 测试
│   ├── test_model.py
│   └── test_extractor.py
├── weights/                   # 权重文件（gitignore，转换后生成）
├── pyproject.toml             # 项目配置（setuptools）
├── requirements.txt           # 依赖
├── requirements-dev.txt       # 开发依赖
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

### 与 PaddleNLP 的对应关系

| PaddleNLP | 本项目 |
|-----------|--------|
| `paddlenlp.transformers.ernie.modeling.UIE` | `uie_pytorch.model.UIE` |
| `paddlenlp.taskflow.information_extraction.UIETask` | `uie_pytorch.extractor.UIEExtractor` |
| `paddlenlp.taskflow.utils.SchemaTree` | `uie_pytorch.utils.SchemaTree` |
| `.pdparams` | `.bin`（通过 `converter` 转换） |

---

## API 参考

### `UIEExtractor`

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
- `set_schema(...)` 后可重复调用 `__call__`

---

## 技术细节

### 权重转换

PaddlePaddle 和 PyTorch 的 `nn.Linear` 权重布局不同：
- PaddlePaddle: `(in_features, out_features)`
- PyTorch: `(out_features, in_features)`

转换器自动对所有 Linear 层权重做转置。Embedding 层布局一致，无需处理。

### ERNIE vs BERT

ERNIE 比 BERT 多一个 **task_type_embeddings** 表（`task_type_vocab_size=3`），这是百度 ERNIE 的独有设计。模型中已包含该层。

---

## 致谢

- **PaddleNLP** — 百度飞桨自然语言处理开发套件，本项目的所有模型权重和架构设计均来源于此
- **UIE 论文** — Lu et al., "Unified Structure Generation for Universal Information Extraction", ACL 2022

## 许可

[Apache License 2.0](LICENSE)，继承自 PaddleNLP。
