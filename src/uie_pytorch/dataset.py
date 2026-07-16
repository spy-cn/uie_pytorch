"""用于微调 UIE 模型的 Dataset 数据处理工具。

UIE (Universal Information Extraction) 将所有的信息抽取任务统一转化为【区间/片段抽取 (Span Extraction)】：
    输入序列  = [CLS] <Schema 提示词 (Prompt)> [SEP] <待抽取的文本 (Text)> [SEP]
    目标标签  = start_positions[] (起点向量), end_positions[] (终点向量) -> 每个 Token 位置输出 0 或 1

本模块负责将标注好的原始 JSONL 样本（支持实体、关系、事件抽取标注）转换成 PyTorch UIE 训练所需的 Tensor。

支持的标注数据格式 (JSON Lines，每行一个 JSON 对象)：

    1. 实体抽取标注示例:
    {"text": "张三在腾讯工作。", "entities": [{"label": "人名", "start": 0, "end": 2}]}

    2. 关系抽取标注示例:
    {"text": "闪电侠的研发者是DC。", "relations": [{"subject": {"start": 0, "end": 3},
                                        "predicate": "研发者",
                                        "object": {"start": 7, "end": 9}}]}
注意：字符级的偏移量 (Offsets) 均采用【左闭右开】区间，即 ``[start, end)``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class UIEExample:
    """包装单个训练样本的 PyTorch Tensor 数据结构。

    该结构中的所有张量都是最终直接送入 UIE 模型前向传播 (forward) 的输入。
    """

    input_ids: torch.Tensor  # Token 序列在词表中的索引 ID, 形状: (seq_len,)
    token_type_ids: torch.Tensor  # 区分 Prompt (通常为0) 和 Text (通常为1) 的 Segment Embedding 标志, 形状: (seq_len,)
    attention_mask: torch.Tensor  # 掩码向量（区分真实 Token 与 [PAD] 填充字符）, 形状: (seq_len,)
    start_ids: torch.Tensor  # 实体起点指针标注（多标签二分类：1.0 表示起点，0.0 表示非起点）, 形状: (seq_len,)
    end_ids: torch.Tensor  # 实体终点指针标注（多标签二分类：1.0 表示终点，0.0 表示非终点）, 形状: (seq_len,)


def _resolve_spans(record: Dict[str, Any]) -> List[tuple]:
    """从单条标注记录中，解析并生成统一的 (prompt_text, char_start, char_end) 列表。

    UIE 的多任务大一统原理：
      - 实体抽取 (Entity)  -> 直接使用 "实体类别" (如: "人名") 作为 Prompt，抽取实体片段。
      - 关系抽取 (Relation)-> 第一阶段抽取主语 (Subject)；第二阶段使用 "主语+的+关系名" (如: "张三的出生地")
                             作为 Prompt，抽取宾语 (Object)。
      - 事件抽取 (Event)   -> 第一阶段使用 "事件类别" 抽取触发词 (Trigger)；第二阶段使用 "触发词+的+角色名"
                             (如: "地震的发生时间") 作为 Prompt，抽取事件论元 (Arguments)。
    """
    text_len = len(record["text"])
    results: List[tuple] = []

    # 1. 解析已展平/自定义的简单区间：[{"prompt": "...", "start": i, "end": j}]
    for sp in record.get("spans", []):
        results.append((sp["prompt"], sp["start"], sp["end"]))

    # 2. 解析命名实体识别 (NER) 的标注
    for ent in record.get("entities", []):
        results.append((ent["label"], ent["start"], ent["end"]))

    # 3. 解析关系抽取 (RE) 的标注
    for rel in record.get("relations", []):
        subj = rel["subject"]
        obj = rel["object"]
        predicate = rel["predicate"]

        # 3.1 提取主语作为第一步的训练实体（Prompt 默认为“实体”，或使用 subj 自带的 label）
        results.append((subj.get("label", "实体"), subj["start"], subj["end"]))

        # 3.2 根据主语文本构建关系抽取的 Prompt：“<主语文本>的<关系/谓语>”
        subj_text = record["text"][subj["start"]:subj["end"]]
        prompt = f"{subj_text}的{predicate}"
        results.append((prompt, obj["start"], obj["end"]))

    # 4. 解析事件抽取 (EE) 的标注
    for ev in record.get("events", []):
        trigger = ev.get("trigger", {})
        # 4.1 提取事件触发词
        if trigger:
            results.append((ev.get("label", "事件"), trigger["start"], trigger["end"]))

        # 4.2 根据触发词构建论元抽取的 Prompt：“<触发词文本>的<角色>”
        for arg in ev.get("arguments", []):
            trig_text = record["text"][trigger["start"]:trigger["end"]] if trigger else ""
            prompt = f"{trig_text}的{arg['role']}" if trig_text else arg["role"]
            results.append((prompt, arg["start"], arg["end"]))

    # 过滤掉由于越界、负索引或无效标注造成的异常区间，确保区间合法且处于 [0, 文本长度] 内
    return [(p, s, e) for (p, s, e) in results if 0 <= s < e <= text_len]


def char_span_to_token_span(
        offset_mapping: Sequence[Sequence[int]],
        char_start: int,
        char_end: int,
) -> Optional[tuple]:
    """将【字符级别】的 [start, end) 偏移量映射为【Token 级别】的 [start, end) 索引。

    因为 BERT / ERNIE 分词后，原本的一个词可能会被拆分为多个 Subtokens，所以必须使用
    HuggingFace Tokenizer 输出的 ``offset_mapping`` 来进行索引校准。

    返回值：
        若成功对齐，返回 (tok_start, tok_end)；
        若区间未能成功对应到任何 Token（例如由于长文本被截断，导致实体落在了 max_seq_len 之外），则返回 ``None``。
    """
    tok_start = None
    tok_end = None

    for idx, (cs, ce) in enumerate(offset_mapping):
        # 过滤特殊字符，如 [CLS]、[SEP]、[PAD] 等，它们的 offset 映射一般为 (0, 0)
        if cs == ce == 0:
            continue

        # 寻找 Token 级别的起点：只要字符起点落在当前 Token 的字符边界内即可
        if tok_start is None and cs <= char_start < ce:
            tok_start = idx

        # 寻找 Token 级别的终点：字符终点落在了当前 Token 的字符边界内
        if cs < char_end <= ce:
            tok_end = idx
            break

    if tok_start is None or tok_end is None:
        return None

    return tok_start, tok_end


class UIEDataset(Dataset):
    """用于 UIE 微调训练的 PyTorch Dataset 类。

    核心设计：
    一条原始数据（例如包含 3 个实体标注和 1 个关系标注）可能会被拆分、膨胀为「多个」训练实例。
    因为 UIE 每次前向传播只能针对【一个具体的 Schema Prompt】进行抽取。

    参数：
        data_path: 存储标注数据的 JSON-Lines 文件路径。
        tokenizer: HuggingFace Fast 分词器（通常为 ``BertTokenizerFast`` 或 ``ErnieTokenizerFast``）。
        max_seq_len: 输入序列的最大截断长度。
        negative_ratio: 负样本（即无答案的 Prompt 样本）的采样比例。
            设为 0 时表示关闭采样。适当比例的负样本可以显著抑制模型推理时的“幻觉”和过度召回。
    """

    def __init__(
            self,
            data_path: str,
            tokenizer,
            max_seq_len: int = 512,
            negative_ratio: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.instances: List[Dict[str, Any]] = []

        # 一次性加载所有原始标注 JSON Lines 数据
        with open(data_path, "r", encoding="utf-8-sig") as f:
            records = [json.loads(line) for line in f if line.strip()]

        positives: List[Dict[str, Any]] = []  # 存储正样本（即在该文本中能成功抽取出目标实体的样例）
        negatives: List[Dict[str, Any]] = []  # 存储负样本（即在该文本中不存在目标实体的样例）

        for record in records:
            text = record["text"]
            spans = _resolve_spans(record)

            # 1. 没有任何有效实体标注的情况 -> 直接作为潜在的负样本
            if not spans:
                schema_labels = record.get("schema", [])
                for label in schema_labels:
                    negatives.append({"prompt": label, "text": text})
                continue

            # 2. 构建正样本
            seen_prompts = set()
            for prompt, c_start, c_end in spans:
                positives.append(
                    {"prompt": prompt, "text": text, "char_start": c_start, "char_end": c_end}
                )
                seen_prompts.add(prompt)

            # 3. 自动构建负样本：
            # 如果某条数据中声明了 "schema"（如 ["人名", "地名", "组织机构"]），
            # 但当前文本中只抽出了 "人名"，那么 "地名" 和 "组织机构" 将作为负样本加入负样本池。
            schema_labels = record.get("schema", [])
            for label in schema_labels:
                if label not in seen_prompts:
                    negatives.append({"prompt": label, "text": text})

        # 默认使用全部正样本
        self.instances = positives

        # 4. 根据设定的比例，随机混入一部分负样本，以此增强模型在实际推理中的鲁棒性
        if negative_ratio > 0 and negatives:
            import random
            n_neg = int(len(positives) * negative_ratio)
            if n_neg > 0:
                # 采样并拼接负样本（负样本数据在 __getitem__ 中不会标注 start_ids 和 end_ids 标签）
                self.instances += random.sample(negatives, min(n_neg, len(negatives)))

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> UIEExample:
        """获取索引对应的训练实例，并通过 Tokenizer 进行编码与 Label 映射。"""
        inst = self.instances[idx]
        prompt = inst["prompt"]
        text = inst["text"]
        char_start = inst.get("char_start")
        char_end = inst.get("char_end")

        # 将 Prompt 和文本拼接编码。
        # ⚠️ 注意：UIE 的 Prompt 是微调的关键，必须保证训练和推理时的构造格式完全对称。
        encoding = self.tokenizer(
            prompt,
            text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_offsets_mapping=True,  # 必须返回字符偏移映射，用于计算指针索引
        )
        offset_mapping = encoding["offset_mapping"]
        seq_len = len(encoding["input_ids"])

        # 初始化起点和终点标签向量，默认全部填充为 0.0 (表示无答案/负样本)
        start_ids = [0.0] * seq_len
        end_ids = [0.0] * seq_len

        # 如果是正样本（含有实际标注），将其字符区间转换为 Token 区间，并在对应位置打上 1.0 标签
        if char_start is not None and char_end is not None:
            token_span = char_span_to_token_span(offset_mapping, char_start, char_end)
            if token_span is not None:
                ts, te = token_span
                start_ids[ts] = 1.0  # 标记起始指针位置
                end_ids[te] = 1.0  # 标记结束指针位置

        return UIEExample(
            input_ids=torch.tensor(encoding["input_ids"], dtype=torch.long),
            token_type_ids=torch.tensor(encoding["token_type_ids"], dtype=torch.long),
            attention_mask=torch.tensor(encoding["attention_mask"], dtype=torch.long),
            start_ids=torch.tensor(start_ids, dtype=torch.float),
            end_ids=torch.tensor(end_ids, dtype=torch.float),
        )
