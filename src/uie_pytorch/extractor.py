"""基于 PyTorch UIE 模型实现的零样本 (Zero-shot) / 少样本 (Few-shot) 通用信息抽取器。

``UIEExtractor`` 是对 PaddleNLP 官方 ``Taskflow("information_extraction", ...)`` API 的高性能 PyTorch 1:1 像素级替代实现。
它支持加载转换后的 PyTorch 格式权重，并对外暴露了极简的 ``__call__`` 接口。
底层通过**嵌套 Schema 树（SchemaTree）**的设计，完美支持**实体提取（Entity）、关系抽取（Relation）以及事件抽取（Event）**。
"""

from __future__ import annotations

import os
from typing import List, Union

import numpy as np
import torch
from transformers import BertTokenizerFast, BertConfig

from .model import UIE
from .utils import (
    SchemaTree,
    build_schema_tree,
    dbc2sbc,
    auto_splitter,
    get_bool_ids_greater_than,
    get_span,
    get_id_and_prob,
)

# 默认的 HuggingFace 权重仓库（托管转换后的 PyTorch 权重）
DEFAULT_REPO = "PaddlePaddle/uie-base"

# 支持的一键导入模型权重字典（涵盖官方全系列轻量化及多语言 UIE 模型）
SUPPORTED_MODELS = {
    "uie-base": "PaddlePaddle/uie-base",
    "uie-medium": "PaddlePaddle/uie-medium",
    "uie-mini": "PaddlePaddle/uie-mini",
    "uie-micro": "PaddlePaddle/uie-micro",
    "uie-nano": "PaddlePaddle/uie-nano",
    "uie-base-en": "PaddlePaddle/uie-base-en",
    "uie-medical-base": "PaddlePaddle/uie-medical-base",
    "uie-m-base": "PaddlePaddle/uie-m-base",
    "uie-m-large": "PaddlePaddle/uie-m-large",
}


class UIEExtractor:
    """零样本/少样本 通用信息抽取管道（Pipeline）。

    参数说明：
        model: 本地权重目录路径，或者是 ``SUPPORTED_MODELS`` 中的模型 Key。
        schema: 抽取目标的 Schema 结构。
            - 实体抽取：``["时间", "选手", "赛事名称"]``
            - 关系/三元组抽取：``[{"人物": ["出生地", "毕业院校"]}]``
            - 也可以在实例化后，随时调用 ``set_schema()`` 动态重设。
        position_prob: 概率过滤阈值。起点/终点预测概率大于该值时，才会被判定为实体边界。
        max_seq_len: 输入序列的最大截断长度（包含 Prompt 和 Text 拼接后的总长）。
        batch_size: 推理时的 Batch 大小，根据显存或内存大小调整。
        device: 推理设备。支持 "cpu", "cuda", "cuda:0", "mps", 或 "auto"（自动选择最优硬件）。
        split_sentence: 是否开启自动拆分超长文本句子的功能，防止长文被硬生生截断导致漏抽。
    """

    def __init__(
            self,
            model: str = "uie-base",
            schema=None,
            position_prob: float = 0.5,
            max_seq_len: int = 512,
            batch_size: int = 16,
            device: str = "auto",
            split_sentence: bool = False,
    ):
        # 1. 路径解析与核心组件加载 ------------------------------------------
        model_path = SUPPORTED_MODELS.get(model, model)

        # 加载分词器、配置文件和模型
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        config = BertConfig.from_pretrained(model_path)
        self.model = UIE(config)
        self._load_weights(model_path)

        # 2. 核心推理参数配置 -----------------------------------------------
        self._position_prob = position_prob
        self._max_seq_len = max_seq_len
        self._batch_size = batch_size
        self._split_sentence = split_sentence

        # 💡 [CLS] + [SEP] + [SEP] 共 3 个特殊 Token 占位符
        self._summary_token_num = 3

        # 检测是否为英文 UIE 模型（英文模型的多阶段多轮 Prompt 拼接模板与中文不同）
        self._is_en = "en" in model_path.lower() or "base-en" in model.lower()

        # 3. 硬件设备分配与设置为评估状态 --------------------------------------
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

        # 4. 初始化 Schema Tree 结构
        self._schema_tree: SchemaTree | None = None
        if schema is not None:
            self.set_schema(schema)

    # ------------------------------------------------------------------
    # 权重与硬件加载逻辑
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """解析运行设备字符串。

        在 "auto" 模式下，设备选择优先级为：
        1. CUDA (NVIDIA 显卡)
        2. MPS (Apple Silicon Mac 芯片加速，需要 PyTorch >= 1.12)
        3. CPU (通用兜底)
        """
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_weights(self, model_path: str):
        """安全载入本地 PyTorch 权重 ``pytorch_model.bin`` 并在载入后校验 key。"""
        pt_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(pt_file):
            try:
                # 兼容 PyTorch 2.0+ 推荐的安全反序列化机制
                state_dict = torch.load(pt_file, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(pt_file, map_location="cpu")

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

            # 过滤不需要理会的 key（例如非持久化的 position_ids 缓存区）
            real_missing = [k for k in missing if not k.endswith("position_ids")]
            if real_missing:
                print(f"警告: 缺失了 {len(real_missing)} 个权重 Key: {real_missing[:5]}")
            if unexpected:
                print(f"警告: 发现了 {len(unexpected)} 个多余/未预期的 Key: {unexpected[:5]}")

    # ------------------------------------------------------------------
    # 外部调用公共 API
    # ------------------------------------------------------------------
    def set_schema(self, schema):
        """动态设置或重置当前的抽取 Schema Tree。"""
        self._schema_tree = build_schema_tree(schema)

    def __call__(self, inputs: Union[str, List[str]]) -> List[dict]:
        """对输入的文本执行通用信息抽取。

        参数：
            inputs: 待抽取的单条文本字符串，或者多条文本字符串组成的列表。

        返回：
            格式化后的抽取字典结果列表，例如：
            ``[{"人物": [{"text": "张三", "start": 0, "end": 2, "probability": 0.99}]}]``
        """
        if self._schema_tree is None:
            raise ValueError("请先调用 set_schema() 配置您的抽取 Schema。")
        if isinstance(inputs, str):
            inputs = [inputs]
        return self._multi_stage_predict(inputs)

    # ------------------------------------------------------------------
    # 核心算法一：多阶段预测（Schema 树深度优先遍历）
    # ------------------------------------------------------------------
    def _multi_stage_predict(self, data: List[str]) -> List[dict]:
        """多阶段、多轮交互式前向抽取算法。

        💡 深度解析（UIE 如何做关系抽取？）：
            1. 首先以 SchemaTree 的根节点直接子节点（如“人物”）作为第一阶段的 Prompt。
            2. 将“人物”送入模型，抽取出文本中的所有具体实体值（如“张三”、“李四”）。
            3. 如果“人物”节点下存在子节点（如“出生地”），则启动第二阶段（下一轮循环）：
               将上一阶段抽取出的实体值，动态拼接组装成新的 Prompt：
               - 中文拼接模板：``"<上一阶段实体值>的<当前子节点Schema>"``（例如："张三的出生地"）
               - 英文拼接模板：``"<当前子节点Schema> of <上一阶段实体值>"``（例如："birthplace of Zhang San"）
            4. 将新 Prompt 再次送入模型进行关系论元的抽取，以此类推，支持无限级嵌套抽取。
        """
        results = [{} for _ in range(len(data))]
        if len(data) < 1 or self._schema_tree is None:
            return results

        # 浅拷贝一份 SchemaTree 节点列表，采用队列方式进行 BFS/DFS 级别遍历
        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0

            if not node.prefix:
                # 情况 A：根级 Schema 抽取（一阶段：如单纯提取“人物”）
                for one_data in data:
                    # 全角转半角，并构建待抽取的 Prompt
                    examples.append({"text": one_data, "prompt": dbc2sbc(node.name)})
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                # 情况 B：嵌套/叶子级 Schema 抽取（多阶段：根据父代输出动态拼接生成新 Prompt）
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            # 按照语言模板，动态进行多阶段 Prompt 组装
                            if self._is_en:
                                if node.name.startswith("[") or "[" in node.name:
                                    prompt = f"{node.name} {p}"
                                else:
                                    prompt = node.name + " of " + p
                            else:
                                prompt = p + node.name
                            examples.append({"text": one_data, "prompt": dbc2sbc(prompt)})
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1

            # 若没有任何拼接出的样例，说明上一阶段未抽取到任何前置关联实体，直接跳过当前节点
            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            # 结果融合（将多阶段零碎的抽取结果归并合并到主字典树上） ------------------
            if not node.parent_relations:
                relations = [[] for _ in range(len(data))]
                for k, v in input_map.items():
                    for index in v:
                        if len(result_list[index]) == 0:
                            continue
                        if node.name not in results[k]:
                            results[k][node.name] = result_list[index]
                        else:
                            results[k][node.name].extend(result_list[index])
                    if node.name in results[k]:
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if "relations" not in relations[k][i]:
                            relations[k][i]["relations"] = {node.name: result_list[v[i]]}
                        elif node.name not in relations[k][i]["relations"]:
                            relations[k][i]["relations"][node.name] = result_list[v[i]]
                        else:
                            relations[k][i]["relations"][node.name].extend(result_list[v[i]])
                # 展平结构，以便为下一层深度的嵌套子节点做准备
                new_relations = [[] for _ in range(len(data))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if "relations" in relations[i][j] and node.name in relations[i][j]["relations"]:
                            for item in relations[i][j]["relations"][node.name]:
                                new_relations[i].append(item)
                relations = new_relations

            # 为下一轮循环中的所有子节点（Children）提前构建前缀关联数据（prefix）
            prefix = [[] for _ in range(len(data))]
            for k, v in input_map.items():
                for index in v:
                    for i in range(len(result_list[index])):
                        if self._is_en:
                            prefix[k].append(" of " + result_list[index][i]["text"])
                        else:
                            prefix[k].append(result_list[index][i]["text"] + "的")

            # 递归地将当前层的关联数据（relations/prefix）下发给所有子节点，并加入到待执行队列中
            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)

        return results

    # ------------------------------------------------------------------
    # 核心算法二：单阶段预测（执行实际的模型前向计算与后处理解码）
    # ------------------------------------------------------------------
    def _single_stage_predict(self, inputs: List[dict]) -> List[dict]:
        """单阶段底层抽取逻辑。负责处理长文本切分、模型 Batch 推理、阈值过滤、指针合并等。"""
        input_texts = [d["text"] for d in inputs]
        prompts = [d["prompt"] for d in inputs]
        max_prompt_len = max(len(p) for p in prompts)

        # 计算当前输入的 Text 在单次计算中能容纳的最大长度（扣除 Prompt 及 3 个特殊符号）
        max_predict_len = self._max_seq_len - max_prompt_len - self._summary_token_num

        # 1. 切分长文本（保证每段子句和 prompt 拼接后不溢出最大限制）
        short_input_texts, input_mapping = auto_splitter(
            input_texts, max_predict_len, split_sentence=self._split_sentence
        )
        short_texts_prompts = []
        for k, v in input_mapping.items():
            short_texts_prompts.extend([prompts[k] for _ in range(len(v))])

        short_inputs = [
            {"text": short_input_texts[i], "prompt": short_texts_prompts[i]}
            for i in range(len(short_input_texts))
        ]

        # 2. 分批次进行 Tokenization 和推理 --------------------------------------
        sentence_ids = []
        probs = []
        for batch_start in range(0, len(short_inputs), self._batch_size):
            batch = short_inputs[batch_start: batch_start + self._batch_size]
            batch_start_probs = []
            batch_end_probs = []
            batch_offset_maps = []
            for example in batch:
                # 编码并前向传播
                encoded = self._encode_example(example)
                batch_start_probs.append(encoded["start_prob"])
                batch_end_probs.append(encoded["end_prob"])
                batch_offset_maps.append(encoded["offset_mapping"])

            # 3. 对批次内的每一个样例结果进行后处理解码 ------------------------------
            for start_prob, end_prob, offset_map in zip(
                    batch_start_probs, batch_end_probs, batch_offset_maps
            ):
                # 过滤出概率大于 position_prob 的起点和终点
                start_ids_list = get_bool_ids_greater_than(
                    start_prob, limit=self._position_prob, return_prob=True
                )
                end_ids_list = get_bool_ids_greater_than(
                    end_prob, limit=self._position_prob, return_prob=True
                )
                # 使用双指针匹配起止 Span
                span_set = get_span(start_ids_list, end_ids_list, with_prob=True)
                # 还原至物理字符偏移量
                sentence_id, prob = get_id_and_prob(span_set, offset_map)
                sentence_ids.append(sentence_id)
                probs.append(prob)

        # 4. 转换 ID 为结构化字典，并将之前切分的超长句子合并还原
        results = self._convert_ids_to_results(short_inputs, sentence_ids, probs)
        results = self._auto_joiner(results, short_input_texts, input_mapping)
        return results

    # ------------------------------------------------------------------
    # 核心算法三：底层编码与前向传播
    # ------------------------------------------------------------------
    def _encode_example(self, example: dict) -> dict:
        """对单条 (prompt, text) 样例执行序列填充，并无梯度运行模型。"""
        prompt = example["prompt"]
        text = example["text"]

        # 💡 特别注意：UIE 的输入对（text_pair）使用的是 HuggingFace 默认的 `(text=prompt, text_pair=text)`
        encoding = self.tokenizer(
            prompt,
            text,
            truncation=True,
            max_length=self._max_seq_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        token_type_ids = encoding["token_type_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 禁用梯度，节省推理时的显存和速度
        with torch.no_grad():
            start_prob, end_prob = self.model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )

        # 转回 CPU 的 numpy 格式方便后续在 cpu 上进行快速解析和双指针扫描
        start_prob = start_prob.squeeze(0).cpu().numpy()
        end_prob = end_prob.squeeze(0).cpu().numpy()
        offset_mapping = encoding["offset_mapping"].squeeze(0).cpu().numpy().tolist()
        return {
            "start_prob": start_prob,
            "end_prob": end_prob,
            "offset_mapping": offset_mapping,
        }

    # ------------------------------------------------------------------
    # 结果拼装与长句还原辅助方法
    # ------------------------------------------------------------------
    def _convert_ids_to_results(self, examples, sentence_ids, probs):
        """将 token 级还原得到的文本位置转换成标准结果字典格式。"""
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example["text"]
            prompt = example["prompt"]
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    # 特殊边缘情况：如果抽取的实体意外落在了 Prompt 本身内部（通常由极不合理的极端输入导致）
                    # 重新将偏移量拉回到 prompt 上
                    start += len(prompt) + 1
                    end += len(prompt) + 1
                    result = {"text": prompt[start:end], "probability": prob[i]}
                    result_list.append(result)
                else:
                    # 正常情况：提取在 Text 内的实体切片，附带其置信概率
                    result = {
                        "text": text[start:end],
                        "start": start,
                        "end": end,
                        "probability": prob[i],
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    def _auto_joiner(self, short_results, short_inputs, input_mapping):
        """长句结果自动拼装合并器。

        对于之前被 `auto_splitter` 切分成多段的超长文本，
        此函数通过加上偏移量累积，把多段子句上的抽取结果和实体位置（start/end）无缝拼接成相对于整篇原始文本的绝对索引。
        """
        concat_results = []
        for k, vs in input_mapping.items():
            offset = 0
            single_results = []
            for v in vs:
                # 若为第一段子句，直接作为起始基础结果，并累加偏移
                if v == 0:
                    single_results = short_results[v]
                    offset += len(short_inputs[v])
                else:
                    # 对于非首段子句，所有抽取实体的 start 和 end 偏移量必须累加之前段落的总字符长度
                    for i in range(len(short_results[v])):
                        if "start" not in short_results[v][i] or "end" not in short_results[v][i]:
                            continue
                        short_results[v][i]["start"] += offset
                        short_results[v][i]["end"] += offset
                    offset += len(short_inputs[v])
                    single_results.extend(short_results[v])
            concat_results.append(single_results)
        return concat_results
