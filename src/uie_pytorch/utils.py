"""用于 UIE (Universal Information Extraction) 零样本及微调推理的工具函数，移植自 PaddleNLP。

这些函数 —— 包括 ``SchemaTree`` (Schema树结构)、``get_bool_ids_greater_than`` (阈值过滤)、
``get_span`` (双指针边界匹配)、``get_id_and_prob`` (字符级索引还原)、``dbc2sbc`` (全角转半角)
以及 ``auto_splitter`` (长文本自动切分器) —— 均是对 ``paddlenlp.taskflow.information_extraction``
中所使用辅助工具的 PyTorch 1:1 像素级复现，保证了推理结果的数学对齐。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Schema 树结构 (SchemaTree) – 支持实体 / 关系 / 事件抽取的多层嵌套 Schema。
# 镜像复现了 ``paddlenlp.taskflow.utils.SchemaTree``。
# ---------------------------------------------------------------------------
class SchemaTree:
    """信息抽取 Schema 树中的单棵子树或节点。

    - 根节点的名称默认为 ``"root"``。
    - 根节点的每个子节点（children）代表一个待抽取的字段。
    - 子节点自身还可以继续拥有子节点，从而支持嵌套的**关系/事件抽取**。
      例如：``{"人物": ["出生地", "职业"]}`` 会被构建为：
      root -> 人物 -> [出生地, 职业]
    """

    def __init__(self, name: str = "root", children=None):
        self.name = name
        self.children: List[SchemaTree] = []
        self.prefix = None  # 在多阶段/多轮交互式抽取中保存中间前缀提示词
        self.parent_relations = None
        self.parent = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node: "SchemaTree"):
        """向当前 Schema 节点添加一个子节点。"""
        assert isinstance(node, SchemaTree), "子节点必须是 SchemaTree 的实例"
        self.children.append(node)


def build_schema_tree(schema) -> SchemaTree:
    """根据用户传入的嵌套 list / dict / str，递归构建一棵 ``SchemaTree`` 对象树。"""
    root = SchemaTree("root")
    # 如果用户只传入了单条 Schema 字符串或单字典，统一包装为 list 方便后续统一遍历
    if isinstance(schema, dict) or isinstance(schema, str):
        schema = [schema]
    for s in schema:
        root.add_child(_build_node(s))
    return root


def _build_node(s):
    """递归辅助函数：将不同 Python 数据结构（str / dict）转换为对应的 SchemaTree 节点。"""
    node = SchemaTree(name=s) if isinstance(s, str) else None
    if isinstance(s, str):
        return node

    # 解析嵌套字典，如 {"人物": ["出生地", "毕业院校"]}
    if isinstance(s, dict):
        for k, v in s.items():
            node = SchemaTree(name=k)
            if isinstance(v, list):
                for child_name in v:
                    if isinstance(child_name, str):
                        node.add_child(SchemaTree(child_name))
                    elif isinstance(child_name, dict):
                        # 处理更深层级的嵌套，例如事件抽取中的 触发词 -> 论元角色
                        for ck, cv in child_name.items():
                            child_node = SchemaTree(ck)
                            for gc in cv:
                                child_node.add_child(SchemaTree(gc))
                            node.add_child(child_node)
            return node
    raise TypeError(f"不支持的 Schema 节点数据类型: {s!r}")


# ---------------------------------------------------------------------------
# 后处理解码辅助函数 – 1:1 像素级移植自 PaddleNLP 官方推理后处理
# ---------------------------------------------------------------------------
def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """筛选并返回概率值超过设定阈值（limit）的所有 Token 的位置索引。

    参数：
        probs: 包含每个 Token 边界预测概率的列表或一维数组。
        limit: 过滤阈值，默认 0.5。
        return_prob: 若为 True，则返回包含 (token_index, probability) 的元组，否则仅返回 index。
    """
    probs = np.array(probs)
    # 若传入的是 Batch 级别的数据（多维），则递归处理每一行
    if probs.ndim > 1:
        return [get_bool_ids_greater_than(p, limit, return_prob) for p in probs]

    result = []
    for i, p in enumerate(probs):
        if p > limit:
            result.append((i, float(p)) if return_prob else i)
    return result


def get_span(start_ids, end_ids, with_prob=False):
    """核心匹配算法（双指针法）：将离散的起点和终点索引配对，组装成不重叠的 span (区间)。

    💡 算法原理解析：
        指针网络分别输出一排「起点概率」和一排「终点概率」。
        我们要将它们合理配对，配对规则为：对于任意一个合法的实体，其起点 $S$ 和终点 $E$ 必须满足 $S \le E$。
        并且采用最近邻配对原则，确保实体区间互不重叠且完全覆盖。

    返回值：
        包含所有匹配成功的 ``(start, end)`` 组成的 ``set`` 集合。
    """
    # 按照 token 索引大小进行升序排序，以便双指针向右滑动匹配
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}  # 用于存储终点到起点的最优匹配映射表

    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        # 1. 起终点重合（单 Token 实体，如单个中文字组成的实体）
        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue

        # 2. 起点在终点左侧（可能是一个潜在的多 Token 实体区间）
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue

        # 3. 如果起点已经越过了终点 (start_id > end_id)，说明当前的终点无意义，终点指针右移寻找下一个终点
        end_pointer += 1

    return {(couple_dict[end], end) for end in couple_dict}


def get_id_and_prob(span_set, offset_mapping):
    """将 Token 级别的 span 区间转换、映射回原始文本的【字符级别】 ``(char_start, char_end)`` 偏移量。

    ⚠️ 难点解析（偏移量纠偏）：
        UIE 的输入序列是由 `[CLS] + Prompt + [SEP] + Text + [SEP]` 拼接而成的。
        然而，Tokenizer 返回的 `offset_mapping` 默认是以整个拼接序列为基准的。
        为了准确拿到实体在 **原始 Text** 中的真实字符索引，本函数通过定位第二个特殊 Token [SEP]，
        精确计算出 Prompt 部分占用的长度偏差（bias），并从最终结果中将这部分偏移量减去（bias 纠偏）。
    """
    offset_mapping = [list(x) for x in offset_mapping]

    # 定位第一个 [SEP] 的位置，它标志着 Prompt 部分的结束（即在拼接序列中偏移映射重新归零 [0, 0] 的地方）
    prompt_end_token_id = offset_mapping[1:].index([0, 0])
    # 计算偏差值：即 Prompt 结束位置在拼接字符流中实际对应的物理偏移量
    bias = offset_mapping[prompt_end_token_id][1] + 1

    # 纠偏：将所有落在 Text 区间内的 token offsets，统一减去偏差值 bias，从而还原出相对于原 Text 的索引
    for idx in range(1, prompt_end_token_id + 1):
        offset_mapping[idx][0] -= bias
        offset_mapping[idx][1] -= bias

    sentence_id = []
    prob = []
    # 遍历每个配对好的 Span
    for start, end in span_set:
        # 综合概率 = 起点预测概率 * 终点预测概率
        prob.append(start[1] * end[1])
        start_id = offset_mapping[start[0]][0]
        end_id = offset_mapping[end[0]][1]
        sentence_id.append((start_id, end_id))
    return sentence_id, prob


def dbc2sbc(text: str) -> str:
    """将文本中的全角字符（DBCS）转换为半角字符（SBCS）。

    这是中文 NLP 中常见的数据清洗步骤，可以使标点符号和英文字符在词表中更容易被正确匹配。
    """
    rs = ""
    for char in text:
        code = ord(char)
        if code == 0x3000:  # 处理全角空格
            code = 0x0020
        else:
            code -= 0xFEE0  # 全半角编码偏移量

        # 限制只转换标准的 ASCII 可见字符范围
        if not (0x0021 <= code <= 0x7E):
            rs += char
            continue
        rs += chr(code)
    return rs


# ---------------------------------------------------------------------------
# 长文本自动切分器 (Text auto-splitter) – 优雅解决文本长度超出 ``max_predict_len`` 限制的问题
# ---------------------------------------------------------------------------
def auto_splitter(input_texts: List[str], max_predict_len: int, split_sentence: bool = False):
    """自动切分超长文本，确保切分后的每一个子句片段都能容纳在模型的最大预测窗口内。

    参数：
        input_texts: 原始输入待抽取的文本列表。
        max_predict_len: 模型单次前向传播能接受的最大文本长度。
        split_sentence: 是否开启智能分句。
            - 若为 True，会尝试在中文句号、惊叹号等天然标点处切分，防止截断实体。
            - 若为 False，则进行粗暴的固定长度硬切分。

    返回值：
        ``(short_texts, input_mapping)`` 的元组。
        - ``short_texts``: 所有的子句/切分后的短文本列表。
        - ``input_mapping``: 映射字典，键为原始文本的索引，值为一个 int 列表（代表该原始文本被拆分成了 short_texts 中的哪几段）。
          在推理结束后，可以通过此映射将零碎的子句抽取结果合并回原始输入。
    """
    short_input_texts = []
    input_mapping = {}
    cnt = 0
    for text in input_texts:
        # 情况 1：文本长度未超限，无需切分，直接保留
        if len(text) <= max_predict_len:
            short_input_texts.append(text)
            input_mapping[cnt] = [cnt]
            cnt += 1

        # 情况 2：文本超长，且开启了智能分句评估
        elif split_sentence:
            import re
            # 正则匹配：利用中文或英文的常见结束标点 [。！？!?\n] 进行智能切分
            sentences = re.split(r"(?<=[。！？!?\n])", text)
            sentences = [s for s in sentences if s]
            temp_text = ""
            temp_list = []
            for s in sentences:
                # 尽量把多个短句拼接在一起，直到接近 max_predict_len 阈值
                if len(temp_text) + len(s) <= max_predict_len:
                    temp_text += s
                else:
                    if temp_text:
                        temp_list.append(temp_text)
                    temp_text = s
            if temp_text:
                temp_list.append(temp_text)

            # 记录分段映射，以便后续合并结果
            input_mapping[cnt] = [len(short_input_texts) + i for i in range(len(temp_list))]
            short_input_texts.extend(temp_list)
            cnt += 1

        # 情况 3：文本超长，采用硬切分（按固定步长直接截断）
        else:
            parts = [text[i:i + max_predict_len] for i in range(0, len(text), max_predict_len)]
            input_mapping[cnt] = [len(short_input_texts) + i for i in range(len(parts))]
            short_input_texts.extend(parts)
            cnt += 1

    return short_input_texts, input_mapping
