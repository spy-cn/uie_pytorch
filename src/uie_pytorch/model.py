"""基于 PyTorch 实现的 UIE (Universal Information Extraction) 模型。

本模块是 PaddleNLP UIE 模型的 1:1 像素级 PyTorch 重构版本。
所有子模块、成员变量的命名完全镜像了 PaddleNLP 的原始命名（例如使用 `layer_norm` 替代 HuggingFace 的 `LayerNorm` 等）。
因此，在转换和加载 Paddle 权重时，可以实现 **零冲突（Zero Key Remapping）** 的直接读取。

参考资料：
    - PaddleNLP 官方仓库: https://github.com/PaddlePaddle/PaddleNLP
    - 原始论文: Lu et al., "Unified Structure Generation for Universal Information Extraction", ACL 2022.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig


# ---------------------------------------------------------------------------
# 1. 嵌入层 (Embeddings) – 包含 Word + Position + Token-Type + Task-Type
# ---------------------------------------------------------------------------
class ErnieEmbeddings(nn.Module):
    """ERNIE 的混合嵌入层。

    它在传统 BERT 的基础上，额外融合了特定任务的 **Task-Type (任务类型) Embeddings**（对多任务微调非常重要）。

    ⚠️ 避坑对齐细节：
        HuggingFace 命名为 `LayerNorm`，而 PaddleNLP 命名为 `layer_norm` (全小写)。
        此处必须使用 ``self_attn.layer_norm`` 的全小写命名，以便无缝载入权重。
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        # 基础字/词向量嵌入
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 绝对位置向量嵌入
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 句段标志向量嵌入（用以区分 Prompt 和 Text）
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # ERNIE 特有的 Task 标签嵌入（默认任务词表大小 task_type_vocab_size = 3）
        task_type_vocab_size = getattr(config, "task_type_vocab_size", 3)
        self.task_type_embeddings = nn.Embedding(task_type_vocab_size, config.hidden_size)

        # 使用 Paddle 命名风格的小写 `layer_norm`
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册不需要梯度的 position_ids 缓存区，避免训练时多余的更新
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if task_ids is None:
            task_ids = torch.zeros_like(input_ids)

        # 四种特征向量相加融合成最终的 Embedding
        embeddings = (
                self.word_embeddings(input_ids)
                + self.position_embeddings(position_ids)
                + self.token_type_embeddings(token_type_ids)
                + self.task_type_embeddings(task_ids)
        )
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# ---------------------------------------------------------------------------
# 2. 多头自注意力层 (Multi-Head Attention)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """标准的多头自注意力层，行为与命名与 PaddleNLP 的 ``MultiHeadAttention`` 保持完全对齐。

    该模块直接将 ``q_proj``, ``k_proj``, ``v_proj`` 和 ``out_proj`` 声明为 Module 的直属子属性。
    权重键名将精确匹配 ``self_attn.q_proj.weight`` 格式。

    注意力机制数学公式表示如下：
    $$Attention(Q, K, V) = Softmax\left(\frac{Q K^T}{\sqrt{d_k}} + M\right)V$$
    其中 $d_k$ 为每个注意力头的维度大小（``head_dim``），$M$ 为掩码（``attention_mask``）。
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """拆分序列的多头视角：
        输入形状: (batch, seq, hidden_size)
        输出形状: (batch, heads, seq, head_dim)
        """
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        return x.view(new_shape).permute(0, 2, 1, 3)

    def forward(self, query: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(query))
        v = self._split_heads(self.v_proj(query))

        # 计算缩放点积注意力分值
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # 融入 Additive Attention Mask 掩码阻断 [PAD] 部分
        scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # 加权求和还原并合并注意力头
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.embed_dim,))
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# 3. Transformer 编码层 (TransformerEncoderLayer)
# ---------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    """单个 Transformer 编码器层。

    采用 PaddleNLP 的标准层级命名结构：
      * ``self_attn`` – 自注意力核心（下辖 q/k/v/out_proj）
      * ``norm1`` – 交互反馈后的第一个 LayerNorm 层
      * ``linear1`` / ``linear2`` – 前馈神经网络 (FFN)
      * ``norm2`` – FFN 反馈后的第二个 LayerNorm 层

    💡 注意：ERNIE 使用的是标准的 **Post-LayerNorm** 架构，数学流向为：
    $$h = LayerNorm(x + Dropout(SelfAttention(x)))$$
    $$out = LayerNorm(h + Dropout(FFN(h)))$$
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 1. 自注意力计算与残差连接 Post-LN
        attn_out = self.self_attn(hidden_states, attention_mask)
        attn_out = self.dropout(attn_out)
        h = self.norm1(hidden_states + attn_out)

        # 2. 前馈神经网络 FFN 与残差连接 Post-LN
        ffn_out = self.activation(self.linear1(h))
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.dropout(ffn_out)
        out = self.norm2(h + ffn_out)
        return out


class TransformerEncoder(nn.Module):
    """Transformer 编码器，由 N 层 ``TransformerEncoderLayer`` 堆叠而成。"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class ErniePooler(nn.Module):
    """提取 [CLS] 位置的特征表示，并经 Dense 层和 Tanh 激活后输出，代表整句句向量。"""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 抽取第 0 位（即 [CLS] 位置的 token 特征）
        return self.activation(self.dense(hidden_states[:, 0]))


# ---------------------------------------------------------------------------
# 4. ERNIE 骨干网络 (ErnieModel)
# ---------------------------------------------------------------------------
class ErnieModel(nn.Module):
    """ERNIE 编码器模块，1:1 镜像对齐 PaddleNLP 中的 ``paddlenlp.transformers.ErnieModel``。

    所有的子模块和内部网络层次在编译和实例化后，都能无阻碍地匹配并载入 Paddle 原生的 ``.pdparams`` 权重。
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ErnieEmbeddings(config)
        self.encoder = TransformerEncoder(config)
        self.pooler = ErniePooler(config)

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 构建叠加式的 Attention Mask 矩阵 (将 1 保持为 0，将需要 Mask 的 0 转为极其微小的极值 $-10^{38}$ 等)
        extended = attention_mask[:, None, None, :].to(dtype=self.embeddings.word_embeddings.weight.dtype)
        extended = (1.0 - extended) * torch.finfo(extended.dtype).min

        # 得到基础 Embedding 向量
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        # 送入 Encoder 进行特征提取
        sequence_output = self.encoder(embedding_output, extended)
        # 获取池化整句表征
        pooled_output = self.pooler(sequence_output)

        # 返回：(所有 token 级序列输出, 整句 pooler 输出)
        return sequence_output, pooled_output


# ---------------------------------------------------------------------------
# 5. UIE 最终预测模型
# ---------------------------------------------------------------------------
class UIE(nn.Module):
    """UIE 主体分类模型：ERNIE 骨干网 + 两个独立的线性分类头（指针网络）。

    通过在每个序列位置上进行独立的 sigmoid 计算，预测每个 token 作为起始或终结位置的联合概率。

    输出结果为：
        (start_prob, end_prob) - 均为形状为 ``(batch, seq_len)`` 的概率张量。
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.ernie = ErnieModel(config)
        # 用于定位起点边界的分类头
        self.linear_start = nn.Linear(config.hidden_size, 1)
        # 用于定位终点边界的分类头
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            input_ids: torch.Tensor | None = None,
            token_type_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播计算。

        返回 (start_prob, end_prob) 指针向量。
        """
        # 1. 抽取骨干网络输出的序列表示 [Batch_Size, Seq_Len, Hidden_Size]
        sequence_output, _ = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # 2. 将隐藏层映射到一维，并通过 squeeze(-1) 挤压掉最后一维，随后计算 sigmoid 归一化概率
        start_prob = self.sigmoid(self.linear_start(sequence_output).squeeze(-1))
        end_prob = self.sigmoid(self.linear_end(sequence_output).squeeze(-1))

        return start_prob, end_prob
