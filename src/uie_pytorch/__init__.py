"""UIE-PyTorch: 针对 PaddleNLP UIE (Universal Information Extraction) 模型的高保真 PyTorch 移植版。

无需安装 PaddlePaddle 深度学习框架，即可在 PyTorch 生态下完美运行高质量的
零样本 (Zero-shot) / 少样本 (Few-shot) 通用信息抽取任务（实体、关系、事件三元组等）。

快速上手示例::

    from uie_pytorch import UIEExtractor

    # 1. 实例化抽取器并设定 Schema 抽取目标
    ie = UIEExtractor(model="weights/uie-base", schema=["时间", "选手", "赛事名称"])

    # 2. 执行抽取推理
    result = ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
    print(result)
"""

# ---- 1. 核心推理与后处理工具导入 ---------------------------------------------
from .model import UIE  # UIE PyTorch 核心模型定义（指针网络）
from .extractor import UIEExtractor  # 主入口推理 Pipeline（Taskflow 的直接替代品）
from .utils import SchemaTree, build_schema_tree  # 树状 Schema 构建与管理工具
from .converter import convert_model  # 官方 Paddle 权重 -> PyTorch 权重无损转换器

# ---- 2. 延迟导入微调训练工具（Lazy Import） ---------------------------------
# 说明：微调训练工具（如 Dataset、Trainer）依赖较多第三方库（如 PyTorch-Lightning、Transformers、Deepspeed 等）。
# 为了确保在「仅需要轻量级推理环境」的用户场景下不因为缺失训练依赖而报错，
# 采用 try-except 块进行优雅降级（Graceful Degradation）。
try:
    from .dataset import UIEDataset, UIEExample  # UIE 特殊格式的微调数据集封装与数据样例定义
    from .trainer import train_uie, uie_loss  # UIE 微调训练器核心入口方法与定制的指针损失函数（BCELoss）
except ImportError:  # pragma: no cover
    # 如果用户环境缺少微调依赖，依然可以正常导入 UIEExtractor 执行零样本推理
    pass

# ---- 3. 公开 API 声明 (Explicit Exports) -------------------------------------
# 显式声明该包对外暴露的公共 API 接口，支持 `from uie_pytorch import *` 时的规范导入
__all__ = [
    "UIE",
    "UIEExtractor",
    "SchemaTree",
    "build_schema_tree",
    "convert_model",
    "UIEDataset",
    "UIEExample",
    "train_uie",
    "uie_loss",
]

# ---- 4. 项目版本号 -----------------------------------------------------------
__version__ = "1.1.0"
