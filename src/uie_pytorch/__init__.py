"""UIE-PyTorch: A faithful PyTorch port of PaddleNLP's UIE model.

Zero-shot / few-shot universal information extraction without PaddlePaddle.

Quick start::

    from uie_pytorch import UIEExtractor

    ie = UIEExtractor(model="weights/uie-base", schema=["时间", "选手", "赛事名称"])
    result = ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")
    print(result)
"""

from .model import UIE
from .extractor import UIEExtractor
from .utils import SchemaTree, build_schema_tree
from .converter import convert_model

__all__ = ["UIE", "UIEExtractor", "SchemaTree", "build_schema_tree", "convert_model"]
__version__ = "1.0.0"
