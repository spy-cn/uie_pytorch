"""关系抽取（三元组/嵌套抽取）示例脚本。

运行方法:
    python examples/relation_extraction.py

本脚本演示了如何利用 UIE 模型的嵌套 Schema 树结构（SchemaTree）实现中文关系抽取。
UIE 底层会采用「多阶段迭代抽取」的逻辑：
  - 第一步：抽取外层的实体（如：公司）。
  - 第二步：将第一步抽出的实体与子节点 Schema 拼接，生成新 Prompt（如：“华为的创始人”），进而抽取关系属性。
"""

from __future__ import annotations

import json
import os
import sys

# 技巧：将项目根目录下的 src 添加到 Python 寻址路径中，
# 确保在未安装整个包时，脚本也能在 examples/ 目录下顺利跑通。
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uie_pytorch import UIEExtractor


def main():
    # 动态定位 PyTorch 版的 uie-base 权重目录
    model_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "uie-base")
    model_dir = os.path.abspath(model_dir)

    # 实例化 UIE 抽取器（自动选择 CUDA/MPS/CPU 硬件设备加速）
    ie = UIEExtractor(model=model_dir, device="auto")

    # ---- 关系抽取示例 1: 歌曲实体与它的关联属性 ---------------------------------
    print("=" * 70)
    print("关系抽取示例 1: 歌曲 → 歌手 + 专辑")
    print("=" * 70)

    #  结构解析：
    # 第一阶段提取：“歌曲名称”
    # 第二阶段提取：“<歌曲名>的歌手”、“<歌曲名>的所属专辑”
    ie.set_schema([{"歌曲名称": ["歌手", "所属专辑"]}])
    text = "《告别了》是孙耀威在专辑爱的故事里面的歌曲"

    print(f"\n待抽取文本: {text}")
    result = ie(text)
    print(f"抽取结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # ---- 关系抽取示例 2: 人物实体与它的关联属性 ---------------------------------
    print("\n" + "=" * 70)
    print("关系抽取示例 2: 人物 → 出生地 + 职业")
    print("=" * 70)

    # 结构解析：
    # 第一阶段提取：“人物”
    # 第二阶段提取：“<人物姓名>的出生地”、“<人物姓名>的职业”
    ie.set_schema([{"人物": ["出生地", "职业"]}])
    text = "姚明1980年出生于上海，前中国职业篮球运动员，司职中锋"

    print(f"\n待抽取文本: {text}")
    result = ie(text)
    print(f"抽取结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # ---- 关系抽取示例 3: 公司实体与它的关联属性 ---------------------------------
    print("\n" + "=" * 70)
    print("关系抽取示例 3: 公司 → 创始人 + 总部")
    print("=" * 70)

    # 结构解析：
    # 第一阶段提取：“公司”
    # 第二阶段提取：“<公司名>的创始人”、“<公司名>的总部地点”
    ie.set_schema([{"公司": ["创始人", "总部地点"]}])
    text = "华为由任正非于1987年在深圳创立，总部位于广东省深圳市龙岗区"

    print(f"\n待抽取文本: {text}")
    result = ie(text)
    print(f"抽取结果: {json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
