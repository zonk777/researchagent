"""方法对比提取。

从论文摘要中提取方法名称、数据集和性能指标，生成对比表。
"""

from __future__ import annotations

import re
from collections import Counter


# 常见 ML 方法名称模式
METHOD_PATTERNS = [
    r"(Transformer|CNN|RNN|LSTM|GAN|BERT|GPT|ResNet|ViT|Diffusion|VAE)",
    r"(attention|self-attention|cross-attention|multi-head)",
    r"(fine-tuning|pre-training|transfer learning|distillation|pruning|quantization)",
    r"(reinforcement learning|supervised|unsupervised|semi-supervised|self-supervised)",
    r"(SGD|Adam|AdamW|LAMB|LARS)",
    r"(knowledge distillation|model compression|sparse|mixture.of.experts)",
]


def extract_methods(papers: list[dict]) -> dict:
    """从论文摘要中提取方法和技术。

    Args:
        papers: 论文列表。

    Returns:
        {"methods": {method_name: count}, "comparison_table": [{...}]}
    """
    method_counter: Counter = Counter()

    for p in papers:
        text = f"{p.get('title', '')} {p.get('summary', p.get('abstract', ''))}"
        text_lower = text.lower()

        for pattern in METHOD_PATTERNS:
            matches = re.findall(pattern, text_lower)
            for m in matches:
                method_counter[m.strip()] += 1

    return {
        "methods": dict(method_counter.most_common(15)),
        "total_papers": len(papers),
    }


def format_method_report(analysis: dict) -> str:
    """生成方法分析报告。"""
    if not analysis.get("methods"):
        return "未检测到方法模式。"

    lines = ["## 方法趋势", ""]
    lines.append(f"基于 {analysis['total_papers']} 篇论文的方法分布：")
    lines.append("")
    for method, count in analysis["methods"].items():
        bar = "█" * count
        lines.append(f"- **{method}**: {count} 篇 {bar}")
    return "\n".join(lines)
