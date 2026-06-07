"""运行完整分析 pipeline。

结合引用关系、方法对比和趋势分析，生成综合报告。
"""

from __future__ import annotations

from researchagent.research.analysis.citation_graph import (
    build_citation_graph,
    format_citation_report,
)
from researchagent.research.analysis.method_comparison import (
    extract_methods,
    format_method_report,
)
from researchagent.research.analysis.trend_analyzer import (
    analyze_keywords,
    analyze_year_distribution,
    format_trend_report,
)


def run_full_analysis(papers: list[dict]) -> str:
    """对论文列表运行完整分析，返回 Markdown 格式报告。

    Args:
        papers: 论文 dict 列表。

    Returns:
        完整的分析报告 Markdown 字符串。
    """
    if not papers:
        return "无论文数据，无法进行分析。"

    citation_graph = build_citation_graph(papers)
    methods = extract_methods(papers)
    year_dist = analyze_year_distribution(papers)
    keywords = analyze_keywords(papers)

    sections = [
        f"# 学术分析报告\n\n分析 {len(papers)} 篇论文\n",
        format_citation_report(citation_graph),
        format_method_report(methods),
        format_trend_report(year_dist, keywords),
    ]

    return "\n\n".join(sections)
