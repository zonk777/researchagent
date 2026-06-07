"""引用关系分析。

分析论文之间的引用和被引关系，生成简单的引用网络。
"""

from __future__ import annotations

from collections import Counter


def build_citation_graph(papers: list[dict]) -> dict:
    """从论文列表构建引用关系图。

    Args:
        papers: 论文列表，每篇含 title, authors, year, citation_count。

    Returns:
        {
            "nodes": [{title, authors, year, citations}, ...],
            "edges": [],  # 需要 Semantic Scholar get_paper_details 获取
            "top_cited": [...],
            "total_citations": int,
            "avg_citations": float,
            "h_index_estimate": int,
        }
    """
    if not papers:
        return {"nodes": [], "edges": [], "top_cited": [], "total_citations": 0, "avg_citations": 0.0, "h_index_estimate": 0}

    nodes = []
    total_citations = 0
    for p in papers:
        citations = int(p.get("citation_count", p.get("citationCount", 0)))
        total_citations += citations
        nodes.append({
            "title": p.get("title", ""),
            "authors": p.get("authors", ""),
            "year": str(p.get("year", "")),
            "citations": citations,
        })

    # 按被引数排序
    nodes.sort(key=lambda n: n["citations"], reverse=True)
    avg = total_citations / len(nodes) if nodes else 0.0

    # 估算 H-index: 至少有 h 篇论文的被引数 >= h
    h_index = 0
    for i, n in enumerate(nodes):
        if n["citations"] >= i + 1:
            h_index = i + 1
        else:
            break

    return {
        "nodes": nodes,
        "edges": [],  # 引用边需要逐篇查 Semantic Scholar
        "top_cited": nodes[:5],
        "total_citations": total_citations,
        "avg_citations": round(avg, 1),
        "h_index_estimate": h_index,
    }


def format_citation_report(graph: dict) -> str:
    """生成引用分析报告（文本格式）。"""
    if not graph["nodes"]:
        return "无引用数据。"
    lines = [
        "## 引用分析",
        "",
        f"- 总被引数: {graph['total_citations']}",
        f"- 平均被引数: {graph['avg_citations']}",
        f"- 估算 H-index: {graph['h_index_estimate']}",
        "",
        "### 高被引论文",
        "",
    ]
    for i, p in enumerate(graph["top_cited"], 1):
        authors = p["authors"]
        if isinstance(authors, str) and len(authors) > 60:
            authors = authors[:60] + "..."
        lines.append(f"{i}. **{p['title']}** ({p['year']}) — {p['citations']} 次引用")
        lines.append(f"   {authors}")
        lines.append("")
    return "\n".join(lines)
