"""ArXiv 学术论文搜索工具。

使用 arxiv Python 库搜索预印本论文。
免费、无需 API Key、返回结构化论文信息。
"""

from __future__ import annotations

from typing import Any

import arxiv


def search_arxiv(
    query: str,
    max_results: int | str = 5,
    sort_by: str = "relevance",
) -> dict[str, Any]:
    """搜索 ArXiv 上的学术论文。

    Args:
        query: 搜索关键词（英文），支持布尔运算符。
              例如: "transformer AND efficiency AND (edge OR mobile)"
        max_results: 最多返回论文数（默认 5，最大 20）。
        sort_by: 排序方式，可选 "relevance"（相关度）、"lastUpdatedDate"（最近更新）、
                "submittedDate"（提交日期）。

    Returns:
        {"ok": True, "query": str, "papers": [...], "total_results": int}
        每篇论文包含:
            - title: 标题
            - authors: 作者列表
            - summary: 摘要（已截断至 800 字符）
            - published: 发布日期
            - url: ArXiv 页面 URL
            - pdf_url: 直接 PDF 链接
            - primary_category: 主分类（如 cs.AI）
    """
    try:
        max_r = int(max_results)
    except (ValueError, TypeError):
        max_r = 5
    max_r = max(1, min(max_r, 20))

    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }
    sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_r,
            sort_by=sort_criterion,
        )
        results = list(client.results(search))

        papers = []
        for r in results:
            summary = r.summary.replace("\n", " ")
            if len(summary) > 800:
                summary = summary[:800] + "..."

            papers.append({
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "summary": summary,
                "published": r.published.isoformat() if r.published else "",
                "url": r.entry_id,
                "pdf_url": r.pdf_url,
                "primary_category": r.primary_category,
                "categories": list(r.categories),
            })

        return {
            "ok": True,
            "query": query,
            "papers": papers,
            "total_results": len(papers),
        }

    except Exception as e:
        return {"ok": False, "query": query, "error": f"ArXiv 搜索失败: {e}"}
