"""Semantic Scholar 学术论文搜索工具。

使用 semanticscholar Python 库搜索已发表论文。
含被引数、参考文献、会议/期刊信息。
免费层需 API Key（非必须，但无 API Key 时速率限制更严格）。
"""

from __future__ import annotations

from typing import Any


def search_semantic_scholar(
    query: str,
    max_results: int | str = 5,
    fields: str = "title,authors,abstract,year,venue,citationCount,referenceCount,externalIds,publicationTypes,url",
) -> dict[str, Any]:
    """搜索 Semantic Scholar 上的学术论文。

    覆盖范围比 ArXiv 更广（含会议论文、期刊）。

    Args:
        query: 搜索关键词（英文）。
        max_results: 最多返回论文数（默认 5，最大 20）。
        fields: 要返回的字段（逗号分隔）。默认返回标题、作者、摘要、年份、
                会议/期刊、被引数、引用数、DOI、论文类型、URL。

    Returns:
        {"ok": True, "query": str, "papers": [...], "total_results": int}
        每篇论文包含请求的 fields。
    """
    try:
        max_r = int(max_results)
    except (ValueError, TypeError):
        max_r = 5
    max_r = max(1, min(max_r, 20))

    try:
        from semanticscholar import SemanticScholar
        sch = SemanticScholar()
        results = sch.search_paper(query, limit=max_r, fields_of_study=fields.split(","))

        papers = []
        for r in results:
            paper_data = {}
            # 安全提取每个字段
            for field_name in ["title", "abstract", "year", "venue", "citationCount",
                               "referenceCount", "url", "publicationTypes"]:
                val = getattr(r, field_name, None)
                if val is not None:
                    paper_data[field_name] = val

            # 作者
            if hasattr(r, "authors") and r.authors:
                paper_data["authors"] = [a.get("name", "") if isinstance(a, dict) else str(a) for a in r.authors]

            # 外部 ID
            if hasattr(r, "externalIds") and r.externalIds:
                paper_data["externalIds"] = r.externalIds

            # 摘要截断
            abstract = paper_data.get("abstract") or ""
            if isinstance(abstract, str) and len(abstract) > 800:
                paper_data["abstract"] = abstract[:800] + "..."

            papers.append(paper_data)

        return {
            "ok": True,
            "query": query,
            "papers": papers,
            "total_results": len(papers),
        }

    except Exception as e:
        return {"ok": False, "query": query, "error": f"Semantic Scholar 搜索失败: {e}"}


def get_paper_details(paper_id: str) -> dict[str, Any]:
    """获取单篇论文的详细信息（含参考文献列表）。

    Args:
        paper_id: Semantic Scholar paper ID 或 DOI 或 ArXiv ID。

    Returns:
        {"ok": True, "paper": {...}} 或 {"ok": False, "error": str}
    """
    try:
        from semanticscholar import SemanticScholar
        sch = SemanticScholar()
        paper = sch.get_paper(paper_id)

        citations = []
        if hasattr(paper, "citations") and paper.citations:
            for c in paper.citations[:10]:
                citing = c.get("citingPaper", {}) if isinstance(c, dict) else getattr(c, "citingPaper", None)
                if citing:
                    citations.append({
                        "title": citing.get("title", "") if isinstance(citing, dict) else getattr(citing, "title", ""),
                        "year": citing.get("year") if isinstance(citing, dict) else getattr(citing, "year", None),
                    })

        references = []
        if hasattr(paper, "references") and paper.references:
            for ref in paper.references[:10]:
                cited = ref.get("citedPaper", {}) if isinstance(ref, dict) else getattr(ref, "citedPaper", None)
                if cited:
                    references.append({
                        "title": cited.get("title", "") if isinstance(cited, dict) else getattr(cited, "title", ""),
                        "year": cited.get("year") if isinstance(cited, dict) else getattr(cited, "year", None),
                    })

        return {
            "ok": True,
            "paper": {
                "title": getattr(paper, "title", ""),
                "abstract": getattr(paper, "abstract", ""),
                "year": getattr(paper, "year", None),
                "citationCount": getattr(paper, "citationCount", 0),
                "referenceCount": getattr(paper, "referenceCount", 0),
                "citations": citations,
                "references": references,
            },
        }

    except Exception as e:
        return {"ok": False, "error": f"获取论文详情失败: {e}"}
