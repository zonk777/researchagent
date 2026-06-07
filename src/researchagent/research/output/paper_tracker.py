"""新论文追踪。

比较两次搜索的结果，发现新增论文。
"""

from __future__ import annotations

import json
from pathlib import Path


def save_search_state(papers: list[dict], state_path: Path) -> None:
    """保存当前搜索结果作为基线。

    Args:
        papers: 论文列表。
        state_path: 状态文件路径 (.json)。
    """
    state = {
        "titles": [p.get("title", "") for p in papers],
        "urls": [p.get("url", "") for p in papers],
        "count": len(papers),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def find_new_papers(papers: list[dict], state_path: Path) -> list[dict]:
    """对比基线状态，找出新论文。

    Args:
        papers: 当前论文列表。
        state_path: 之前保存的状态文件。

    Returns:
        新增的论文列表。
    """
    if not state_path.exists():
        return []

    state = json.loads(state_path.read_text(encoding="utf-8"))
    known_titles = set(state.get("titles", []))
    known_urls = set(state.get("urls", []))

    new_papers = []
    for p in papers:
        title = p.get("title", "")
        url = p.get("url", "")
        if title not in known_titles and url not in known_urls:
            new_papers.append(p)

    return new_papers


def format_update_report(new_papers: list[dict]) -> str:
    """生成增量更新报告。"""
    if not new_papers:
        return "无新增论文。"

    lines = [f"## 新增论文 ({len(new_papers)} 篇)", ""]
    for i, p in enumerate(new_papers, 1):
        title = p.get("title", "")
        year = str(p.get("year", ""))
        url = p.get("url", "")
        lines.append(f"{i}. **{title}** ({year})")
        if url:
            lines.append(f"   {url}")
        lines.append("")
    return "\n".join(lines)
