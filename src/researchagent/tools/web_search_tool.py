"""Tavily 网络搜索工具 (Web Search Tool)。

为 Agent 提供联网搜索能力，使用 Tavily API 返回结构化的搜索结果
（包含标题、URL、摘要、相关性评分）以及可选的 AI 摘要。

配置:
    在 .env 中设置 TAVILY_API_KEY，从 https://tavily.com 获取。

返回格式：
    {
        "ok": True,
        "query": str,
        "answer": str,        # Tavily AI 摘要
        "results": [
            {"title": str, "url": str, "content": str, "score": float},
            ...
        ]
    }
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# 模块级别加载一次，避免每次搜索都 I/O
load_dotenv()


def web_search(
    query: str,
    max_results: int | str = 5,
    include_answer: bool | str = True,
) -> dict[str, Any]:
    """使用 Tavily 搜索网络并返回结构化结果。

    Args:
        query: 搜索查询字符串。
        max_results: 最多返回的结果数量 (默认 5, 最大 10)。
        include_answer: 是否包含 Tavily AI 摘要 (默认 True)。

    Returns:
        搜索结果字典:
        {
            "ok": bool,
            "query": str,
            "answer": str | None,     # AI 摘要
            "results": [
                {"title": str, "url": str, "content": str, "score": float | None},
                ...
            ]
        }
        失败时返回 {"ok": False, "error": str}。
    """

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "缺少必需的 .env 配置项: TAVILY_API_KEY。请从 https://tavily.com 获取 API Key。",
        }

    # 参数规范化
    try:
        max_results_int = int(max_results)
    except (ValueError, TypeError):
        max_results_int = 5
    max_results_int = max(1, min(max_results_int, 10))

    if isinstance(include_answer, str):
        include_answer_bool = include_answer.strip().lower() not in {"false", "0", "no", "off"}
    else:
        include_answer_bool = bool(include_answer)

    # 调用 Tavily
    try:
        from tavily import TavilyClient
    except ImportError:
        return {
            "ok": False,
            "error": "tavily-python 未安装。请运行: uv sync",
        }

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results_int,
            include_answer=include_answer_bool,
        )

        results = []
        for item in response.get("results", []):
            content = item.get("content", "")
            if isinstance(content, str) and len(content) > 1200:
                content = content[:1200] + "..."
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": content,
                "score": item.get("score"),
            })

        return {
            "ok": True,
            "query": query,
            "answer": response.get("answer"),
            "results": results,
        }

    except Exception as exc:
        return {
            "ok": False,
            "query": query,
            "error": f"Tavily 搜索失败: {exc}",
        }
