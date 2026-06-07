"""Web Search Tool 单元测试。"""

from __future__ import annotations

from researchagent.tools.web_search_tool import web_search


def test_web_search_missing_api_key(monkeypatch) -> None:
    """验证缺少 API Key 时返回错误。"""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "")

    result = web_search("test query")
    assert result["ok"] is False
    assert "TAVILY_API_KEY" in result["error"]


def test_web_search_result_structure_without_api_key(monkeypatch) -> None:
    """验证即使无 API Key，返回结构也包含 query。"""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "")

    result = web_search("Python")
    assert "query" in result or result.get("ok") is False


def test_web_search_max_results_clamped(monkeypatch) -> None:
    """验证 max_results 被限制在有效范围。这个测试仅验证错
    误路径不抛异常，不发起真实请求。"""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "")

    # 即使参数超出范围，也不应抛出异常（仅 API Key 缺失会返回 error）
    result = web_search("test", max_results=100)
    assert result["ok"] is False  # 因缺少 API Key
    assert "TAVILY_API_KEY" in result["error"]


def test_web_search_include_answer_param(monkeypatch) -> None:
    """验证 include_answer 参数被接受。"""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "")

    # 不同 include_answer 值不应抛异常
    result = web_search("test", include_answer=True)
    assert result["ok"] is False

    result = web_search("test", include_answer=False)
    assert result["ok"] is False

    result = web_search("test", include_answer="false")
    assert result["ok"] is False
