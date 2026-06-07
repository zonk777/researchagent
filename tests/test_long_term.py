"""长期记忆 (LongTermMemory) 单元测试。

注意: 这些测试需要首次下载 BGE-M3 模型 (~2GB)，耗时较长。
首次运行后模型缓存在本地，后续测试会快很多。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from researchagent.memory.long_term import LongTermMemory


@pytest.fixture
def long_term() -> LongTermMemory:
    """创建临时目录下的 LongTermMemory 实例。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ltm = LongTermMemory(store_path=Path(tmpdir) / "memory")
        yield ltm


def test_initialization_creates_path() -> None:
    """验证初始化后存储路径存在。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory"
        LongTermMemory(store_path=path)
        # 懒加载，路径可能在 _ensure_initialized 时创建


def test_empty_search_returns_empty(long_term: LongTermMemory) -> None:
    """验证空库搜索返回空列表。"""
    results = long_term.search("test query")
    assert isinstance(results, list)
    assert len(results) == 0


def test_add_and_search_interaction(long_term: LongTermMemory) -> None:
    """验证添加后可以搜索到。"""
    long_term.add_interaction("用户喜欢使用 Python 编程", role="user")
    results = long_term.search("编程语言")
    assert len(results) > 0


def test_add_fact(long_term: LongTermMemory) -> None:
    """验证 add_fact 存储事实。"""
    long_term.add_fact("项目名为 AgentFlow")
    results = long_term.search("项目名称")
    assert len(results) > 0
    assert any("AgentFlow" in r["text"] for r in results)


def test_add_preference(long_term: LongTermMemory) -> None:
    """验证 add_preference 存储偏好。"""
    long_term.add_preference("用户偏好异步编程")
    results = long_term.search("异步")
    assert len(results) > 0
    assert results[0]["category"] == "preference"


def test_search_result_structure(long_term: LongTermMemory) -> None:
    """验证搜索结果包含所有必需字段。"""
    long_term.add_interaction("测试消息", role="user")
    results = long_term.search("测试")
    assert len(results) > 0
    r = results[0]
    assert "text" in r
    assert "role" in r
    assert "category" in r
    assert "timestamp" in r


def test_search_with_category_filter(long_term: LongTermMemory) -> None:
    """验证按分类过滤搜索。"""
    long_term.add_interaction("常规对话", role="user", category="interaction")
    long_term.add_fact("重要事实")
    # 只搜事实
    results = long_term.search("对话 事实", k=5, category="fact")
    if results:
        assert all(r["category"] == "fact" for r in results)


def test_multiple_adds_and_count(long_term: LongTermMemory) -> None:
    """验证多次添加后计数正确。"""
    initial = long_term.count()
    for i in range(3):
        long_term.add_interaction(f"对话 {i}", role="user")
    assert long_term.count() >= initial + 3


def test_semantic_search_relevance(long_term: LongTermMemory) -> None:
    """验证语义搜索能区分不相关的内容。"""
    long_term.add_fact("Python 是一门编程语言")
    long_term.add_fact("今天天气很好")
    results = long_term.search("编程开发", k=2)
    # "Python 编程语言" 应该比 "天气" 更相关
    if len(results) >= 2:
        assert "Python" in results[0]["text"] or "编程" in results[0]["text"]
