"""MemoryManager 单元测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from researchagent.core.state import RuntimeState
from researchagent.memory.manager import MemoryManager


class FakeLLM:
    """模拟 LLM。"""

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="[摘要] 用户进行了对话。")


@pytest.fixture
def state() -> RuntimeState:
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        yield RuntimeState(workspace=workspace)


@pytest.fixture
def manager(state: RuntimeState) -> MemoryManager:
    return MemoryManager(FakeLLM(), state)


def test_manager_initialization(manager: MemoryManager) -> None:
    """验证管理器正确初始化。"""
    assert manager.short_term is not None
    assert manager.long_term is not None


def test_add_interaction_short_term(manager: MemoryManager) -> None:
    """验证 add_interaction 同时写入短期记忆。"""
    manager.add_interaction("user", "你好")
    messages = manager.get_recent_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "你好"


def test_add_interaction_long_term(manager: MemoryManager) -> None:
    """验证 add_interaction 同时写入长期记忆。"""
    manager.add_interaction("user", "唯一的搜索词")
    results = manager.search_memories("唯一的搜索词")
    assert len(results) > 0


def test_build_context_includes_short_term(manager: MemoryManager) -> None:
    """验证 build_context 包含短期记忆。"""
    manager.add_interaction("user", "Python 是什么？")
    manager.add_interaction("assistant", "Python 是一门编程语言。")
    ctx = manager.build_context()
    assert "Python" in ctx
    assert "最近对话" in ctx


def test_build_context_includes_long_term(manager: MemoryManager) -> None:
    """验证 build_context 包含相关长期记忆。"""
    manager.remember_preference("用户偏好 Python 语言")
    ctx = manager.build_context("编程语言")
    assert "相关历史记忆" in ctx
    assert "Python" in ctx


def test_remember_fact(manager: MemoryManager) -> None:
    """验证 remember_fact 存储事实。"""
    manager.remember_fact("项目使用 LanceDB 作为向量库")
    results = manager.search_memories("向量数据库")
    assert len(results) > 0


def test_remember_preference(manager: MemoryManager) -> None:
    """验证 remember_preference 存储偏好。"""
    manager.remember_preference("偏好异步编程")
    results = manager.search_memories("异步")
    assert len(results) > 0


def test_get_short_term_summary(manager: MemoryManager) -> None:
    """验证获取短期摘要。"""
    # 在低阈值下添加多条消息触发摘要
    manager.short_term.max_tokens = 10
    manager.short_term.keep_recent = 1
    for i in range(8):
        manager.add_interaction("user", "x" * 100)
    summary = manager.get_short_term_summary()
    assert summary is not None


def test_memory_count(manager: MemoryManager) -> None:
    """验证记忆计数。"""
    initial = manager.memory_count()
    manager.add_interaction("user", "测试记忆")
    assert manager.memory_count() >= initial + 1


def test_clear_short_term(manager: MemoryManager) -> None:
    """验证清空短期记忆不影响长期记忆。"""
    manager.add_interaction("user", "测试短记忆")
    long_count_before = manager.memory_count()
    manager.clear_short_term()
    assert len(manager.get_recent_messages()) == 0
    assert manager.memory_count() >= long_count_before  # 长期记忆保留
