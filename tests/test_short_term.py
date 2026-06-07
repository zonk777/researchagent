"""短期记忆 (ConversationBuffer) 单元测试。"""

from __future__ import annotations

from researchagent.memory.short_term import ConversationBuffer


class FakeLLM:
    """模拟 LLM，返回固定摘要。"""

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="[Fake Summary] 用户询问了 Python 相关问题。")


def make_buffer(max_tokens: int = 200, keep_recent: int = 2) -> ConversationBuffer:
    return ConversationBuffer(
        llm=FakeLLM(),
        max_tokens=max_tokens,
        keep_recent=keep_recent,
    )


def test_empty_buffer_context() -> None:
    """验证空缓冲的上下文为空字符串。"""
    buf = make_buffer()
    assert buf.get_context() == ""


def test_add_single_message() -> None:
    """验证添加单条消息。"""
    buf = make_buffer()
    buf.add_message("user", "你好")
    assert len(buf.messages) == 1
    assert buf.messages[0]["role"] == "user"
    assert buf.messages[0]["content"] == "你好"


def test_get_context_with_messages() -> None:
    """验证上下文包含最近对话。"""
    buf = make_buffer()
    buf.add_message("user", "什么是 Python？")
    ctx = buf.get_context()
    assert "最近对话" in ctx
    assert "什么是 Python？" in ctx


def test_estimate_tokens() -> None:
    """验证 token 估计。"""
    buf = make_buffer()
    buf.add_message("user", "a" * 400)  # 400 chars → ~100 tokens
    buf.add_message("assistant", "b" * 400)  # 400 chars → ~100 tokens
    assert buf._estimate_tokens() >= 180  # 800/4 = 200, 允许一些误差


def test_summary_triggered_by_token_limit() -> None:
    """验证 token 超阈值时触发摘要。"""
    buf = make_buffer(max_tokens=10, keep_recent=1)  # 极低阈值确保触发
    # 添加大量长消息
    for i in range(5):
        buf.add_message("user", "x" * 100)
    # 应该只保留了 keep_recent 条消息 + 摘要
    assert len(buf.messages) <= 1  # keep_recent=1
    assert buf.summary is not None
    assert "[Fake Summary]" in buf.summary


def test_context_includes_summary() -> None:
    """验证上下文包含摘要。"""
    buf = make_buffer(max_tokens=10, keep_recent=1)
    for i in range(5):
        buf.add_message("user", "x" * 100)
    ctx = buf.get_context()
    assert "对话摘要" in ctx
    assert "[Fake Summary]" in ctx


def test_clear_removes_all() -> None:
    """验证 clear 清空所有消息和摘要。"""
    buf = make_buffer(max_tokens=10, keep_recent=1)
    for i in range(5):
        buf.add_message("user", "x" * 100)
    buf.clear()
    assert buf.summary is None
    assert len(buf.messages) == 0


def test_keep_recent_messages_after_summary() -> None:
    """验证摘要后保留最近消息。"""
    buf = make_buffer(max_tokens=10, keep_recent=3)
    # 添加足够多的消息触发摘要
    for i in range(10):
        buf.add_message("user", "x" * 100)
    # 摘要后应保留 keep_recent 条
    assert len(buf.messages) <= 3


def test_multiple_roles() -> None:
    """验证支持 user/assistant/system 不同角色。"""
    buf = make_buffer()
    buf.add_message("system", "系统提示")
    buf.add_message("user", "用户问题")
    buf.add_message("assistant", "助手回答")
    assert len(buf.messages) == 3
    roles = {m["role"] for m in buf.messages}
    assert roles == {"system", "user", "assistant"}
