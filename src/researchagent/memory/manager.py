"""记忆管理器：统一协调短期和长期记忆。

MemoryManager 是记忆系统的对外统一接口，协调:
    - ConversationBuffer: 短期对话管理
    - LongTermMemory: 长期语义搜索

使用方式:
    from researchagent.memory import MemoryManager

    manager = MemoryManager(llm, state)
    manager.add_interaction("user", "我喜欢用 Python 写爬虫")
    manager.add_interaction("assistant", "Python 的 requests 库很适合爬虫")

    # 构建带记忆增强的上下文
    context = manager.build_context("推荐一个爬虫框架")

    # 存储重要偏好
    manager.remember_preference("用户偏好 Python 语言")
"""

from __future__ import annotations

import logging

from researchagent.core.state import RuntimeState
from researchagent.memory.long_term import LongTermMemory
from researchagent.memory.short_term import ConversationBuffer

logger = logging.getLogger(__name__)


class MemoryManager:
    """统一记忆管理器。

    协调短期对话缓冲和长期向量记忆，为 Agent 提供记忆能力。
    """

    def __init__(self, llm, state: RuntimeState) -> None:
        """初始化记忆管理器。

        Args:
            llm: LangChain ChatModel 实例，用于短期记忆的摘要生成。
            state: 运行时状态，提供记忆系统配置。
        """
        self.short_term = ConversationBuffer(
            llm=llm,
            max_tokens=state.short_term_max_tokens,
            keep_recent=state.short_term_keep_recent,
        )
        self.long_term = LongTermMemory(
            store_path=state.memory_store_path
            if state.memory_store_path.is_absolute()
            else state.workspace / state.memory_store_path,
        )

    # ---- 记录对话 ----

    def add_interaction(self, role: str, content: str) -> None:
        """记录一次完整对话交互。

        同时写入短期缓冲和长期向量库。

        Args:
            role: 消息角色 (user / assistant / system)。
            content: 消息内容。
        """
        # 短期记忆：直接追加，超阈值自动摘要
        self.short_term.add_message(role, content)

        # 长期记忆：存储到向量库
        ok = self.long_term.add_interaction(content, role=role)
        if not ok:
            logger.warning(
                "Long-term memory write failed: role=%s, content=%r", role, content[:200],
            )

    # ---- 记忆检索 ----

    def build_context(self, user_query: str = "") -> str:
        """构建增强的上下文。综合短期记忆和长期记忆。

        格式:
            [对话摘要]
            [最近对话]
            [相关历史记忆]
            [当前问题]

        Args:
            user_query: 当前用户问题，用于搜索相关长期记忆。

        Returns:
            格式化后的上下文字符串。
        """
        parts: list[str] = []

        # 短期记忆上下文
        short_context = self.short_term.get_context()
        if short_context:
            parts.append(short_context)

        # 长期记忆：语义搜索相关历史
        if user_query:
            memories = self.long_term.search(user_query, k=3)
            if memories:
                parts.append("")
                parts.append("[相关历史记忆]:")
                for mem in memories:
                    cat_tag = f"[{mem['category']}]" if mem.get("category") else ""
                    parts.append(f"- {cat_tag} {mem['text']}")

        return "\n".join(parts)

    def search_memories(self, query: str, k: int = 5) -> list[dict]:
        """在长期记忆中搜索相关记录。

        Args:
            query: 搜索查询。
            k: 最大返回数。

        Returns:
            相关记忆列表。
        """
        return self.long_term.search(query, k=k)

    # ---- 关键信息存储 ----

    def remember_fact(self, fact: str) -> None:
        """存储一条重要事实。

        Args:
            fact: 事实文本。
        """
        self.long_term.add_fact(fact)

    def remember_preference(self, preference: str) -> None:
        """存储一条用户偏好。

        Args:
            preference: 偏好文本。
        """
        self.long_term.add_preference(preference)

    # ---- 工具方法 ----

    def get_short_term_summary(self) -> str | None:
        """获取当前短期记忆的摘要。

        Returns:
            摘要文本，无摘要时返回 None。
        """
        return self.short_term.summary

    def get_recent_messages(self) -> list[dict[str, str]]:
        """获取最近的对话消息。

        Returns:
            消息列表，每项含 role 和 content。
        """
        return list(self.short_term.messages)

    def memory_count(self) -> int:
        """获取长期记忆中存储的总条数。

        Returns:
            记录数量。
        """
        return self.long_term.count()

    def clear_short_term(self) -> None:
        """清空短期记忆（长期记忆保留）。"""
        self.short_term.clear()
