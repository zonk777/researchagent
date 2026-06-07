"""短期记忆：对话缓冲 + 自动摘要压缩。

ConversationBuffer 管理会话中的近期消息，当消息估计 token 数
超过阈值时自动调用 LLM 将旧消息压缩为摘要。

设计:
    - 消息以 list[dict] 形式存储 (role + content)
    - 粗略 token 估计: 总字符数 / 4
    - 保留最近 keep_recent 条消息不参与摘要
    - 摘要通过 LLM 调用实现，新旧摘要融合
"""

from __future__ import annotations


class ConversationBuffer:
    """短期对话缓冲，支持自动摘要压缩。

    使用方式:
        buf = ConversationBuffer(llm)
        buf.add_message("user", "什么是 Python？")
        buf.add_message("assistant", "Python 是一门...")
        context = buf.get_context()  # 获取格式化上下文
    """

    def __init__(
        self,
        llm,
        max_tokens: int = 4000,
        keep_recent: int = 6,
    ) -> None:
        """初始化对话缓冲。

        Args:
            llm: LangChain ChatModel 实例，用于生成摘要。
            max_tokens: 触发自动摘要的 token 阈值。
            keep_recent: 摘要时保留的最近消息条数。
        """
        self._llm = llm
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.messages: list[dict[str, str]] = []
        self.summary: str | None = None

    # ---- 公共方法 ----

    def add_message(self, role: str, content: str) -> None:
        """添加一条消息到缓冲中。

        如果估计 token 数超过 max_tokens，自动触发摘要压缩。

        Args:
            role: 消息角色 (user / assistant / system)。
            content: 消息内容。
        """
        self.messages.append({"role": role, "content": content})
        if self._needs_summarization():
            self._summarize()

    def get_context(self) -> str:
        """构建完整的上下文字符串。

        格式: [对话摘要] + [最近消息]
        供 LLM prompt 拼接使用。

        Returns:
            格式化后的上下文字符串。
        """
        parts: list[str] = []

        if self.summary:
            parts.append(f"[对话摘要]:\n{self.summary}\n")

        if self.messages:
            parts.append("[最近对话]:")
            for m in self.messages:
                parts.append(f"{m['role']}: {m['content']}")

        return "\n".join(parts)

    def clear(self) -> None:
        """清空所有消息和摘要。"""
        self.messages.clear()
        self.summary = None

    # ---- 内部方法 ----

    def _needs_summarization(self) -> bool:
        """检查是否需要触发摘要压缩。"""
        return self._estimate_tokens() > self.max_tokens

    def _estimate_tokens(self) -> int:
        """粗略估计当前消息的 token 数量。

        使用简单的字符数 / 4 估算 (含摘要文本)。
        """
        total_chars = sum(len(m["content"]) for m in self.messages)
        if self.summary:
            total_chars += len(self.summary)
        return total_chars // 4

    def _summarize(self) -> None:
        """将旧消息压缩为摘要。

        流程:
            1. 保留最近 keep_recent 条消息
            2. 旧消息 + 已有摘要 → LLM 融合
            3. 消息列表替换为仅保留的最近消息
        """
        # 保留最近 N 条
        if len(self.messages) <= self.keep_recent:
            return

        old_messages = self.messages[: -self.keep_recent]
        self.messages = self.messages[-self.keep_recent:]

        # 构建摘要 prompt
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in old_messages
        )

        if self.summary:
            prompt = (
                f"现有摘要:\n{self.summary}\n\n"
                f"新对话需要合并:\n{conversation_text}\n\n"
                f"请将以上内容合并为一段简洁的摘要，保留关键事实、"
                f"用户偏好和重要决策。用中文输出。"
            )
        else:
            prompt = (
                f"对话内容:\n{conversation_text}\n\n"
                f"请将以上对话总结为一段简洁的摘要，保留关键事实、"
                f"用户偏好和重要决策。用中文输出。"
            )

        try:
            from langchain_core.messages import HumanMessage

            response = self._llm.invoke([HumanMessage(content=prompt)])
            self.summary = response.content
        except Exception:
            # 摘要失败时保留原始消息（降级处理）
            self.messages = old_messages + self.messages
