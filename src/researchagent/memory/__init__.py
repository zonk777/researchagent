"""researchagent 记忆系统。

提供双层记忆架构：
    - ConversationBuffer: 短期对话缓冲 + 自动摘要压缩
    - LongTermMemory: 向量语义搜索 (LanceDB + BGE-M3)
    - MemoryManager: 统一协调接口

用法:
    from researchagent.memory import MemoryManager

    memory = MemoryManager(llm, state)
    memory.add_interaction("user", "我想学习 Python")
    context = memory.build_context("Python 学习路径")
"""

from researchagent.memory.manager import MemoryManager

__all__ = ["MemoryManager"]
