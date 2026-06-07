"""运行时状态 (Runtime State) 定义。

本模块定义 `RuntimeState` 数据类，用于在整个 agent 会话中传递
运行时配置。当前为最小化版本，后续 Step 将逐步扩展：
    - Step 2: 添加工具注册表和工作区路径管理
    - Step 3: 添加记忆系统配置（短期对话 + 长期向量库）
    - Step 4: 添加 LangGraph 图状态和事件流配置
    - Step 5: 添加任务规划和反思机制的追踪字段

参考 MokioAgent 的 `core/state.py` 设计模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RuntimeState:
    """researchagent 会话的运行时配置。

    作为整个 agent 系统中传递配置的载体。在 Step 1 中仅包含
    工作区和 LLM 配置，后续将逐步添加更多字段。

    Attributes:
        workspace: 会话的工作目录。
        model_name: 当前使用的 LLM 模型名称。
        base_url: OpenAI 兼容 API 的端点地址。
        api_key: API 认证密钥。
        max_attempts: Agent 最大重试次数（后续 Step 使用）。
        metadata: 可扩展的元数据字典。
    """

    # ---- 必需字段 ----
    workspace: Path
    """会话的工作目录，所有文件操作在此目录下进行。"""

    # ---- LLM 配置 ----
    model_name: str = ""
    """当前使用的 LLM 模型名称，例如 "gpt-4o-mini"。"""

    base_url: str = ""
    """OpenAI 兼容 API 的端点地址，例如 "https://api.openai.com/v1"。"""

    api_key: str = ""
    """API 认证密钥。"""

    # ---- Shell 执行配置 ----
    bash_default_timeout_seconds: int = 120
    """BashTool 默认超时时间（秒）。"""

    bash_max_timeout_seconds: int = 600
    """BashTool 允许的最大超时时间（秒）。"""

    bash_max_output_chars: int = 6000
    """BashTool 输出截断阈值（字符数），超出部分写入文件。"""

    # ---- Agent 配置（后续 Step 使用） ----
    max_attempts: int = 3
    """Agent 在遇到错误后最多重试的次数。"""

    # ---- 记忆系统配置 ----
    memory_store_path: Path = field(default_factory=lambda: Path(".researchagent/memory"))
    """长期记忆向量库的持久化路径。"""

    short_term_max_tokens: int = 4000
    """短期记忆触发自动摘要的 token 阈值。"""

    short_term_keep_recent: int = 6
    """摘要时保留的最近消息条数。"""

    # ---- 扩展字段 ----
    metadata: dict = field(default_factory=dict)
    """可自由扩展的元数据字典，用于存储自定义配置。"""

    # ---- 内部缓存 (不通过 __init__ 设置) ----
    _memory_manager: Any = field(default=None, repr=False, init=False)
    """MemoryManager 缓存，避免每个节点重复创建和加载 BGE-M3。"""
