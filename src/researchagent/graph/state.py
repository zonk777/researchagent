"""Agent 图状态定义。

定义 ReAct Agent 循环中使用的 LangGraph State TypedDict。
Step 5 新增规划+反思相关字段。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from researchagent.core.state import RuntimeState


class AgentState(TypedDict, total=False):
    """ReAct Agent 的图状态。

    LangGraph 在每个节点执行后将返回的 dict 合并到当前状态中。
    messages 字段使用 add_messages 归约器自动追加 (而非替换)。

    字段:
        messages: 对话历史 (自动追加)。
        runtime: 运行时配置 (引用传递)。
        task: 当前用户任务。
        iteration_count: ReAct 循环迭代次数。
        max_iterations: 最大迭代次数 (默认 10)。
        todos: 子任务列表 [{id, content, status}]。
        plan_summary: 规划摘要。
        acceptance_criteria: 验收标准列表。
        verification_commands: 反思验证命令。
        attempts: 规划+反思重试次数。
        max_attempts: 最大重试次数 (默认 3)。
        passed: 反思是否通过。
        verifier_summary: 反思摘要。
    """

    messages: Annotated[list[BaseMessage], add_messages]
    runtime: RuntimeState
    task: str
    iteration_count: int
    max_iterations: int
    todos: list[dict]
    plan_summary: str
    acceptance_criteria: list[str]
    verification_commands: list[str]
    attempts: int
    max_attempts: int
    passed: bool
    verifier_summary: str
