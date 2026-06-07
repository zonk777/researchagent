"""Agent 图结构测试。

验证:
    - 图正确编译 (4 节点: planner, agent, tools, reflector)
    - 路由逻辑正确
    - 新节点基本行为
"""

from __future__ import annotations

from pathlib import Path

from researchagent.core.state import RuntimeState
from researchagent.graph.state import AgentState
from researchagent.graph.nodes import (
    reflector_route,
    route_after_agent,
)
from researchagent.graph.workflow import build_agent_graph


def _make_state(**overrides) -> AgentState:
    """创建最小测试状态。"""
    base: AgentState = {
        "messages": [],
        "runtime": RuntimeState(workspace=Path.cwd()),
        "task": "test",
        "iteration_count": 0,
        "max_iterations": 10,
        "todos": [],
        "plan_summary": "",
        "acceptance_criteria": [],
        "verification_commands": [],
        "attempts": 0,
        "max_attempts": 3,
        "passed": False,
        "verifier_summary": "",
    }
    base.update(overrides)
    return base


# ---- 图编译测试 ----

def test_build_graph_returns_compiled_graph() -> None:
    """验证 4 节点图正确编译。"""
    graph = build_agent_graph()
    assert graph is not None
    nodes = graph.get_graph().nodes
    assert "planner" in nodes
    assert "agent" in nodes
    assert "tools" in nodes
    assert "reflector" in nodes


# ---- agent 路由测试 ----

def test_route_after_agent_no_tool_calls() -> None:
    """验证无 tool_calls 时路由到 __end__。"""
    from langchain_core.messages import AIMessage

    state = _make_state(messages=[AIMessage(content="Hello")])
    assert route_after_agent(state) == "__end__"


def test_route_after_agent_with_tool_calls() -> None:
    """验证有 tool_calls 时路由到 tools。"""
    from langchain_core.messages import AIMessage

    msg = AIMessage(content="")
    msg.tool_calls = [{"name": "CalculatorTool", "args": {"expression": "1+1"}, "id": "call_1"}]

    state = _make_state(messages=[msg])
    assert route_after_agent(state) == "tools"


def test_route_at_max_iterations() -> None:
    """验证达到最大迭代次数时不再路由到 tools。"""
    from langchain_core.messages import AIMessage

    msg = AIMessage(content="")
    msg.tool_calls = [{"name": "CalculatorTool", "args": {"expression": "1+1"}, "id": "call_1"}]

    state = _make_state(messages=[msg], iteration_count=10, max_iterations=10)
    assert route_after_agent(state) == "__end__"


def test_route_empty_messages() -> None:
    """验证空消息时路由到 __end__。"""
    state = _make_state(messages=[])
    assert route_after_agent(state) == "__end__"


# ---- reflector 路由测试 ----

def test_reflector_route_passed() -> None:
    """验证反思通过时路由到 __end__。"""
    state = _make_state(passed=True)
    assert reflector_route(state) == "__end__"


def test_reflector_route_not_passed_retry() -> None:
    """验证未通过且未达上限时路由到 agent 重试。"""
    state = _make_state(passed=False, attempts=1, max_attempts=3)
    assert reflector_route(state) == "agent"


def test_reflector_route_max_attempts() -> None:
    """验证达到最大重试次数时路由到 __end__。"""
    state = _make_state(passed=False, attempts=3, max_attempts=3)
    assert reflector_route(state) == "__end__"


def test_reflector_route_default_passed() -> None:
    """验证默认 passed=False 时触发 agent 重试。"""
    state = _make_state(passed=False, attempts=0, max_attempts=3)
    assert reflector_route(state) == "agent"


# ---- 状态测试 ----

def test_agent_state_new_fields() -> None:
    """验证 Step 5 新增字段可正常访问。"""
    state = _make_state()
    assert state["todos"] == []
    assert state["plan_summary"] == ""
    assert state["acceptance_criteria"] == []
    assert state["attempts"] == 0
    assert state["max_attempts"] == 3
    assert state["passed"] is False
    assert state["verifier_summary"] == ""


def test_agent_state_partial_init() -> None:
    """验证 total=False 时只需部分字段。"""
    state: AgentState = {
        "messages": [],
        "runtime": RuntimeState(workspace=Path.cwd()),
        "task": "minimal",
    }
    assert state["task"] == "minimal"
