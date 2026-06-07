"""Tool Registry 单元测试。"""

from __future__ import annotations

from pathlib import Path

from researchagent.core.state import RuntimeState
from researchagent.tools.registry import build_read_only_tools, build_tools


def _make_state() -> RuntimeState:
    return RuntimeState(workspace=Path.cwd())


def test_build_tools_returns_three_tools() -> None:
    """验证 build_tools 返回 3 个工具。"""
    state = _make_state()
    tools = build_tools(state)
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"CalculatorTool", "BashTool", "WebSearchTool"}


def test_build_read_only_tools_returns_two_tools() -> None:
    """验证 build_read_only_tools 返回 2 个工具 (不含 BashTool)。"""
    state = _make_state()
    tools = build_read_only_tools(state)
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"CalculatorTool", "WebSearchTool"}


def test_tools_have_description() -> None:
    """验证所有工具都有描述。"""
    state = _make_state()
    for tool in build_tools(state):
        assert tool.description and len(tool.description) > 10, (
            f"{tool.name} 描述过短: {tool.description}"
        )


def test_tools_can_be_invoked() -> None:
    """验证 CalculatorTool 可以被调用。"""
    state = _make_state()
    tools = build_tools(state)
    calc = next(t for t in tools if t.name == "CalculatorTool")

    result = calc.invoke({"expression": "1 + 1"})
    assert result is not None
    # 返回的 dict 应包含 ok
    assert "ok" in result or isinstance(result, str)


def test_bash_tool_can_be_invoked() -> None:
    """验证 BashTool 可以被调用。"""
    state = _make_state()
    tools = build_tools(state)
    bash = next(t for t in tools if t.name == "BashTool")

    result = bash.invoke({"command": "echo from_registry_test"})
    assert result is not None
    assert "ok" in result or isinstance(result, str)
