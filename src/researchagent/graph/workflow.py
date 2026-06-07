"""Agent 图构建。

使用 LangGraph StateGraph 构建规划-ReAct-反思 Agent 循环:

Step 5:
    START → planner → agent → {tool_calls → tools → agent}
                           → {no tools → reflector}
                                     → {passed → END}
                                     → {not passed → planner}

Step 4 (原):
    START → agent → {tool_calls → tools → agent} → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from researchagent.graph.nodes import (
    agent_node,
    planner_node,
    reflector_node,
    reflector_route,
    route_after_agent,
    tools_node,
)
from researchagent.graph.state import AgentState


def build_agent_graph() -> StateGraph:
    """构建并编译规划-ReAct-反思 Agent 图。

    4 节点流程:
        planner: 拆解任务 → 生成待办和验收标准
        agent:   调用 LLM 决定行动 (ReAct 循环)
        tools:   执行工具调用
        reflector: 检查结果 → 通过则结束，否则重新规划

    Returns:
        编译后的 LangGraph StateGraph。
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("reflector", reflector_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "__end__": "reflector",  # 无工具调用时进入反思
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges(
        "reflector",
        reflector_route,
        {
            "__end__": END,
            "agent": "agent",
        },
    )

    return graph.compile()
