"""researchagent 图 (Graph) 模块。

LangGraph StateGraph 定义和节点函数，实现 ReAct Agent 主循环。

导出:
    build_agent_graph: 构建编译后的 Agent 图
    AgentState: 图状态 TypedDict
"""

from researchagent.graph.state import AgentState
from researchagent.graph.workflow import build_agent_graph

__all__ = ["AgentState", "build_agent_graph"]
