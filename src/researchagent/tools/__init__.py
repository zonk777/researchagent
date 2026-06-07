"""researchagent 工具系统。

提供 Agent 可调用的工具集，包括：
    - CalculatorTool: 安全数学表达式求值
    - BashTool: Shell 命令执行
    - WebSearchTool: Tavily 网络搜索

用法:
    from researchagent.tools import build_tools, build_read_only_tools

    tools = build_tools(state)             # 全部工具
    readonly_tools = build_read_only_tools(state)  # 只读工具
"""

from researchagent.tools.registry import build_read_only_tools, build_tools

__all__ = ["build_read_only_tools", "build_tools"]
