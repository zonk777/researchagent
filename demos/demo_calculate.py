"""Demo: 数学计算场景。

测试 CalculatorTool — 复杂数学表达式求值。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目路径在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from researchagent.graph import build_agent_graph
from researchagent.core.state import RuntimeState


def main() -> None:
    print("=" * 50)
    print("Demo 1: 数学计算场景")
    print("=" * 50)

    state = RuntimeState(workspace=Path.cwd())
    graph = build_agent_graph()

    tasks = [
        "计算 (123 + 456) * 789 的结果",
        "计算 sqrt(2^10) + sin(pi/4)",
    ]

    for task in tasks:
        print(f"\n>>> 任务: {task}\n")
        result = graph.invoke({
            "messages": [],
            "runtime": state,
            "task": task,
            "iteration_count": 0,
            "max_iterations": 5,
            "todos": [],
            "plan_summary": "",
            "acceptance_criteria": [],
            "verification_commands": [],
            "attempts": 0,
            "max_attempts": 2,
            "passed": False,
            "verifier_summary": "",
        })
        # 输出最终消息
        for msg in result["messages"]:
            content = getattr(msg, "content", "")
            if content and getattr(msg, "tool_calls", None) is None:
                print(f"Agent: {str(content)[:500]}")
        print("-" * 50)

    print("\nDemo 1 完成!")


if __name__ == "__main__":
    main()
