"""Demo: 信息研究场景。

测试 WebSearchTool — 搜索最新资讯并总结。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from researchagent.graph import build_agent_graph
from researchagent.core.state import RuntimeState


def main() -> None:
    print("=" * 50)
    print("Demo 2: 信息研究场景")
    print("=" * 50)

    state = RuntimeState(workspace=Path.cwd())
    graph = build_agent_graph()

    tasks = [
        "搜索 LangGraph 最新版本和主要特性，用中文总结",
        "搜索 Python 3.14 的新功能并列出 3 个最重要的",
    ]

    for task in tasks:
        print(f"\n>>> 任务: {task}\n")
        result = graph.invoke({
            "messages": [],
            "runtime": state,
            "task": task,
            "iteration_count": 0,
            "max_iterations": 8,
            "todos": [],
            "plan_summary": "",
            "acceptance_criteria": [],
            "verification_commands": [],
            "attempts": 0,
            "max_attempts": 2,
            "passed": False,
            "verifier_summary": "",
        })
        for msg in result["messages"]:
            content = getattr(msg, "content", "")
            if content and getattr(msg, "tool_calls", None) is None:
                print(f"Agent: {str(content)[:800]}")
        print("-" * 50)

    print("\nDemo 2 完成!")


if __name__ == "__main__":
    main()
