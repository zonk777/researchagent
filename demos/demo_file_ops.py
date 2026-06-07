"""Demo: 文件操作场景。

测试 BashTool — Shell 命令执行: 创建文件、查看目录、运行命令。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from researchagent.graph import build_agent_graph
from researchagent.core.state import RuntimeState


def main() -> None:
    print("=" * 50)
    print("Demo 3: 文件操作场景")
    print("=" * 50)

    state = RuntimeState(workspace=Path.cwd())
    graph = build_agent_graph()

    tasks = [
        "列出当前项目目录下的所有 Python 文件",
        "查看 pyproject.toml 的内容并告诉我这个项目叫什么名字",
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

    print("\nDemo 3 完成!")


if __name__ == "__main__":
    main()
