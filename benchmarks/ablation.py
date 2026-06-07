"""消融实验: 逐一关闭模块，测量对成功率的影响。

配置:
    Full: 完整 pipeline (planner + agent + tools + reflector)
    No-Reflector: 跳过反思节点
    No-Planner: 跳过规划节点
    No-Memory: 跳过记忆注入
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from researchagent.graph import build_agent_graph
from researchagent.core.state import RuntimeState
from benchmarks.e2e_bench import check_result, count_tool_calls, extract_final_output, load_tasks


def run_with_config(tasks: list[dict], config: str) -> list[dict]:
    """用指定配置运行全部任务。

    config:
        "full"  — 完整 pipeline
        "no_reflector" — 去掉 reflector
        "no_planner"   — 去掉 planner
        "no_memory"    — 去掉记忆上下文注入
    """
    results = []

    for task in tasks:
        print(f"  [{config}] {task['id']}: {task['task'][:50]}...")

        state = RuntimeState(workspace=Path.cwd())
        start = time.time()

        try:
            if config == "no_reflector":
                result = _run_no_reflector(task, state)
            elif config == "no_planner":
                result = _run_no_planner(task, state)
            elif config == "no_memory":
                result = _run_no_memory(task, state)
            else:
                result = _run_full(task, state)
        except Exception as exc:
            results.append({"id": task["id"], "passed": False, "error": str(exc)})
            print(f"    ERROR: {exc}")
            continue

        elapsed = time.time() - start
        messages = result.get("messages", [])
        output = extract_final_output(messages)
        passed, reason = check_result(task, output)
        tc = count_tool_calls(messages)
        iters = result.get("iteration_count", 0)

        status = "PASS" if passed else "FAIL"
        print(f"    {status} | {elapsed:.1f}s | {iters} iter | {tc} tools")

        results.append({
            "id": task["id"],
            "difficulty": task["difficulty"],
            "passed": passed,
            "elapsed_s": round(elapsed, 1),
            "iterations": iters,
            "tool_calls": tc,
        })

    return results


def _run_full(task: dict, state: RuntimeState) -> dict:
    graph = build_agent_graph()
    return graph.invoke({
        "messages": [],
        "runtime": state,
        "task": task["task"],
        "iteration_count": 0,
        "max_iterations": 10,
        "todos": [],
        "plan_summary": "",
        "acceptance_criteria": [],
        "verification_commands": [],
        "attempts": 0,
        "max_attempts": 2,
        "passed": False,
        "verifier_summary": "",
    })


def _run_no_reflector(task: dict, state: RuntimeState) -> dict:
    """用 Step 4 的 2 节点图 (无 planner + reflector)。"""
    from langgraph.graph import END, START, StateGraph
    from researchagent.graph.nodes import agent_node, tools_node, route_after_agent
    from researchagent.graph.state import AgentState

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")
    compiled = graph.compile()

    return compiled.invoke({
        "messages": [],
        "runtime": state,
        "task": task["task"],
        "iteration_count": 0,
        "max_iterations": 10,
    })


def _run_no_planner(task: dict, state: RuntimeState) -> dict:
    """用带 reflector 但跳过 planner 的图。"""
    from langgraph.graph import END, START, StateGraph
    from researchagent.graph.nodes import agent_node, tools_node, route_after_agent, reflector_node, reflector_route
    from researchagent.graph.state import AgentState

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("reflector", reflector_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "__end__": "reflector"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("reflector", reflector_route, {"__end__": END, "planner": "agent"})
    return graph.compile().invoke({
        "messages": [],
        "runtime": state,
        "task": task["task"],
        "iteration_count": 0,
        "max_iterations": 10,
        "todos": [],
        "plan_summary": "",
        "acceptance_criteria": [],
        "verification_commands": [],
        "attempts": 0,
        "max_attempts": 2,
        "passed": False,
        "verifier_summary": "",
    })


def _run_no_memory(task: dict, state: RuntimeState) -> dict:
    """用完整图但不注入长期记忆上下文。"""
    # 通过环境变量临时禁用记忆
    import os
    os.environ["RESEARCH_SKIP_MEMORY"] = "1"
    try:
        result = _run_full(task, state)
    finally:
        os.environ.pop("RESEARCH_SKIP_MEMORY", None)
    return result


def main() -> None:
    tasks = load_tasks()
    configs = ["full", "no_reflector", "no_planner", "no_memory"]

    all_results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running config: {config}")
        print(f"{'='*60}")
        results = run_with_config(tasks, config)
        passed = sum(1 for r in results if r["passed"])
        all_results[config] = {
            "passed": passed,
            "total": len(tasks),
            "rate": f"{passed}/{len(tasks)} ({passed/len(tasks)*100:.0f}%)",
            "avg_elapsed_s": round(sum(r.get("elapsed_s", 0) for r in results) / len(tasks), 1),
            "avg_iterations": round(sum(r.get("iterations", 0) for r in results) / len(tasks), 1),
            "avg_tool_calls": round(sum(r.get("tool_calls", 0) for r in results) / len(tasks), 1),
            "details": results,
        }

    # 保存
    out_path = Path(__file__).parent / "ablation.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nAblation results saved to {out_path}")

    # 对比表
    print("\n消融实验对比:")
    print(f"{'Config':<15} {'Success':<10} {'Avg Time':<10} {'Avg Iter':<10} {'Avg Tools':<10}")
    print("-" * 60)
    for config in configs:
        r = all_results[config]
        print(f"{config:<15} {r['rate']:<10} {r['avg_elapsed_s']}s{'':>5} {r['avg_iterations']}{'':>5} {r['avg_tool_calls']}{'':>5}")


if __name__ == "__main__":
    main()
