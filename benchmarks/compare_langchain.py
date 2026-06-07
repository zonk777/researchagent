"""对比实验: researchagent vs LangChain 原生 create_react_agent。

对相同的 10 个 Benchmark 任务分别用两套 Agent 执行并对比结果。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_tasks() -> list[dict]:
    path = Path(__file__).parent / "tasks.json"
    return json.loads(path.read_text(encoding="utf-8"))["tasks"]


def count_tool_calls(messages: list) -> int:
    count = 0
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        count += len(tool_calls)
    return count


def extract_final_output(messages: list) -> str:
    texts = []
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None) or []
        if content and not tool_calls:
            texts.append(str(content))
        if len(texts) >= 2:
            break
    return "\n".join(reversed(texts))


def check_result(task: dict, output: str) -> bool:
    """简化版检查：验证关键词是否在输出中。"""
    keywords = task.get("golden_keywords", [])
    output_lower = output.lower()
    check_type = task.get("check", "")

    if check_type == "contains_number":
        numbers = re.findall(r"\d+\.?\d*", output)
        return any(k in numbers for k in keywords)
    elif check_type == "contains_number_near_1":
        numbers = [float(n) for n in re.findall(r"\d+\.?\d*", output)]
        return any(abs(n - 1.0) < 0.01 for n in numbers)
    elif check_type == "file_created":
        workspace = Path.cwd()
        return any((workspace / kw).exists() or list(workspace.rglob(kw)) for kw in keywords)
    elif check_type == "contains_any_keyword":
        return any(k.lower() in output_lower for k in keywords)
    else:
        return all(k.lower() in output_lower for k in keywords)


# ============================================================
# KorinAgentFlow Agent
# ============================================================

def run_researchagent(task: dict) -> dict:
    from researchagent.graph import build_agent_graph
    from researchagent.core.state import RuntimeState

    graph = build_agent_graph()
    state = RuntimeState(workspace=Path.cwd())

    start = time.time()
    result = graph.invoke({
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
    elapsed = time.time() - start

    messages = result.get("messages", [])
    return {
        "elapsed_s": round(elapsed, 1),
        "iterations": result.get("iteration_count", 0),
        "tool_calls": count_tool_calls(messages),
        "output": extract_final_output(messages),
    }


# ============================================================
# LangChain 原生 ReAct Agent
# ============================================================

def run_langchain_react(task: dict) -> dict:
    from langgraph.prebuilt import create_react_agent
    from researchagent.providers.openai_provider import create_model
    from researchagent.tools.registry import build_tools
    from researchagent.core.state import RuntimeState

    state = RuntimeState(workspace=Path.cwd())
    model = create_model()
    tools = build_tools(state)

    agent = create_react_agent(model, tools)

    start = time.time()
    result = agent.invoke({"messages": [("user", task["task"])]})
    elapsed = time.time() - start

    messages = result.get("messages", [])
    return {
        "elapsed_s": round(elapsed, 1),
        "iterations": len([m for m in messages if getattr(m, "tool_calls", None)]),
        "tool_calls": count_tool_calls(messages),
        "output": extract_final_output(messages),
    }


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    tasks = load_tasks()

    korin_results = []
    lc_results = []

    korin_passed = 0
    lc_passed = 0

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"[{task['id']}] {task['difficulty']}")
        print(f"Task: {task['task'][:80]}...")
        print(f"{'='*60}")

        # KorinAgentFlow
        try:
            r1 = run_researchagent(task)
            p1 = check_result(task, r1["output"])
            if p1:
                korin_passed += 1
            korin_results.append({**r1, "id": task["id"], "passed": p1})
            print(f"  Korin: {'PASS' if p1 else 'FAIL'} | {r1['elapsed_s']}s | {r1['iterations']} iter | {r1['tool_calls']} tools")
        except Exception as e:
            korin_results.append({"id": task["id"], "passed": False, "error": str(e)})
            print(f"  Korin: ERROR — {e}")

        # LangChain ReAct
        try:
            r2 = run_langchain_react(task)
            p2 = check_result(task, r2["output"])
            if p2:
                lc_passed += 1
            lc_results.append({**r2, "id": task["id"], "passed": p2})
            print(f"  LangChain: {'PASS' if p2 else 'FAIL'} | {r2['elapsed_s']}s | {r2['iterations']} iter | {r2['tool_calls']} tools")
        except Exception as e:
            lc_results.append({"id": task["id"], "passed": False, "error": str(e)})
            print(f"  LangChain: ERROR — {e}")

    # 汇总
    n = len(tasks)
    comparison = {
        "framework": {
            "researchagent": f"{korin_passed}/{n} ({korin_passed/n*100:.0f}%)",
            "langchain_react": f"{lc_passed}/{n} ({lc_passed/n*100:.0f}%)",
        },
        "researchagent": {
            "avg_elapsed_s": round(sum(r.get("elapsed_s", 0) for r in korin_results) / n, 1),
            "avg_iterations": round(sum(r.get("iterations", 0) for r in korin_results) / n, 1),
            "avg_tool_calls": round(sum(r.get("tool_calls", 0) for r in korin_results) / n, 1),
        },
        "langchain_react": {
            "avg_elapsed_s": round(sum(r.get("elapsed_s", 0) for r in lc_results) / n, 1),
            "avg_iterations": round(sum(r.get("iterations", 0) for r in lc_results) / n, 1),
            "avg_tool_calls": round(sum(r.get("tool_calls", 0) for r in lc_results) / n, 1),
        },
        "details": {
            "researchagent": korin_results,
            "langchain_react": lc_results,
        },
    }

    out_path = Path(__file__).parent / "comparison.json"
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"\nComparison saved to {out_path}")
    print(f"\nKorinAgentFlow: {korin_passed}/{n} | LangChain ReAct: {lc_passed}/{n}")


if __name__ == "__main__":
    main()
