"""端到端 Benchmark 评测。

运行 10 个预定义任务，对每个任务评分，输出汇总报告。
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from researchagent.graph import build_agent_graph
from researchagent.core.state import RuntimeState


def load_tasks() -> list[dict]:
    path = Path(__file__).parent / "tasks.json"
    return json.loads(path.read_text(encoding="utf-8"))["tasks"]


def check_result(task: dict, output: str) -> tuple[bool, str]:
    """根据任务定义检查输出。

    Returns:
        (passed, reason)
    """
    check_type = task.get("check", "contains_all_keywords")
    keywords = task.get("golden_keywords", [])
    output_lower = output.lower()

    if check_type == "contains_all_keywords":
        missing = [k for k in keywords if k.lower() not in output_lower]
        if missing:
            return False, f"缺少关键词: {missing}"
        return True, "所有关键词匹配"

    elif check_type == "contains_any_keyword":
        found = [k for k in keywords if k.lower() in output_lower]
        if not found:
            return False, f"未匹配任何关键词: {keywords}"
        return True, f"匹配: {found}"

    elif check_type == "contains_number":
        numbers = re.findall(r"\d+\.?\d*", output)
        for k in keywords:
            if k in numbers:
                return True, f"数值 {k} 存在"
        return False, f"未找到预期数值: {keywords}"

    elif check_type == "contains_number_near_1":
        numbers = [float(n) for n in re.findall(r"\d+\.?\d*", output)]
        for n in numbers:
            if abs(n - 1.0) < 0.01:
                return True, f"结果约等于 1.0 (实际 {n})"
        return False, f"未找到接近 1.0 的数值"

    elif check_type == "file_created":
        workspace = Path.cwd()
        for kw in keywords:
            path = workspace / kw
            if path.exists() or any(p.exists() for p in workspace.rglob(kw)):
                return True, f"文件 {kw} 已创建"
        return False, f"未找到文件: {keywords}"

    return False, f"未知检查类型: {check_type}"


def extract_final_output(messages: list) -> str:
    """从消息列表中提取最终输出。"""
    texts = []
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None) or []
        if content and not tool_calls:
            texts.append(str(content))
        if len(texts) >= 2:
            break
    return "\n".join(reversed(texts))


def count_tool_calls(messages: list) -> int:
    """统计工具调用次数。"""
    count = 0
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        count += len(tool_calls)
    return count


def run_benchmark() -> dict:
    """运行全部 10 个 Benchmark 任务。"""
    tasks = load_tasks()
    graph = build_agent_graph()

    results = []
    passed_count = 0
    total_time = 0.0
    total_iterations = 0
    total_tool_calls = 0

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"[{task['id']}] {task['difficulty']} — {task['task'][:60]}...")
        print(f"{'='*60}")

        state = RuntimeState(workspace=Path.cwd())
        start = time.time()

        try:
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
        except Exception as exc:
            elapsed = time.time() - start
            results.append({
                "id": task["id"],
                "difficulty": task["difficulty"],
                "task": task["task"],
                "passed": False,
                "error": str(exc),
                "elapsed_s": round(elapsed, 1),
                "iterations": 0,
                "tool_calls": 0,
            })
            print(f"  ERROR: {exc}")
            continue

        elapsed = time.time() - start
        messages = result.get("messages", [])
        output = extract_final_output(messages)
        iterations = result.get("iteration_count", 0)
        tc = count_tool_calls(messages)
        passed, reason = check_result(task, output)

        total_time += elapsed
        total_iterations += iterations
        total_tool_calls += tc
        if passed:
            passed_count += 1

        status = "PASS" if passed else "FAIL"
        print(f"  {status} | {elapsed:.1f}s | {iterations} iter | {tc} tools | {reason[:80]}")

        results.append({
            "id": task["id"],
            "difficulty": task["difficulty"],
            "task": task["task"],
            "passed": passed,
            "reason": reason,
            "elapsed_s": round(elapsed, 1),
            "iterations": iterations,
            "tool_calls": tc,
            "output_preview": output[:300],
        })

    # 汇总
    summary = {
        "total_tasks": len(tasks),
        "passed": passed_count,
        "failed": len(tasks) - passed_count,
        "pass_rate": f"{passed_count / len(tasks) * 100:.1f}%",
        "total_time_s": round(total_time, 1),
        "avg_time_s": round(total_time / len(tasks), 1),
        "avg_iterations": round(total_iterations / len(tasks), 1),
        "avg_tool_calls": round(total_tool_calls / len(tasks), 1),
    }

    # 按难度分组
    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {"total": 0, "passed": 0}
        by_difficulty[d]["total"] += 1
        if r["passed"]:
            by_difficulty[d]["passed"] += 1

    summary["by_difficulty"] = {
        d: f"{v['passed']}/{v['total']} ({v['passed']/v['total']*100:.0f}%)"
        for d, v in sorted(by_difficulty.items())
    }

    output_data = {"summary": summary, "results": results}

    # 保存结果
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")
    print(f"\nSummary: {json.dumps(summary, ensure_ascii=False, indent=2)}")

    return output_data


if __name__ == "__main__":
    run_benchmark()
