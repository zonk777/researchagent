"""Token 用量追踪器。

记录每次 LLM 调用的 token 消耗，支持按节点/任务聚合统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """单次 LLM 调用的 token 用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    node: str = ""  # planner / agent / reflector


@dataclass
class TokenTracker:
    """Token 用量追踪器。

    用法:
        tracker = TokenTracker()
        response = model.invoke(messages)
        tracker.record(response, node="agent")
        print(tracker.report())
    """

    usages: list[TokenUsage] = field(default_factory=list)

    def record(self, response: Any, node: str = "agent") -> None:
        """从 LLM 响应中提取并记录 token 用量。

        Args:
            response: LangChain AIMessage 或包含 usage_metadata 的响应。
            node: 当前节点名称。
        """
        usage = TokenUsage(node=node)

        # LangChain ChatOpenAI 响应
        if hasattr(response, "usage_metadata"):
            meta = response.usage_metadata
            usage.prompt_tokens = meta.get("input_tokens", 0)
            usage.completion_tokens = meta.get("output_tokens", 0)
            usage.total_tokens = meta.get("total_tokens", 0)
        elif hasattr(response, "response_metadata"):
            meta = response.response_metadata
            usage = TokenUsage(
                prompt_tokens=meta.get("token_usage", {}).get("prompt_tokens", 0),
                completion_tokens=meta.get("token_usage", {}).get("completion_tokens", 0),
                total_tokens=meta.get("token_usage", {}).get("total_tokens", 0),
                model=meta.get("model_name", ""),
                node=node,
            )

        if hasattr(response, "model_name"):
            usage.model = response.model_name

        self.usages.append(usage)

    def summary(self) -> dict:
        """返回 token 用量统计摘要。

        Returns:
            {
                "total_prompt": int,
                "total_completion": int,
                "total": int,
                "total_calls": int,
                "by_node": {node: {prompt, completion, total, calls}},
            }
        """
        by_node: dict[str, dict] = {}
        total_prompt = 0
        total_completion = 0

        for u in self.usages:
            total_prompt += u.prompt_tokens
            total_completion += u.completion_tokens

            if u.node not in by_node:
                by_node[u.node] = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
            by_node[u.node]["prompt"] += u.prompt_tokens
            by_node[u.node]["completion"] += u.completion_tokens
            by_node[u.node]["total"] += u.total_tokens
            by_node[u.node]["calls"] += 1

        return {
            "total_prompt": total_prompt,
            "total_completion": total_completion,
            "total": total_prompt + total_completion,
            "total_calls": len(self.usages),
            "by_node": by_node,
        }

    def report(self) -> str:
        """生成可读的 Token 用量报告。"""
        s = self.summary()
        lines = [
            f"Token Usage Report",
            f"{'='*40}",
            f"Total calls: {s['total_calls']}",
            f"Total tokens: {s['total']:,} (P: {s['total_prompt']:,} / C: {s['total_completion']:,})",
            f"",
            f"{'Node':<12} {'Calls':<8} {'Prompt':<10} {'Completion':<12} {'Total':<10}",
            f"{'-'*52}",
        ]
        for node, stats in s["by_node"].items():
            lines.append(
                f"{node:<12} {stats['calls']:<8} {stats['prompt']:<10,} "
                f"{stats['completion']:<12,} {stats['total']:<10,}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        """重置所有统计。"""
        self.usages.clear()
