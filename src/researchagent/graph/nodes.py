"""ReAct Agent 图节点。

提供:
    - planner_node: 任务拆解为子任务计划
    - agent_node: 调用 LLM (绑定工具) 决定下一步行动
    - tools_node: 执行 LLM 请求的工具调用
    - reflector_node: 反思执行结果，决定是否重试
    - route_after_agent: 条件路由 (有 tool_calls → tools)
    - reflector_route: 反思后路由 (通过 → END, 未通过 → planner)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

from researchagent.core.state import RuntimeState
from researchagent.graph.state import AgentState
from researchagent.memory.manager import MemoryManager
from researchagent.prompts.agent import AGENT_SYSTEM_PROMPT
from researchagent.core.llm_factory import get_model_for_node
from researchagent.core.retry import retry_llm
from researchagent.tools.registry import build_tools


@retry_llm(max_retries=3)
def _invoke_with_retry(llm, messages):
    """带重试的 LLM 调用。"""
    return llm.invoke(messages)


def agent_node(state: AgentState) -> dict[str, Any]:
    """Agent 节点: 调用 LLM 分析当前状态并决定行动。

    每次调用:
        1. 构建记忆增强的上下文
        2. 绑定工具到 LLM
        3. 发送消息历史给 LLM
        4. 返回 LLM 的响应和更新后的迭代计数

    Args:
        state: 当前图状态。

    Returns:
        包含 messages 和 iteration_count 的更新 dict。
    """
    runtime: RuntimeState = state["runtime"]
    messages: list = list(state["messages"])
    task: str = state.get("task", "")

    # 创建 LLM 实例并绑定工具
    llm = get_model_for_node("agent")
    # 支持通过 runtime 注入自定义工具集（用于 research 等垂直场景）
    tool_builder = getattr(runtime, "tool_builder", None)
    tools = tool_builder(runtime) if tool_builder else build_tools(runtime)
    agent_llm = llm.bind_tools(tools)

    # 始终初始化 MemoryManager（后续无条件使用）
    memory = getattr(runtime, "_memory_manager", None)
    if memory is None:
        memory = MemoryManager(llm, runtime)
        object.__setattr__(runtime, "_memory_manager", memory)

    # 支持 RESEARCH_SKIP_MEMORY 环境变量用于消融实验
    skip_memory = os.environ.get("RESEARCH_SKIP_MEMORY", "").lower() in ("1", "true", "yes")
    memory_context = "" if skip_memory else memory.build_context(task)

    # 首次调用或反射后的重试（消息被清空）时，注入系统提示
    if not messages or not any(isinstance(m, SystemMessage) for m in messages):
        system_content = AGENT_SYSTEM_PROMPT
        if memory_context:
            system_content += f"\n\n## 历史记忆\n{memory_context}"
        system_msg = SystemMessage(content=system_content)
        if not messages:
            messages = [system_msg, HumanMessage(content=f"<user_query>\n{task}\n</user_query>")]
        else:
            messages = [system_msg] + list(messages)

    # 调用 LLM
    response = _invoke_with_retry(agent_llm, messages)

    # 记录交互到记忆系统
    if isinstance(response, AIMessage) and response.content:
        memory.add_interaction("assistant", str(response.content))

    # 记录 token 用量
    tracker = getattr(runtime, "_token_tracker", None)
    if tracker is not None:
        tracker.record(response, node="agent")
    logger.info("agent_node: %d messages, iteration=%d", len(messages), state.get("iteration_count", 0))

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def tools_node(state: AgentState) -> dict[str, Any]:
    """工具节点: 执行 LLM 请求的工具调用。

    从最后一条 AIMessage 中提取 tool_calls，
    逐个执行并将结果包装为 ToolMessage 返回。

    Args:
        state: 当前图状态。

    Returns:
        包含 ToolMessage 列表的更新 dict。
    """
    runtime: RuntimeState = state["runtime"]
    messages: list = list(state["messages"])

    # 获取最后一条消息
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []

    # 构建工具注册表
    tools = build_tools(runtime)
    tools_by_name = {t.name: t for t in tools}

    # 复用缓存的 MemoryManager
    memory = getattr(runtime, "_memory_manager", None)
    if memory is None:
        llm = get_model_for_node("tools")
        memory = MemoryManager(llm, runtime)
        object.__setattr__(runtime, "_memory_manager", memory)

    tool_messages: list[ToolMessage] = []
    for call in tool_calls:
        tool_name = call.get("name", "")
        tool_args = call.get("args", {})
        tool_call_id = call.get("id", f"{tool_name}-call")

        tool = tools_by_name.get(tool_name)
        if tool is None:
            result = {"ok": False, "error": f"未知工具: {tool_name}"}
        else:
            try:
                result = tool.invoke(tool_args)
            except Exception as exc:
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        # 确保结果为字符串
        if isinstance(result, dict):
            result_str = json.dumps(result, ensure_ascii=False, default=str)
        else:
            result_str = str(result)

        tool_messages.append(ToolMessage(
            content=result_str,
            name=tool_name,
            tool_call_id=tool_call_id,
        ))

        # 记录工具调用到记忆
        memory.add_interaction("system", f"[Tool: {tool_name}] {result_str[:500]}")

    return {"messages": tool_messages}


def route_after_agent(state: AgentState) -> str:
    """条件路由: 根据最后一条消息是否包含 tool_calls 决定下一步。

    Args:
        state: 当前图状态。

    Returns:
        "tools" — 需要执行工具调用。
        "__end__" — LLM 已给出最终回复。
    """
    messages: list = list(state["messages"])
    if not messages:
        return "__end__"

    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []

    # 检查迭代次数限制
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 10)

    if tool_calls and iteration < max_iter:
        return "tools"
    return "__end__"


def planner_node(state: AgentState) -> dict[str, Any]:
    """规划节点: 将用户任务拆解为可执行的子步骤。

    仅在首次进入时执行 (todos 为空时)。生成结构化计划:
        - todos: 子任务列表
        - acceptance_criteria: 验收标准
        - verification_commands: 验证命令

    Args:
        state: 当前图状态。

    Returns:
        包含 todos/plan_summary/attempts 的更新 dict。
    """
    # 已有计划则跳过
    if state.get("todos"):
        return {}

    task: str = state.get("task", "")

    # 简单任务跳过规划
    if _is_simple_task(task):
        return {
            "todos": [{"id": "1", "content": task, "status": "pending"}],
            "plan_summary": "简单任务，直接执行",
            "acceptance_criteria": [],
            "verification_commands": [],
            "attempts": 0,
        }

    # 注入记忆上下文
    runtime: RuntimeState = state["runtime"]
    memory = getattr(runtime, "_memory_manager", None)
    memory_section = ""
    if memory is not None:
        mc = memory.build_context(task)
        if mc:
            memory_section = f"\n## 用户历史偏好\n{mc}\n"

    llm = get_model_for_node("planner")
    prompt = (
        f"{PLANNER_PROMPT}\n{memory_section}{task}\n\n"
        f"请输出 JSON 计划。"
    )

    try:
        response = _invoke_with_retry(llm,[HumanMessage(content=prompt)])
        plan = _extract_json(str(response.content))
    except Exception:
        plan = {}

    todos = plan.get("todos", [{"id": "1", "content": task, "status": "pending"}])
    criteria = plan.get("acceptance_criteria", [])
    commands = plan.get("verification_commands", [])
    summary = plan.get("plan_summary", task)

    return {
        "todos": todos,
        "plan_summary": summary,
        "acceptance_criteria": criteria,
        "verification_commands": commands,
        "attempts": state.get("attempts", 0),
    }


def reflector_node(state: AgentState) -> dict[str, Any]:
    """反思节点: 检查 Agent 执行结果是否满足验收标准。

    分析:
        1. Todo 完成情况
        2. 最终回答质量
        3. 验收标准达成情况

    Args:
        state: 当前图状态。

    Returns:
        包含 passed/verifier_summary/attempts 的更新 dict。
    """
    task: str = state.get("task", "")
    todos = state.get("todos", [])
    criteria = state.get("acceptance_criteria", [])
    plan_summary = state.get("plan_summary", "")

    messages: list = list(state["messages"])

    # 检查是否使用过工具
    had_tool_calls = False
    for msg in messages:
        if getattr(msg, "tool_calls", None):
            had_tool_calls = True
            break

    # 提取最终回答
    final_answer = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and getattr(msg, "tool_calls", None) is None:
            final_answer = str(content)
            break

    # 简单对话（无工具调用 + 有回复）→ 自动通过
    if not had_tool_calls and final_answer.strip():
        return {
            "passed": True,
            "verifier_summary": "简单对话，直接通过。",
            "attempts": state.get("attempts", 0) + 1,
        }

    # 注入记忆上下文（可参考历史偏好判断回答质量）
    runtime_ref: RuntimeState = state["runtime"]
    memory_ref = getattr(runtime_ref, "_memory_manager", None)
    memory_section_ref = ""
    if memory_ref is not None:
        mc = memory_ref.build_context(task)
        if mc:
            memory_section_ref = f"\n## 用户历史偏好（用于判断回答是否符合用户预期）\n{mc}\n"

    llm = get_model_for_node("reflector")
    prompt = (
        f"{REFLECTOR_PROMPT}\n"
        f"- 原始任务: {task}\n"
        f"- 执行计划: {plan_summary}\n"
        f"- 子任务: {json.dumps(todos, ensure_ascii=False)}\n"
        f"- 验收标准: {json.dumps(criteria, ensure_ascii=False)}\n"
        f"- 最终回答: {final_answer[:1500]}\n"
        f"{memory_section_ref}"
        f"请反思并输出 JSON 结论。"
    )

    try:
        response = _invoke_with_retry(llm,[HumanMessage(content=prompt)])
        verdict = _extract_json(str(response.content))
    except Exception:
        verdict = {}

    # FIX: JSON 解析失败时默认不通过，宁重试不错过
    passed = verdict.get("passed", False)
    if not verdict:
        reason = "反射器未返回有效 JSON，自动判定为未通过。"
    else:
        reason = verdict.get("reason", "")
    checks = verdict.get("checks", [])
    suggestion = verdict.get("recommended_next_instruction", "")

    attempts = state.get("attempts", 0) + 1

    result: dict[str, Any] = {
        "passed": passed,
        "verifier_summary": reason,
        "attempts": attempts,
    }

    if passed:
        # 通过: 记录反思结果
        result["messages"] = [
            AIMessage(content=f"[反思结果] 通过: {reason}")
        ]
    else:
        # 未通过: 清空中间消息，保留系统提示，注入新的重试指令
        from langgraph.graph.message import RemoveMessage, REMOVE_ALL_MESSAGES

        retry_instruction = (
            f"上一轮执行未通过反思检查。\n\n"
            f"原始任务: {task}\n\n"
            f"问题: {reason}\n\n"
            f"修正要求: {suggestion}\n\n"
            f"重要提示: 如果你已经获取了相关信息，"
            f"请直接基于已有信息修正回答，不要重复调用工具。"
            f"例如搜索结果已是英文，直接翻译/整理为中文即可，不要再搜索。"
            f"最终回答必须为中文。"
        )
        result["messages"] = [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 清空所有历史消息
            HumanMessage(content=retry_instruction),  # 注入新指令
        ]

    return result


def reflector_route(state: AgentState) -> str:
    """反思后路由: 决定结束还是重试。

    Args:
        state: 当前图状态。

    Returns:
        "__end__" — 通过或超过最大尝试次数。
        "planner" — 未通过，重新规划。
    """
    # 注意: passed 默认 False (解析失败时宁重试不错过)
    if state.get("passed", False):
        return "__end__"
    if state.get("attempts", 0) >= state.get("max_attempts", 3):
        return "__end__"
    # 未通过 → 回到 agent 重试 (planner 已有计划会跳过)
    return "agent"


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象。

    使用非贪婪匹配 + 大括号深度追踪作为后备。

    Args:
        text: LLM 原始输出文本。

    Returns:
        解析出的 dict，解析失败返回空 dict。
    """
    import re

    # 1) 尝试匹配 ```json ... ``` 代码块
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1)

    # 2) 非贪婪匹配第一个 JSON 对象
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 3) 后备: 大括号深度追踪，找到配对的 }
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    return {}


def _is_simple_task(task: str) -> bool:
    """判断是否为简单任务 (跳过详细规划)。

    Args:
        task: 任务描述。

    Returns:
        True 表示简单任务，不需要拆解。
    """
    simple_patterns = ["你好", "谢谢", "再见", "hi", "hello", "bye", "早上好", "晚上好"]
    task_lower = task.strip().lower()
    if any(p in task_lower for p in simple_patterns) and len(task) < 20:
        return True
    # 其他任务交由 LLM 判断复杂度
    return False


# 导入放在末尾以避免循环引用
from researchagent.prompts.agent import PLANNER_PROMPT, REFLECTOR_PROMPT  # noqa: E402
