"""KorinAgentFlow CLI —— 基于 Typer 的命令行入口。

提供 `researchagent` 命令及其子命令。

用法:
    researchagent                              # 显示帮助
    researchagent test                         # LLM 连接测试
    researchagent test "自定义提示词"           # 使用自定义提示词测试
    researchagent tool-test -t calculator "2+3"  # 测试计算器
    researchagent tool-test -t bash "dir"        # 测试 Bash
    researchagent tool-test -t search "Python"   # 测试搜索
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

# 修复 Windows GBK 终端无法输出 emoji/中文等 Unicode 字符的问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ⚠️ 必须在导入 researchagent 之前检测缓存并设置 HF_HUB_OFFLINE
# researchagent.providers 会导入 langchain_openai，可能触发 huggingface_hub 初始化
_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
if (_cache_dir / "snapshots").is_dir():
    os.environ["HF_HUB_OFFLINE"] = "1"

from researchagent.core.state import RuntimeState  # noqa: E402
from researchagent.providers.openai_provider import create_model  # noqa: E402
from researchagent.core.logging_config import setup_logging  # noqa: E402

setup_logging()

app = typer.Typer(
    name="researchagent",
    help="KorinAgentFlow: LLM-powered agent workflow built with LangGraph.",
    invoke_without_command=True,
)

console = Console(legacy_windows=False)


@app.callback()
def main(
    ctx: typer.Context,
) -> None:
    """主回调 —— 无子命令时显示帮助信息。"""
    if ctx.invoked_subcommand is not None:
        return
    console.print(ctx.get_help())
    raise typer.Exit()


@app.command("test")
def test_command(
    prompt: Annotated[
        str,
        typer.Argument(
            help="发送给 LLM 的测试提示词。默认为简单的英文问候。"
        ),
    ] = "Hello! Please respond with a brief greeting.",
) -> None:
    """测试 LLM 连接：发送提示词并打印模型响应。

    此命令会：
    1. 从 .env 文件加载 LLM 配置
    2. 创建 ChatOpenAI 实例
    3. 发送提示词并流式输出响应
    4. 在成功时显示绿色成功标记，失败时显示红色错误信息
    """
    console.print()
    console.print("[bold cyan]--- KorinAgentFlow LLM Connection Test ---[/bold cyan]")
    console.print()

    try:
        model = create_model()
        console.print("[green][OK] Provider initialized successfully.[/green]")
        console.print(f"[dim]  Model: {model.model_name}[/dim]")
        console.print(f"[dim]  Prompt: {prompt}[/dim]")
        console.print()

        with console.status("[bold yellow]Waiting for LLM response...[/bold yellow]"):
            response = model.invoke(prompt)

        console.print("[bold]Response:[/bold]")
        console.print(f"  {response.content}")
        console.print()
        console.print("[green][OK] Test passed! LLM connection is working.[/green]")

    except Exception as e:
        console.print(f"[red][FAIL] Test failed: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command("tool-test")
def tool_test_command(
    input_value: Annotated[
        str,
        typer.Argument(help="工具的输入值，如数学表达式、shell 命令或搜索关键词。"),
    ] = "",
    tool: Annotated[
        str,
        typer.Option(
            "--tool", "-t",
            help="要测试的工具: calculator (计算器), bash (命令执行), search (网络搜索)。",
        ),
    ] = "calculator",
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout",
            help="Bash 工具的超时秒数 (仅对 bash 有效)。",
        ),
    ] = None,
) -> None:
    """单独测试指定工具。用于开发调试和工具验证。

    工具列表:
        calculator: 安全数学表达式求值
        bash:       Shell 命令执行 (Windows cmd)
        search:     Tavily 网络搜索 (需配置 TAVILY_API_KEY)
    """
    tool = tool.lower().strip()
    valid_tools = {"calculator", "bash", "search"}

    if tool not in valid_tools:
        console.print(f"[red]未知工具: {tool}。请选择: {', '.join(sorted(valid_tools))}[/red]")
        raise typer.Exit(code=1)

    # 创建最小 RuntimeState (使用当前目录作为 workspace)
    state = RuntimeState(workspace=Path.cwd())

    console.print()
    console.print(f"[bold cyan]--- Tool Test: {tool} ---[/bold cyan]")

    try:
        if tool == "calculator":
            _test_calculator(input_value)
        elif tool == "bash":
            _test_bash(state, input_value, timeout)
        elif tool == "search":
            _test_search(input_value)

    except Exception as e:
        console.print(f"[red][FAIL] {e}[/red]")
        raise typer.Exit(code=1) from e


def _test_calculator(expression: str) -> None:
    """测试 CalculatorTool。"""
    from researchagent.tools.calculator_tool import evaluate

    if not expression:
        expression = "2 + 3 * 4"

    console.print(f"[dim]  Expression: {expression}[/dim]")
    result = evaluate(expression)

    if result["ok"]:
        console.print(f"[green]  Result: {result['result']}[/green]")
        console.print("[green][OK] Calculator test passed.[/green]")
    else:
        console.print(f"[red]  Error: {result['error']}[/red]")
        console.print("[red][FAIL] Calculator test failed.[/red]")


def _test_bash(state: RuntimeState, command: str, timeout: int | None) -> None:
    """测试 BashTool。"""
    from researchagent.tools.bash_tool import run_bash

    if not command:
        # 默认使用平台对应的目录列表命令
        import platform
        command = "dir" if platform.system().lower() == "windows" else "ls"

    console.print(f"[dim]  Command: {command}[/dim]")
    console.print(f"[dim]  Workspace: {state.workspace}[/dim]")
    console.print()

    result = run_bash(state, command, timeout_seconds=timeout)

    console.print(f"[bold]Exit Code:[/bold] {result.get('exit_code')}")
    console.print(f"[bold]Duration:[/bold] {result.get('duration_ms')}ms")
    console.print(f"[bold]Timed Out:[/bold] {result.get('timed_out', False)}")

    stdout = result.get("stdout", "")
    if stdout:
        console.print("[bold]stdout:[/bold]")
        console.print(f"  {stdout}")
    if result.get("stdout_truncated"):
        console.print(f"[dim]  (output truncated, full log: {result.get('stdout_path')})[/dim]")

    stderr = result.get("stderr", "")
    if stderr:
        console.print("[bold yellow]stderr:[/bold yellow]")
        console.print(f"  {stderr}")

    if result.get("ok"):
        console.print("[green][OK] Bash test passed.[/green]")
    else:
        console.print(f"[red][FAIL] Bash test failed: {result.get('error', 'exit_code != 0')}[/red]")


def _test_search(query: str) -> None:
    """测试 WebSearchTool。"""
    from researchagent.tools.web_search_tool import web_search

    if not query:
        query = "Python programming"

    console.print(f"[dim]  Query: {query}[/dim]")
    console.print()

    with console.status("[bold yellow]Searching web...[/bold yellow]"):
        result = web_search(query)

    if result["ok"]:
        answer = result.get("answer")
        if answer:
            console.print("[bold]AI Summary:[/bold]")
            console.print(f"  {answer}")
            console.print()

        results = result.get("results", [])
        console.print(f"[bold]Results ({len(results)}):[/bold]")
        for i, r in enumerate(results, 1):
            console.print(f"  {i}. [bold]{r['title']}[/bold]")
            console.print(f"     [dim]{r['url']}[/dim]")
            console.print(f"     {r['content'][:200]}...")
        console.print()
        console.print("[green][OK] Search test passed.[/green]")
    else:
        console.print(f"[red]  Error: {result.get('error')}[/red]")
        console.print("[red][FAIL] Search test failed.[/red]")


@app.command("run")
def run_command(
    task: Annotated[
        str,
        typer.Argument(help="要执行的任务描述。支持中文。"),
    ],
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations", "-n",
            help="Agent 最大迭代次数（默认 10）。",
        ),
    ] = 10,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature", "-t",
            help="LLM 采样温度 (0.0-2.0)。",
        ),
    ] = 0.0,
    no_stream: Annotated[
        bool,
        typer.Option(
            "--no-stream",
            help="禁用流式输出，仅显示最终结果。",
        ),
    ] = False,
) -> None:
    """运行 ReAct Agent，使用工具完成任务。

    示例:
        researchagent run "计算 123 * 456"
        researchagent run "搜索 Python 最新版本并总结"
        researchagent run "列出当前目录的文件"
        researchagent run "帮我查 LangGraph 是什么，然后算 2^10"
    """

    from researchagent.graph import build_agent_graph

    from researchagent.core.tracker import TokenTracker

    state = RuntimeState(workspace=Path.cwd())
    tracker = TokenTracker()
    object.__setattr__(state, "_token_tracker", tracker)
    graph = build_agent_graph()

    initial_state = {
        "messages": [],
        "runtime": state,
        "task": task,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "todos": [],
        "plan_summary": "",
        "acceptance_criteria": [],
        "verification_commands": [],
        "attempts": 0,
        "max_attempts": 3,
        "passed": False,
        "verifier_summary": "",
    }

    console.print()
    console.print("[bold cyan]--- KorinAgentFlow Agent ---[/bold cyan]")
    console.print(f"[dim]Task: {task}[/dim]")
    console.print(f"[dim]Max iterations: {max_iterations}[/dim]")
    console.print()

    try:
        if no_stream:
            result = graph.invoke(initial_state)
            _display_final_result(result)
        else:
            _stream_run(graph, initial_state)

    except Exception as e:
        console.print(f"[red][FAIL] Agent error: {e}[/red]")
        raise typer.Exit(code=1) from e


def _stream_run(graph, initial_state: dict) -> None:
    """流式执行并逐步渲染过程。"""
    import json

    step = 0
    final_content = ""

    for mode, chunk in graph.stream(
        initial_state,
        stream_mode=["updates"],
    ):
        if chunk is None:
            continue

        if mode == "updates":
            step += 1
            for node_name, node_update in chunk.items():
                if node_update is None:
                    continue
                messages = node_update.get("messages", [])

                if node_name == "agent":
                    for msg in messages:
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        content = getattr(msg, "content", "")

                        if tool_calls:
                            for tc in tool_calls:
                                console.print(
                                    f"[bold yellow][Step {step}][/bold yellow] "
                                    f"[bold]Tool Call:[/bold] {tc.get('name')}"
                                )
                                args = tc.get("args", {})
                                if args:
                                    console.print(f"  [dim]Args: {json.dumps(args, ensure_ascii=False)}[/dim]")
                        elif content:
                            if len(content) > 200:
                                console.print(f"[dim]  {content[:200]}...[/dim]")
                            else:
                                console.print(f"[dim]  {content}[/dim]")
                            final_content = content

                elif node_name == "tools":
                    for msg in messages:
                        tool_name = getattr(msg, "name", "unknown")
                        tool_content = getattr(msg, "content", "")
                        short_result = tool_content[:200] + "..." if len(tool_content) > 200 else tool_content
                        console.print(
                            f"[bold yellow][Step {step}][/bold yellow] "
                            f"[bold green]Tool Result ({tool_name}):[/bold green] {short_result}"
                        )

    # 流式输出后换行
    if final_content and step > 0:
        console.print()
    # Token 用量报告
    try:
        runtime = initial_state.get("runtime") if isinstance(initial_state, dict) else None
        tracker = getattr(runtime, "_token_tracker", None) if runtime is not None else None
        if tracker is not None and tracker.usages:
            console.print()
            console.print(tracker.report())
    except Exception:
        pass
    console.print()
    console.print("[green][OK] Agent completed.[/green]")


def _display_final_result(result: dict) -> None:
    """显示非流式执行的最终结果。"""
    import json

    messages = result.get("messages", [])
    if not messages:
        console.print("[yellow]Agent returned no messages.[/yellow]")
        return

    tool_calls_seen = 0
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            tool_calls_seen += 1
            console.print(
                f"[bold yellow][Tool Call {tool_calls_seen}][/bold yellow] "
                f"{tc.get('name')}: {json.dumps(tc.get('args', {}), ensure_ascii=False)}"
            )

    last_msg = messages[-1]
    content = getattr(last_msg, "content", "")
    if content:
        console.print()
        console.print("[bold]Final Answer:[/bold]")
        console.print(f"  {content}")

    console.print()
    iterations = result.get("iteration_count", 0)
    console.print(f"[dim]Completed in {iterations} iterations, {tool_calls_seen} tool calls.[/dim]")
    console.print("[green][OK] Agent completed.[/green]")


@app.command("memory-test")
def memory_test_command(
    input_text: Annotated[
        str,
        typer.Argument(help="要添加/搜索的文本。"),
    ] = "",
    action: Annotated[
        str,
        typer.Option(
            "--add", "-a",
            help="添加一条记忆。用法: --add '我喜欢Python'。",
        ),
    ] = "",
    search: Annotated[
        str,
        typer.Option(
            "--search", "-s",
            help="搜索记忆。用法: --search '编程语言偏好'。",
        ),
    ] = "",
    pref: Annotated[
        str,
        typer.Option(
            "--pref",
            help="存储用户偏好。用法: --pref '偏好Python'。",
        ),
    ] = "",
    fact: Annotated[
        str,
        typer.Option(
            "--fact",
            help="存储重要事实。用法: --fact '用户是程序员'。",
        ),
    ] = "",
) -> None:
    """测试记忆系统：添加、搜索、查看长期记忆。

    使用方式:
        researchagent memory-test --add "我喜欢Python"
        researchagent memory-test --pref "偏好异步编程"
        researchagent memory-test --fact "项目使用LangGraph"
        researchagent memory-test --search "编程语言"
    """
    from researchagent.providers.openai_provider import create_model
    from researchagent.memory import MemoryManager

    state = RuntimeState(workspace=Path.cwd())

    console.print()
    console.print("[bold cyan]--- Memory Test ---[/bold cyan]")
    console.print()

    try:
        llm = create_model()
        manager = MemoryManager(llm, state)

        # -- 添加记忆
        if action:
            manager.add_interaction("user", action)
            console.print(f"[green][OK] Added interaction: {action}[/green]")

        if pref:
            manager.remember_preference(pref)
            console.print(f"[green][OK] Added preference: {pref}[/green]")

        if fact:
            manager.remember_fact(fact)
            console.print(f"[green][OK] Added fact: {fact}[/green]")

        # -- 搜索
        if search:
            console.print(f"[dim]Searching for: {search}[/dim]")
            results = manager.search_memories(search, k=5)
            if results:
                console.print(f"[bold]Found {len(results)} results:[/bold]")
                for i, r in enumerate(results, 1):
                    console.print(f"  {i}. [{r['category']}] {r['text']}")
                    console.print(f"     [dim]{r.get('timestamp', '')}[/dim]")
            else:
                console.print("[yellow]No results found.[/yellow]")

        # -- 查看上下文
        if not action and not pref and not fact and not search:
            ctx = manager.build_context(input_text if input_text else "")
            if ctx:
                console.print("[bold]Current Context:[/bold]")
                console.print(ctx)
            else:
                console.print("[yellow]Memory is empty. Add some interactions first.[/yellow]")

        console.print(f"[dim]Long-term memory count: {manager.memory_count()}[/dim]")

    except Exception as e:
        console.print(f"[red][FAIL] {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command("research")
def research_command(
    topic: Annotated[
        str,
        typer.Argument(help="研究课题。支持中文。"),
    ],
    max_papers: Annotated[
        int,
        typer.Option("--max-papers", "-p", help="搜索的最大论文数（默认 10）。"),
    ] = 10,
    output_dir: Annotated[
        str,
        typer.Option("--output", "-o", help="综述和 .bib 文件的输出目录。"),
    ] = "./research_output",
) -> None:
    """学术调研：搜索论文并生成文献综述 + 参考文献。"""
    from researchagent.graph import build_agent_graph
    from researchagent.core.tracker import TokenTracker
    from researchagent.research.tools.registry import build_research_tools

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    state = RuntimeState(workspace=output_path)
    object.__setattr__(state, "tool_builder", build_research_tools)

    tracker = TokenTracker()
    object.__setattr__(state, "_token_tracker", tracker)

    graph = build_agent_graph()

    task = (
        f"研究课题: {topic}\n\n"
        f"请搜索 ArXiv 和 Semantic Scholar，找到 {max_papers} 篇最相关的论文，"
        f"然后生成一份结构化的文献综述。\n\n"
        f"搜索要求:\n"
        f"1. 提取英文关键词后分别用 ArXiv 和 Semantic Scholar 搜索\n"
        f"2. 优先选择近 3 年发表的论文\n"
        f"3. 关注高被引论文\n\n"
        f"输出要求:\n"
        f"1. 综述保存为 {output_dir}/review.md\n"
        f"2. 参考文献保存为 {output_dir}/references.bib\n"
        f"3. 综述使用中文撰写，含引言、相关方法、讨论与趋势三个部分"
    )

    console.print()
    console.print("[bold cyan]--- KorinAgentFlow Research Agent ---[/bold cyan]")
    console.print(f"[dim]Topic: {topic}[/dim]")
    console.print(f"[dim]Max papers: {max_papers}[/dim]")
    console.print(f"[dim]Output: {output_path.absolute()}[/dim]")
    console.print()

    initial_state = {
        "messages": [],
        "runtime": state,
        "task": task,
        "iteration_count": 0,
        "max_iterations": 15,
        "todos": [],
        "plan_summary": "",
        "acceptance_criteria": [],
        "verification_commands": [],
        "attempts": 0,
        "max_attempts": 2,
        "passed": False,
        "verifier_summary": "",
    }

    try:
        result = graph.invoke(initial_state)
        _display_final_result(result)
        console.print()
        console.print(f"[dim]Output directory: {output_path.absolute()}[/dim]")
        if tracker.usages:
            console.print(tracker.report())
    except Exception as e:
        console.print(f"[red][FAIL] Research error: {e}[/red]")
        raise typer.Exit(code=1) from e
