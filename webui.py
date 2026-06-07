"""ResearchAgent Chat Web UI.

Usage: uv run python webui.py
Visit: http://localhost:7860
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# 必须在 import gradio 之前检测 BGE-M3 缓存并启用离线模式
_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
if (_cache_dir / "snapshots").is_dir():
    os.environ["HF_HUB_OFFLINE"] = "1"

import gradio as gr  # noqa: E402

from researchagent.graph import build_agent_graph  # noqa: E402
from researchagent.core.state import RuntimeState  # noqa: E402

HISTORY_DIR = Path(__file__).parent / ".researchagent" / "chats"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# ---- 会话管理 ----

def list_sessions() -> list[str]:
    files = sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.stem for f in files]


def load_session(session_id: str) -> list[dict]:
    """加载会话，返回 Chatbot 消息格式 [{"role": ..., "content": ...}, ...]."""
    path = HISTORY_DIR / f"{session_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("history", [])


def delete_session(session_id: str) -> bool:
    path = HISTORY_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def clean_response(text: str) -> str:
    """去除回复中的内部状态信息（如子任务状态、执行计划等）。"""
    import re
    # 去掉 "子任务 X 状态：..." 这类行
    text = re.sub(r"子任务\s*\d+\s*状态[：:]\s*[✅✓].*?\n", "", text)
    text = re.sub(r"子任务\s*\d+\s*状态[：:]\s*[❌✗].*?\n", "", text)
    # 去掉空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_session(session_id: str, history: list[dict]) -> None:
    path = HISTORY_DIR / f"{session_id}.json"
    data = {
        "session_id": session_id,
        "updated": datetime.now(timezone.utc).isoformat(),
        "history": history,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---- Agent 执行 ----

def run_agent(
    user_message: str,
    history: list[dict],
    max_iterations: int,
    cancel_flag: list = None,
    progress=gr.Progress(),
) -> tuple[list[dict], str]:
    """运行 Agent，返回 (对话历史, 执行报告HTML)。

    cancel_flag: 可变容器 [bool]，设为 [True] 可中断执行。
    """

    if cancel_flag is None:
        cancel_flag = [False]

    if not user_message.strip():
        return history, ""

    # 初始化
    progress(0.05, desc="正在初始化...")
    from researchagent.core.tracker import TokenTracker

    state = RuntimeState(workspace=Path.cwd())
    tracker = TokenTracker()
    object.__setattr__(state, "_token_tracker", tracker)
    graph = build_agent_graph()
    progress(0.15, desc="模型就绪，执行中...")

    # 后台线程执行
    result_container = {"output": None, "error": None}
    stream_done = threading.Event()

    def _execute():
        try:
            init = {
                "messages": [],
                "runtime": state,
                "task": user_message,
                "iteration_count": 0,
                "max_iterations": max_iterations,
                "todos": [],
                "plan_summary": "",
                "acceptance_criteria": [],
                "verification_commands": [],
                "attempts": 0,
                "max_attempts": 2,
                "passed": False,
                "verifier_summary": "",
            }
            steps = []
            for _mode, chunk in graph.stream(init, stream_mode=["updates"]):
                steps.append(chunk)
            result_container["output"] = steps
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            stream_done.set()

    thread = threading.Thread(target=_execute, daemon=True)
    thread.start()

    wait_step = 0
    while not stream_done.is_set():
        if cancel_flag[0]:
            stream_done.set()  # 停止轮询
            progress(1.0, desc="已取消")
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "⏹ 任务已取消"})
            return history, ""
        wait_step += 1
        dots = "." * (wait_step % 4 + 1)
        pct = 0.15 + min(wait_step, 10) * 0.03 + max(0, wait_step - 10) * 0.01
        pct = min(pct, 0.92)
        progress(pct, desc=f"执行中 ({(wait_step+1)//2}s){dots}")
        thread.join(timeout=0.5)

    if result_container["error"]:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": f"Error: {result_container['error']}"})
        return history, ""

    steps_data = result_container["output"] or []
    progress(0.92, desc="处理结果中...")

    # ---- 解析结果 ----
    tool_rows: list[str] = []
    reflection_items: list[str] = []
    final_answer = ""
    step_count = len(steps_data)

    for chunk in steps_data:
        for _node_name, node_update in chunk.items():
            if node_update is None:
                continue
            messages = node_update.get("messages", [])
            for msg in messages:
                tcs = getattr(msg, "tool_calls", None) or []
                content = getattr(msg, "content", "")

                for tc in tcs:
                    args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
                    tool_rows.append(
                        f"<tr><td><b>{tc.get('name')}</b></td>"
                        f"<td><code>{args_str[:120]}</code></td></tr>"
                    )

                if content:
                    text = str(content)
                    if text.startswith("[反思结果]"):
                        reflection_items.append(f"<li>{text}</li>")
                    elif not tcs:
                        final_answer = text

    # ---- 对话历史 (dict 格式) ----
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": clean_response(final_answer) or "(无响应)"})

    # ---- 执行报告 ----
    report_parts = [
        "<details><summary><b>📋 执行报告</b> (点击展开)</summary>",
        "<table>",
        f"<tr><td>执行步数</td><td>{step_count}</td></tr>",
        f"<tr><td>工具调用</td><td>{len(tool_rows)}</td></tr>",
        "</table>",
    ]

    # Token 用量
    tracker = getattr(state, "_token_tracker", None)
    if tracker is not None and tracker.usages:
        s = tracker.summary()
        report_parts.append(f"<br><b>💰 Token 用量:</b> 总计 {s['total']:,} (输入 {s['total_prompt']:,} / 输出 {s['total_completion']:,}) — {s['total_calls']} 次调用<br>")

    if tool_rows:
        report_parts.append("<br><b>🔧 工具调用:</b><table>")
        report_parts.extend(tool_rows)
        report_parts.append("</table>")

    if reflection_items:
        report_parts.append("<br><b>🔍 反思检查:</b><ul>")
        report_parts.extend(reflection_items)
        report_parts.append("</ul>")

    report_parts.append("</details>")
    report = "\n".join(report_parts)

    progress(1.0, desc="完成")
    return history, report


# ---- UI ----

CSS = """
.chat-area { min-height: 500px; }
.report-area { font-size: 13px; margin-top: 8px; }
footer { display: none !important; }
"""

with gr.Blocks(title="ResearchAgent") as app:
    gr.Markdown("# 🔬 ResearchAgent\n学术调研 Agent — 论文搜索 + 分析 + 综述生成")

    session_state = gr.State("")
    cancel_state = gr.State([False])

    with gr.Row():
        # ---- 左侧面板 ----
        with gr.Column(scale=1):
            gr.Markdown("### 💬 会话")
            session_radio = gr.Radio(
                label="历史记录",
                choices=list_sessions(),
                value=None,
            )
            with gr.Row():
                load_btn = gr.Button("📂 加载", size="sm")
                new_btn = gr.Button("🆕 新对话", size="sm")
            with gr.Row():
                confirm_check = gr.Checkbox(label="确认删除", value=False)
                del_btn = gr.Button("🗑 删除选中", size="sm", variant="stop")

            gr.Markdown("### ⚙️ 设置")
            max_iter = gr.Slider(1, 20, value=10, step=1, label="最大迭代次数")
            gr.Markdown(
                """
                ### 🛠 工具
                - 🔢 计算器
                - 💻 Bash 命令
                - 🔍 网络搜索
                """
            )

        # ---- 右侧: 对话区 ----
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话",
                elem_classes=["chat-area"],
                height=500,
            )
            report_html = gr.HTML(
                label="执行详情",
                elem_classes=["report-area"],
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="输入任务... 例如: 计算 123 * 456",
                    scale=4,
                    show_label=False,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)
                cancel_btn = gr.Button("取消", variant="stop", scale=1)

    # ---- 事件处理 ----

    def _on_send(msg, hist, max_it, sid, cancel):
        cancel[0] = False  # 重置取消标志
        new_hist, report = run_agent(msg, hist, max_it, cancel, gr.Progress())
        if not sid:
            sid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return new_hist, report, sid, cancel

    def _on_new():
        sid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return [], "", sid

    def _on_cancel(cancel):
        cancel[0] = True
        return cancel

    def _on_load(sid):
        if not sid:
            return [], "", ""
        history = load_session(sid)
        return history, "", sid

    def _on_save(history, sid):
        if history and sid:
            save_session(sid, history)
            return gr.Radio(choices=list_sessions())
        return gr.Radio(choices=list_sessions())

    send_btn.click(
        fn=_on_send,
        inputs=[msg_input, chatbot, max_iter, session_state, cancel_state],
        outputs=[chatbot, report_html, session_state, cancel_state],
    ).then(lambda: "", None, msg_input).then(
        _on_save, [chatbot, session_state], session_radio,
    )

    msg_input.submit(
        fn=_on_send,
        inputs=[msg_input, chatbot, max_iter, session_state, cancel_state],
        outputs=[chatbot, report_html, session_state, cancel_state],
    ).then(lambda: "", None, msg_input).then(
        _on_save, [chatbot, session_state], session_radio,
    )

    cancel_btn.click(
        fn=_on_cancel,
        inputs=cancel_state,
        outputs=cancel_state,
    )

    def _on_delete(sid, confirmed):
        if not sid:
            return [], "", "", gr.Radio(choices=list_sessions()), False
        if not confirmed:
            return [], "", sid, gr.Radio(choices=list_sessions()), False
        delete_session(sid)
        choices = list_sessions()
        return [], "", "", gr.Radio(choices=choices, value=None), False

    load_btn.click(_on_load, session_radio, [chatbot, report_html, session_state])
    del_btn.click(
        _on_delete,
        [session_radio, confirm_check],
        [chatbot, report_html, session_state, session_radio, confirm_check],
    )
    new_btn.click(_on_new, None, [chatbot, report_html, session_state])

    gr.Examples(
        examples=[
            ["计算 (123 + 456) * 789"],
            ["搜索 LangGraph 是什么并用中文总结"],
            ["列出当前项目的所有 Python 文件"],
            ["你好"],
        ],
        inputs=msg_input,
    )

if __name__ == "__main__":
    from researchagent.core.logging_config import setup_logging  # noqa: E402
    setup_logging()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CSS,
    )
