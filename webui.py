"""ResearchAgent Web UI。

Usage: uv run python webui.py
Visit: http://localhost:7860
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
if (_cache_dir / "snapshots").is_dir():
    os.environ["HF_HUB_OFFLINE"] = "1"

import gradio as gr  # noqa: E402
from researchagent.graph import build_agent_graph  # noqa: E402
from researchagent.core.state import RuntimeState  # noqa: E402

logger = logging.getLogger(__name__)
HISTORY_DIR = Path(__file__).parent / ".researchagent" / "chats"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
_execution_lock = threading.Lock()

# ============================================================
# 快捷任务模板
# ============================================================
QUICK_TASKS = {
    "文献综述": "用 ArXiv 和 Semantic Scholar 搜索关于 {topic} 的近期论文（3-5篇），用中文总结每篇的核心方法和发现，最后给出领域概览",
    "趋势分析": "搜索关于 {topic} 的最新论文（5篇），分析发表年份分布、热门方法和关键词趋势",
    "方法对比": "搜索关于 {topic} 的论文（3-5篇），提取每篇的方法、数据集和指标，整理为对比分析",
    "研究空白": "搜索关于 {topic} 的近期论文，分析现有方法的局限性和未来可能的研究方向",
    "选题建议": "基于关于 {topic} 的研究现状，推荐 3 个有价值的研究选题方向",
    "论文速览": "搜索关于 {topic} 的最新论文（3篇），列出每篇的标题、作者、核心发现和被引数",
}

# ============================================================
# 会话管理
# ============================================================
def _safe_io(fn):
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            logger.warning("%s failed: %s", fn.__name__, e)
            return None
    return wrapper

@_safe_io
def list_sessions():
    files = sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.stem for f in files] or []

@_safe_io
def load_session(sid):
    path = HISTORY_DIR / f"{sid}.json"
    return json.loads(path.read_text(encoding="utf-8")).get("history", []) if path.exists() else []

@_safe_io
def save_session(sid, history):
    data = {"session_id": sid, "updated": datetime.now(timezone.utc).isoformat(), "history": history}
    (HISTORY_DIR / f"{sid}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

@_safe_io
def delete_session(sid):
    p = HISTORY_DIR / f"{sid}.json"
    if p.exists():
        p.unlink()

def clean_answer(text):
    text = re.sub(r"子任务\s*\d+\s*状态[：:].*?\n", "", text)
    text = re.sub(r"(?:\[DONE\]|\[完成\]).*?\n", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# ============================================================
# Agent 执行
# ============================================================
def execute_task(task, cancel_flag, progress=gr.Progress()):
    """执行 Agent 任务，返回 (final_answer, report_html, token_summary)"""
    progress(0.10, desc="正在初始化...")
    from researchagent.core.tracker import TokenTracker
    state = RuntimeState(workspace=Path.cwd())
    tracker = TokenTracker()
    object.__setattr__(state, "_token_tracker", tracker)
    graph = build_agent_graph()
    progress(0.20, desc="模型就绪，开始思考...")

    result = {"output": [], "error": None}
    done = threading.Event()

    def _run():
        try:
            init = {"messages": [], "runtime": state, "task": task,
                    "iteration_count": 0, "max_iterations": 8,
                    "todos": [], "plan_summary": "", "acceptance_criteria": [],
                    "verification_commands": [], "attempts": 0, "max_attempts": 1,
                    "passed": False, "verifier_summary": ""}
            for _mode, chunk in graph.stream(init, stream_mode=["updates"]):
                if cancel_flag[0]:
                    break
                result["output"].append(chunk)
        except Exception as e:
            result["error"] = str(e)
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    step = 0
    while not done.is_set():
        if cancel_flag[0]:
            done.set()
            progress(1.0, desc="已取消")
            return "任务已取消", "", {}
        step += 1
        # 基于输出步数显示真实进度
        output_count = len(result["output"])
        if output_count > 0:
            pct = 0.20 + min(output_count * 0.15, 0.72)
            progress(pct, desc=f"已完成 {output_count} 步 (LLM调用+工具执行)")
        else:
            progress(0.20, desc=f"等待 LLM 响应... ({step * 0.5:.0f}s)")
        done.wait(0.5)

    if result["error"]:
        return f"执行出错: {result['error']}", "", {}

    # ---- 解析 ----
    steps = result["output"]
    progress(0.92, desc="分析结果...")
    tool_rows, reflections, answer = [], [], ""

    for chunk in steps:
        for node_update in chunk.values():
            if not node_update:
                continue
            for msg in node_update.get("messages", []):
                tcs = getattr(msg, "tool_calls", None) or []
                for tc in tcs:
                    tool_rows.append(
                        f"<tr><td><b>{html.escape(tc.get('name',''))}</b></td>"
                        f"<td><code>{html.escape(json.dumps(tc.get('args',{}), ensure_ascii=False)[:120])}</code></td></tr>"
                    )
                content = getattr(msg, "content", "")
                if content:
                    text = str(content)
                    if text.startswith("[反思"):
                        reflections.append(f"<li>{html.escape(text)}</li>")
                    elif not tcs:
                        answer = text

    # 报告
    report_parts = [
        f"<details><summary><b>执行报告</b> ({len(steps)}步)</summary>",
        f"<table><tr><td>步数</td><td>{len(steps)}</td></tr><tr><td>工具调用</td><td>{len(tool_rows)}</td></tr></table>",
    ]
    if tracker.usages:
        s = tracker.summary()
        report_parts.append(f"<br><b>Token:</b> {s['total']:,} ({s['total_calls']}次)")
    if tool_rows:
        report_parts.append("<br><b>工具调用:</b><table>" + "".join(tool_rows) + "</table>")
    if reflections:
        report_parts.append("<br><b>反思:</b><ul>" + "".join(reflections) + "</ul>")
    report_parts.append("</details>")
    report = "\n".join(report_parts)

    # Token 摘要
    token_summary = {}
    if tracker.usages:
        s = tracker.summary()
        token_summary = {"total": s["total"], "calls": s["total_calls"]}

    progress(1.0, desc="完成")
    return clean_answer(answer) or "(无响应)", report, token_summary


# ============================================================
# UI 事件处理
# ============================================================
def handle_send(user_msg, history, cancel, topic, sid):
    """普通发送消息——使用已有 session ID，不新建"""
    if not user_msg.strip():
        return history, "", cancel, sid, gr.update()
    if _execution_lock.locked():
        history.append({"role": "assistant", "content": "上一个任务仍在运行中，请等待完成"})
        return history, "", cancel, sid, gr.update()

    _execution_lock.acquire()
    try:
        cancel[0] = False
        t = topic.strip() or "artificial intelligence"
        msg = user_msg.replace("{topic}", t).replace("{TOPIC}", t)

        history.append({"role": "user", "content": msg})
        answer, report, _ = execute_task(msg, cancel, gr.Progress())
        history.append({"role": "assistant", "content": answer})

        if not sid:
            sid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        save_session(sid, history)
        return history, report, cancel, sid, gr.update(choices=list_sessions())
    finally:
        _execution_lock.release()


def handle_quick(label, history, cancel, topic, sid):
    """快捷按钮——使用已有 session ID"""
    t = topic.strip() or "artificial intelligence"
    task = QUICK_TASKS.get(label, QUICK_TASKS["文献综述"]).replace("{topic}", t)

    if _execution_lock.locked():
        history.append({"role": "assistant", "content": "上一个任务仍在运行中，请等待完成"})
        return history, "", cancel, sid, gr.update()

    _execution_lock.acquire()
    try:
        cancel[0] = False
        history.append({"role": "user", "content": f"[{label}] {t}"})
        answer, report, _ = execute_task(task, cancel, gr.Progress())
        history.append({"role": "assistant", "content": answer})

        if not sid:
            sid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        save_session(sid, history)
        return history, report, cancel, sid, gr.update(choices=list_sessions())
    finally:
        _execution_lock.release()


def handle_cancel(cancel):
    cancel[0] = True
    return cancel


def handle_load(sid):
    if sid:
        hist = load_session(sid) or []
        return hist, "", sid, gr.update()
    return [], "", "", gr.update()


def handle_new():
    return [], "", datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"), gr.update()


def handle_delete(sid, confirmed, hist, rep):
    if not sid or not confirmed:
        return hist, rep, sid, gr.update(), False
    delete_session(sid)
    return [], "", "", gr.update(choices=list_sessions(), value=None), False


# ============================================================
# UI 布局
# ============================================================
CSS = """
.chat-area { min-height: 480px; }
.report-area { font-size: 13px; margin-top: 4px; }
footer { display: none !important; }
"""

with gr.Blocks(title="ResearchAgent") as app:
    gr.Markdown("# ResearchAgent\n### 学术调研 — 论文搜索 · 综述 · 分析")

    session_state = gr.State("")
    cancel_state = gr.State([False])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 研究主题")
            topic_input = gr.Textbox(
                label="输入研究主题，快捷按钮会自动使用",
                placeholder="例如: Transformer efficiency optimization",
                lines=2,
            )

            gr.Markdown("### 快捷任务")
            quick_labels = list(QUICK_TASKS.keys())
            btn1 = gr.Button(quick_labels[0], size="sm")
            btn2 = gr.Button(quick_labels[1], size="sm")
            btn3 = gr.Button(quick_labels[2], size="sm")
            btn4 = gr.Button(quick_labels[3], size="sm")
            btn5 = gr.Button(quick_labels[4], size="sm")
            btn6 = gr.Button(quick_labels[5], size="sm")

            gr.Markdown("---")
            max_iter = gr.Slider(1, 20, value=10, step=1, label="最大迭代")

            gr.Markdown("---")
            gr.Markdown("### 会话历史")
            session_radio = gr.Radio(label="选择会话", choices=list_sessions(), value=None)
            with gr.Row():
                load_btn = gr.Button("加载", size="sm")
                new_btn = gr.Button("新对话", size="sm")
            with gr.Row():
                confirm_check = gr.Checkbox(label="确认删除", value=False)
                del_btn = gr.Button("删除", size="sm", variant="stop")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话", elem_classes=["chat-area"], height=520)
            report_html = gr.HTML(label="执行报告", elem_classes=["report-area"])

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="输入任务...",
                    scale=4,
                    show_label=False,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)
                cancel_btn = gr.Button("取消", variant="stop", scale=1)

    # ==== 事件绑定 ====
    inputs_send = [msg_input, chatbot, cancel_state, topic_input, session_state]
    outputs_send = [chatbot, report_html, cancel_state, session_state, session_radio]

    send_btn.click(handle_send, inputs_send, outputs_send).then(
        lambda: "", None, msg_input
    )

    msg_input.submit(handle_send, inputs_send, outputs_send).then(
        lambda: "", None, msg_input
    )

    # 快捷按钮
    inputs_quick = [chatbot, cancel_state, topic_input, session_state]
    outputs_quick = [chatbot, report_html, cancel_state, session_state, session_radio]

    # 每个快捷按钮用闭包捕获 label
    btn1.click(
        lambda hist, canc, top, sid: handle_quick(quick_labels[0], hist, canc, top, sid),
        inputs_quick, outputs_quick
    )
    btn2.click(
        lambda hist, canc, top, sid: handle_quick(quick_labels[1], hist, canc, top, sid),
        inputs_quick, outputs_quick
    )
    btn3.click(
        lambda hist, canc, top, sid: handle_quick(quick_labels[2], hist, canc, top, sid),
        inputs_quick, outputs_quick
    )
    btn4.click(
        lambda hist, canc, top, sid: handle_quick(quick_labels[3], hist, canc, top, sid),
        inputs_quick, outputs_quick
    )
    btn5.click(
        lambda hist, canc, top, sid: handle_quick(quick_labels[4], hist, canc, top, sid),
        inputs_quick, outputs_quick
    )
    btn6.click(
        lambda hist, canc, top, sid: handle_quick(quick_labels[5], hist, canc, top, sid),
        inputs_quick, outputs_quick
    )

    cancel_btn.click(handle_cancel, cancel_state, cancel_state)
    load_btn.click(handle_load, session_radio, [chatbot, report_html, session_state, session_radio])
    new_btn.click(handle_new, None, [chatbot, report_html, session_state, session_radio])
    del_btn.click(
        handle_delete,
        [session_radio, confirm_check, chatbot, report_html],
        [chatbot, report_html, session_state, session_radio, confirm_check],
    )


if __name__ == "__main__":
    from researchagent.core.logging_config import setup_logging  # noqa: E402
    setup_logging()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft(), css=CSS)
