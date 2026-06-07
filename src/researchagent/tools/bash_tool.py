"""Shell 命令执行工具 (Bash Tool)。

为 Agent 提供执行 Shell 命令的能力。参考 MokioAgent 的 `bash_tool.py` 简化而来：
    - 保留: 平台检测、命令适配、危险命令拦截、超时控制、输出截断
    - 移除: 审批系统、后台执行、shim 工具链、env 文件注入 (后续 Step 补充)

安全性:
    - 工作目录限制在 RuntimeState.workspace 内
    - 危险命令 (rm -rf, format, shutdown 等) 直接拒绝
    - 超时和输出截断防止资源滥用
"""

from __future__ import annotations

import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from researchagent.core.state import RuntimeState

# ---- 常量 ----

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
MAX_OUTPUT_CHARS = 6000

# 危险命令正则模式 (Windows + Unix)
DANGEROUS_PATTERNS: list[str] = [
    r"\brm\s+-rf\b",                       # Unix 递归强制删除
    r"\bRemove-Item\b.*\b-Recurse\b",       # PowerShell 递归删除
    r"\bdel\s+/[sq]\b",                     # cmd 静默/递归删除
    r"\bformat\b",                           # 格式化磁盘
    r"\bshutdown\b",                        # 关机
    r"\breboot\b",                          # 重启
    r"\bchmod\s+777\b",                     # 危险权限
    r">\s*(?:[A-Za-z]:\\|/(?!dev/null\b))",  # 向根目录写入
    r"\bdocker\s+rm\b",                     # 删除 Docker 容器
    r"\bgit\s+push\s+(-f|--force)\b",       # 强制推送
    r"\bcurl\b.*\|\s*(?:ba)?sh\b",          # curl pipe to shell
    r"\bwget\b.*\|\s*(?:ba)?sh\b",          # wget pipe to shell
    r">\s*/dev/sd[a-z]",                     # 直接写入磁盘
    r"\bmkfs\.",                              # 创建文件系统
    r"\bdd\s+if=",                           # dd 磁盘操作
    r":\(\)\s*\{[^}]*:\|:&\s*\}[^}]*;:",    # fork bomb
    r"\beval\s",                             # eval 执行
    r"\bnc\s+-[lL]\s",                      # netcat 监听
    r"\bsocat\s",                            # socat
    r"\bchmod\s+[0-7]*7[0-7]*7\b",          # 更宽泛的危险权限
]


# ---- 平台适配 ----

def bash_tool_description() -> str:
    """根据当前平台生成 BashTool 的 LLM 描述文本。

    返回包含通用说明 + 平台特定提示的描述字符串。
    """
    system = platform.system().lower()
    common = (
        "Run a safe shell command inside the workspace with timeout and output capture. "
        "Each call starts a fresh shell; environment variables do not persist between calls. "
        "Prefer cross-platform Python one-liners when possible. "
    )
    if system == "windows":
        return (
            f"{common}"
            f"Current platform: Windows (cmd.exe). "
            f"Use Windows commands: dir (not ls), type (not cat), findstr (not grep), "
            f"mkdir (not mkdir -p), del (not rm), copy (not cp), move (not mv). "
        )
    return (
        f"{common}"
        f"Current platform: {platform.system()}. "
        f"Commands are executed by a POSIX shell."
    )


def _looks_dangerous(command: str) -> str | None:
    """检查命令是否匹配危险模式。

    Args:
        command: 待检查的原始命令字符串。

    Returns:
        如果安全则返回 None；如果匹配危险模式则返回模式描述。
    """
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"命令匹配危险模式: {pattern}"
    return None


def _normalize_command(command: str) -> str:
    """根据平台对命令做基本适配。

    在 Windows 上将常见的 Unix 命令翻译为对应的 cmd 命令。
    """
    if platform.system().lower() != "windows":
        return command

    replacements = {
        "python3 ": "python ",
        "python3\n": "python\n",
        " ls ": " dir ",
        " cat ": " type ",
    }
    result = command
    for old, new in replacements.items():
        # 仅在命令开头或空格分隔的词位置替换
        result = result.replace(old, new)
    return result


def _decode_output(data: bytes | str | None) -> str:
    """将 subprocess 输出解码为字符串，尝试多种编码。"""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    for encoding in ("utf-8", "utf-8-sig", "gbk", "cp936"):
        try:
            return data.decode(encoding, errors="replace")
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _truncate_output(
    stdout: str,
    stderr: str,
    max_chars: int,
    workspace: Path,
) -> dict[str, Any]:
    """截断过长的输出，将完整内容写入文件。

    Args:
        stdout: 标准输出内容。
        stderr: 标准错误内容。
        max_chars: 截断阈值。
        workspace: 工作目录，用于存放溢出文件。

    Returns:
        包含 truncated 标记和可能的文件路径的字典。
    """
    result: dict[str, Any] = {}

    if len(stdout) > max_chars:
        output_dir = workspace / ".researchagent" / "bash-outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        path = output_dir / f"stdout-{stamp}.txt"
        path.write_text(stdout, encoding="utf-8")
        result["stdout"] = stdout[:max_chars]
        result["stdout_truncated"] = True
        result["stdout_path"] = str(path.relative_to(workspace))
    else:
        result["stdout"] = stdout

    if len(stderr) > max_chars:
        output_dir = workspace / ".researchagent" / "bash-outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        path = output_dir / f"stderr-{stamp}.txt"
        path.write_text(stderr, encoding="utf-8")
        result["stderr"] = stderr[:max_chars]
        result["stderr_truncated"] = True
        result["stderr_path"] = str(path.relative_to(workspace))
    else:
        result["stderr"] = stderr

    return result


def _coerce_timeout(
    timeout_seconds: int | str | float | None,
    default: int,
    max_val: int,
) -> int:
    """将 timeout 参数转换为有效的整数秒数。"""
    if timeout_seconds is None:
        return default
    try:
        value = int(timeout_seconds)
    except (ValueError, TypeError):
        return default
    if value <= 0:
        return default
    return min(value, max_val)


# ---- 核心执行函数 ----

def run_bash(
    state: RuntimeState,
    command: str,
    timeout_seconds: int | str | float | None = None,
) -> dict[str, Any]:
    """在 workspace 内安全执行 Shell 命令。

    执行流程:
        1. 验证命令非空
        2. 危险模式检测 → 拒绝
        3. 平台命令适配
        4. subprocess.run() 执行
        5. 输出解码 + 截断

    Args:
        state: 运行时状态 (提供 workspace 等配置)。
        command: 要执行的 Shell 命令。
        timeout_seconds: 超时秒数，None 时使用 state 的默认值。

    Returns:
        执行结果字典，格式:
        {
            "ok": bool,           # 是否成功 (exit_code == 0)
            "command": str,       # 规范化后的命令
            "exit_code": int,     # 进程退出码
            "stdout": str,        # 标准输出 (可能被截断)
            "stderr": str,        # 标准错误 (可能被截断)
            "timed_out": bool,    # 是否超时
            "duration_ms": int,   # 执行耗时
            "stdout_truncated": bool,   # (可选)
            "stderr_truncated": bool,   # (可选)
            "stdout_path": str,         # (可选) 溢出输出文件路径
            "stderr_path": str,         # (可选) 溢出错误文件路径
        }
    """
    # 1. 验证
    if not command or not command.strip():
        return {"ok": False, "error": "命令不能为空。"}

    command = command.strip()

    # 2. 危险模式检测
    danger = _looks_dangerous(command)
    if danger is not None:
        return {
            "ok": False,
            "command": command,
            "error": f"危险命令已拒绝: {danger}",
        }

    # 3. 平台适配
    normalized = _normalize_command(command)

    # 4. 超时处理
    timeout = _coerce_timeout(
        timeout_seconds,
        state.bash_default_timeout_seconds,
        state.bash_max_timeout_seconds,
    )

    # 5. 执行
    start = time.monotonic()
    try:
        proc = subprocess.run(
            normalized,
            cwd=str(state.workspace),
            shell=True,
            capture_output=True,
            timeout=timeout,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        stdout_str = _decode_output(proc.stdout)
        stderr_str = _decode_output(proc.stderr)

        # 截断处理
        output = _truncate_output(
            stdout_str,
            stderr_str,
            state.bash_max_output_chars,
            state.workspace,
        )

        return {
            "ok": proc.returncode == 0,
            "command": normalized,
            "exit_code": proc.returncode,
            "timed_out": False,
            "duration_ms": duration_ms,
            **output,
        }

    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "ok": False,
            "command": normalized,
            "exit_code": None,
            "stdout": "",
            "stderr": f"命令执行超时 (>{timeout}s)。",
            "timed_out": True,
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "ok": False,
            "command": normalized,
            "error": f"执行失败: {exc}",
            "duration_ms": duration_ms,
        }
