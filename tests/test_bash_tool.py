"""Bash Tool 单元测试。"""

from __future__ import annotations

import platform
from pathlib import Path

from researchagent.core.state import RuntimeState
from researchagent.tools.bash_tool import (
    _looks_dangerous,
    _normalize_command,
    bash_tool_description,
    run_bash,
)


def _make_state() -> RuntimeState:
    """创建测试用的 RuntimeState。"""
    return RuntimeState(workspace=Path.cwd())


def test_bash_tool_description_contains_platform() -> None:
    """验证描述包含平台信息。"""
    desc = bash_tool_description()
    system = platform.system().lower()
    if system == "windows":
        assert "Windows" in desc
    else:
        assert "POSIX" in desc or "Linux" in desc or platform.system() in desc


def test_looks_dangerous_rm_rf() -> None:
    """验证危险命令 rm -rf 被检测。"""
    assert _looks_dangerous("rm -rf /") is not None
    assert _looks_dangerous("sudo rm -rf /var") is not None


def test_looks_dangerous_shutdown() -> None:
    """验证危险命令 shutdown 被检测。"""
    assert _looks_dangerous("shutdown /s") is not None


def test_looks_dangerous_format() -> None:
    """验证危险命令 format 被检测。"""
    assert _looks_dangerous("format C:") is not None


def test_looks_dangerous_safe_commands() -> None:
    """验证正常命令不被拦截。"""
    assert _looks_dangerous("dir") is None
    assert _looks_dangerous("echo hello") is None
    assert _looks_dangerous("python --version") is None
    assert _looks_dangerous("pip list") is None


def test_normalize_command_on_windows() -> None:
    """验证 Windows 下命令适配。"""
    if platform.system().lower() == "windows":
        result = _normalize_command("python3 script.py")
        assert "python " in result
        assert "python3" not in result


def test_run_bash_empty_command() -> None:
    """验证空命令返回错误。"""
    state = _make_state()
    result = run_bash(state, "")
    assert result["ok"] is False
    assert "error" in result


def test_run_bash_echo() -> None:
    """验证基本命令执行。"""
    state = _make_state()
    result = run_bash(state, "echo hello_test_42")
    assert result["ok"] is True
    assert "hello_test_42" in result["stdout"]
    assert result["exit_code"] == 0


def test_run_bash_returns_command() -> None:
    """验证返回结果包含命令。"""
    state = _make_state()
    result = run_bash(state, "echo test")
    assert "command" in result
    assert result["duration_ms"] >= 0


def test_run_bash_dangerous_command_blocked() -> None:
    """验证危险命令被拒绝。"""
    state = _make_state()
    result = run_bash(state, "rm -rf /")
    assert result["ok"] is False
    assert "error" in result
    assert "危险" in result["error"] or "dangerous" in result["error"].lower()


def test_run_bash_invalid_command() -> None:
    """验证无效命令仍然返回结果 (不抛异常)。"""
    state = _make_state()
    result = run_bash(state, "nonexistent_command_xyz_123")
    # 命令不存在不应导致异常，应正常返回
    assert "command" in result
