"""CLI 冒烟测试 + LLM Provider 单元测试。

测试覆盖:
    - CLI: 无参数显示帮助、test --help 正常
    - Provider: 缺少环境变量抛异常、正确读取配置
"""

from __future__ import annotations

from typer.testing import CliRunner

from researchagent.cli.app import app
from researchagent.providers.openai_provider import create_model


# ═══════════════════════════════════════════════════════════════
# CLI 冒烟测试
# ═══════════════════════════════════════════════════════════════


def test_cli_no_args_shows_help() -> None:
    """验证无子命令时 CLI 显示帮助信息。"""
    runner = CliRunner()
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "researchagent" in result.output.lower()


def test_cli_test_command_help() -> None:
    """验证 test 子命令的 --help 正常工作。"""
    runner = CliRunner()
    result = runner.invoke(app, ["test", "--help"])
    assert result.exit_code == 0
    assert "test" in result.output.lower()
    assert "llm" in result.output.lower() or "prompt" in result.output.lower()


# ═══════════════════════════════════════════════════════════════
# Provider 单元测试
# ═══════════════════════════════════════════════════════════════


def test_create_model_missing_env(monkeypatch) -> None:
    """验证缺少环境变量时 create_model 抛出 RuntimeError。

    清除所有三个必需的环境变量 (API_KEY, MODEL, BASE_URL)，
    确认 create_model() 抛出包含缺失变量名的 RuntimeError。
    """
    # 清除环境变量
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)

    # 确保 load_dotenv 不会从 .env 文件补充值
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("MODEL", "")
    monkeypatch.setenv("BASE_URL", "")

    try:
        create_model()
    except RuntimeError as e:
        msg = str(e)
        # 错误消息应包含至少一个缺失的变量名
        assert any(
            name in msg for name in ["API_KEY", "MODEL", "BASE_URL"]
        ), f"错误消息应包含缺失变量名，实际: {msg}"
    else:
        raise AssertionError("缺少环境变量时应抛出 RuntimeError")


def test_create_model_missing_env_message_format(monkeypatch) -> None:
    """验证错误消息包含中文提示和 .env.example 指引。"""
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("MODEL", "")
    monkeypatch.setenv("BASE_URL", "")

    try:
        create_model()
    except RuntimeError as e:
        msg = str(e)
        assert ".env.example" in msg
        assert ".env" in msg or "配置" in msg


def test_create_model_with_env(monkeypatch) -> None:
    """验证设置环境变量后 create_model 能成功创建 ChatOpenAI 实例。

    此测试仅验证配置解析，不发起真实的 API 请求。
    """
    monkeypatch.setenv("API_KEY", "test-key-12345")
    monkeypatch.setenv("MODEL", "gpt-4o-mini")
    monkeypatch.setenv("BASE_URL", "https://api.openai.com/v1")

    model = create_model()
    assert model is not None, "create_model() 不应返回 None"
    assert model.model_name == "gpt-4o-mini"


def test_create_model_respects_temperature(monkeypatch) -> None:
    """验证 create_model 的 temperature 参数生效。"""
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.setenv("BASE_URL", "https://test.example.com/v1")

    # 非默认 temperature
    model = create_model(temperature=0.7)
    assert model is not None
    # ChatOpenAI 将 temperature 存储在内部字段
    assert model.temperature == 0.7

    # 默认 temperature 为 0.0
    model_default = create_model()
    assert model_default.temperature == 0.0
