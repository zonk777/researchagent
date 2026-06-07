"""Calculator Tool 单元测试。"""

from __future__ import annotations

from researchagent.tools.calculator_tool import evaluate


def test_basic_arithmetic() -> None:
    """验证基本四则运算。"""
    result = evaluate("2 + 3 * 4")
    assert result["ok"] is True
    assert result["result"] == 14  # 3*4 + 2


def test_parentheses() -> None:
    """验证括号优先级。"""
    result = evaluate("(2 + 3) * 4")
    assert result["ok"] is True
    assert result["result"] == 20


def test_power() -> None:
    """验证幂运算。"""
    result = evaluate("2 ** 10")
    assert result["ok"] is True
    assert result["result"] == 1024


def test_float_division() -> None:
    """验证浮点除法。"""
    result = evaluate("10 / 3")
    assert result["ok"] is True
    assert abs(result["result"] - 3.333333) < 0.001


def test_sqrt() -> None:
    """验证 sqrt 函数。"""
    result = evaluate("sqrt(144)")
    assert result["ok"] is True
    assert result["result"] == 12.0


def test_sin_cos() -> None:
    """验证三角函数。"""
    result = evaluate("sin(pi / 2)")
    assert result["ok"] is True
    assert abs(result["result"] - 1.0) < 0.001

    result = evaluate("cos(0)")
    assert result["ok"] is True
    assert abs(result["result"] - 1.0) < 0.001


def test_log() -> None:
    """验证对数函数。"""
    result = evaluate("log(100, 10)")
    assert result["ok"] is True
    assert abs(result["result"] - 2.0) < 0.001


def test_constants() -> None:
    """验证数学常量。"""
    result = evaluate("pi")
    assert result["ok"] is True
    assert abs(result["result"] - 3.14159) < 0.001

    result = evaluate("e")
    assert result["ok"] is True
    assert abs(result["result"] - 2.71828) < 0.001


def test_abs_ceil_floor() -> None:
    """验证取整函数。"""
    result = evaluate("abs(-5.7)")
    assert result["ok"] is True
    assert result["result"] == 5.7

    result = evaluate("ceil(3.2)")
    assert result["ok"] is True
    assert result["result"] == 4

    result = evaluate("floor(3.8)")
    assert result["ok"] is True
    assert result["result"] == 3


def test_expression_in_result() -> None:
    """验证返回结果包含原始表达式。"""
    result = evaluate("42")
    assert result["expression"] == "42"


def test_syntax_error() -> None:
    """验证语法错误返回 ok=False。"""
    result = evaluate("(2 + 3")  # 括号不匹配
    assert result["ok"] is False
    assert "error" in result


def test_undefined_function() -> None:
    """验证不支持的操作返回错误。"""
    result = evaluate("__import__('os').system('dir')")
    assert result["ok"] is False, "dangerous imports should be blocked"
    assert "error" in result


def test_undefined_variable() -> None:
    """验证未定义变量返回错误。"""
    result = evaluate("x + 1")
    assert result["ok"] is False
    assert "error" in result
