"""安全数学表达式求值工具 (Calculator Tool)。

使用 `simpleeval` 库基于 AST 解析进行安全的数学表达式求值，
不使用 Python 的 `eval()`，防止任意代码执行。

支持:
    - 基本运算: +, -, *, /, **, //, %
    - 数学函数: sqrt, sin, cos, tan, asin, acos, atan
    - 对数函数: log, log10, log2, exp
    - 取整函数: ceil, floor, trunc, abs, round
    - 常量: pi, e, inf, nan
"""

from __future__ import annotations

import math
from typing import Any


def evaluate(expression: str) -> dict[str, Any]:
    """安全求值数学表达式。

    使用 AST 解析 + 白名单函数的方式安全执行数学计算。
    不支持变量赋值、属性访问、导入语句等危险操作。

    Args:
        expression: 单行数学表达式。示例:
            "2 + 3 * 4"
            "sqrt(144)"
            "sin(pi / 2)"
            "log(100, 10)"

    Returns:
        {"ok": True, "expression": str, "result": number}
        或 {"ok": False, "expression": str, "error": str}
    """
    from simpleeval import NameNotDefined, SimpleEval

    evaluator = SimpleEval()

    # 数学函数 → evaluator.functions
    evaluator.functions.update({
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "degrees": math.degrees,
        "radians": math.radians,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "ceil": math.ceil,
        "floor": math.floor,
        "trunc": math.trunc,
        "round": round,
        "abs": abs,
    })

    # 常量和内置值 → evaluator.names
    evaluator.names.update({
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
        "nan": math.nan,
    })

    try:
        result = evaluator.eval(expression.strip())
        return {"ok": True, "expression": expression.strip(), "result": result}
    except NameNotDefined as exc:
        return {
            "ok": False,
            "expression": expression.strip(),
            "error": f"不支持的函数或变量: {exc}。请使用支持的运算符和数学函数。",
        }
    except SyntaxError as exc:
        return {
            "ok": False,
            "expression": expression.strip(),
            "error": f"表达式语法错误: {exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "expression": expression.strip(),
            "error": f"计算失败: {exc}",
        }
