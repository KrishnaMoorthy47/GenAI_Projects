from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import tool

# Safe operations allowed in calculator
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return _SAFE_NAMES[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        operand = _safe_eval(node.operand)
        return _SAFE_OPS[type(node.op)](operand)
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name in _SAFE_NAMES:
            args = [_safe_eval(a) for a in node.args]
            return _SAFE_NAMES[func_name](*args)
    raise ValueError(f"Unsafe expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely. Useful for financial calculations.

    Supports: +, -, *, /, **, sqrt(), log(), abs(), round(), pi, e.

    Args:
        expression: Mathematical expression string (e.g. '(150 - 120) / 120 * 100')
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as exc:
        return f"Error evaluating expression: {exc}"
