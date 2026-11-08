from __future__ import annotations

import ast
import operator
from typing import Any

from .course_data import COURSES


def search_courses(keyword: str) -> list[dict[str, Any]]:
    """根据课程编号、名称、院系或描述中的关键词搜索课程。"""
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError("keyword 必须是非空字符串")
    key = keyword.strip().lower()
    return [c for c in COURSES if key in " ".join(str(v).lower() for v in c.values())]


def get_course_detail(course_id: str) -> dict[str, Any]:
    """根据课程编号取得一门课程的完整信息。"""
    for course in COURSES:
        if course["id"].lower() == course_id.strip().lower():
            return course
    raise LookupError(f"找不到课程: {course_id}")


def filter_courses(
    department: str | None = None,
    min_credits: int | None = None,
) -> list[dict[str, Any]]:
    """按院系和最低学分筛选课程。"""
    if min_credits is not None and (not isinstance(min_credits, int) or min_credits < 0):
        raise ValueError("min_credits 必须是非负整数")
    return [
        c for c in COURSES
        if (department is None or c["department"] == department)
        and (min_credits is None or c["credits"] >= min_credits)
    ]


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def calculate(expression: str) -> float:
    """计算只包含数字和 + - * / 括号的算式；禁止执行任意代码。"""
    if not isinstance(expression, str) or len(expression) > 100:
        raise ValueError("expression 必须是长度不超过 100 的字符串")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("只允许数字、括号和 + - * /")

    return visit(ast.parse(expression, mode="eval"))


TOOL_REGISTRY = {
    "search_courses": search_courses,
    "get_course_detail": get_course_detail,
    "filter_courses": filter_courses,
    "calculate": calculate,
}
