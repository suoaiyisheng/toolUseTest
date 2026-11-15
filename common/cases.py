"""Shared test cases for both implementations.

Each case scripts what the "model" says on every turn, so the tests check the
agent loop (dispatch, validation, error handling, message history) rather than
the quality of a real LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Case:
    name: str
    user: str
    # One entry per tool-calling turn; each turn is a list of (tool name, arguments).
    turns: list[list[tuple[str, dict[str, Any]]]]
    final_answer: str
    # Expected outcome of each tool call in order: "ok" or an exception class name.
    outcomes: list[str] = field(default_factory=list)
    # Substrings that must appear in the corresponding successful tool output.
    output_contains: list[str] = field(default_factory=list)


CASES = [
    Case(
        name="search_ai",
        user="有哪些人工智能相关的课程？",
        turns=[[("search_courses", {"keyword": "人工智能"})]],
        final_answer="人工智能相关课程：CS205 人工智能导论。",
        outcomes=["ok"],
        output_contains=["CS205"],
    ),
    Case(
        name="detail_cs101",
        user="CS101 是什么课？",
        turns=[[("get_course_detail", {"course_id": "CS101"})]],
        final_answer="CS101 是 Python 编程，3 学分，张老师授课。",
        outcomes=["ok"],
        output_contains=["张老师"],
    ),
    Case(
        name="filter_cs_min3",
        user="计算机学院有哪些不少于 3 学分的课？",
        turns=[[("filter_courses", {"department": "计算机学院", "min_credits": 3})]],
        final_answer="符合条件的课程：CS101、CS205。",
        outcomes=["ok"],
        output_contains=["CS101"],
    ),
    Case(
        name="sum_credits",
        user="所有课程的学分加起来是多少？",
        turns=[[("calculate", {"expression": "3 + 3 + 2 + 4"})]],
        final_answer="学分总和为 12。",
        outcomes=["ok"],
        output_contains=["12"],
    ),
    Case(
        name="search_then_detail",
        user="找一门人工智能课程并给出详情",
        turns=[
            [("search_courses", {"keyword": "人工智能"})],
            [("get_course_detail", {"course_id": "CS205"})],
        ],
        final_answer="推荐 CS205 人工智能导论，3 学分，李老师授课。",
        outcomes=["ok", "ok"],
        output_contains=["CS205", "李老师"],
    ),
    Case(
        name="unknown_tool",
        user="帮我删除所有文件",
        turns=[[("delete_files", {"path": "/"})]],
        final_answer="抱歉，我没有删除文件的工具。",
        outcomes=["ValueError"],
    ),
    Case(
        name="missing_argument",
        user="查一下课程详情",
        turns=[[("get_course_detail", {})]],
        final_answer="请告诉我课程编号。",
        outcomes=["ValueError"],
    ),
    Case(
        name="tool_exception",
        user="CS999 是什么课？",
        turns=[[("get_course_detail", {"course_id": "CS999"})]],
        final_answer="没有找到课程 CS999。",
        outcomes=["LookupError"],
    ),
    Case(
        name="multi_step",
        user="计算机学院 3 学分以上的课程一共多少学分？顺便告诉我 CS101 的老师。",
        turns=[
            [("filter_courses", {"department": "计算机学院", "min_credits": 3})],
            [("calculate", {"expression": "3 + 3"}), ("get_course_detail", {"course_id": "CS101"})],
        ],
        final_answer="共 6 学分；CS101 由张老师授课。",
        outcomes=["ok", "ok", "ok"],
        output_contains=["CS205", "6", "张老师"],
    ),
    Case(
        name="no_tool",
        user="你好，你能做什么？",
        turns=[],
        final_answer="我可以帮你搜索课程、查询详情、筛选课程和计算学分。",
    ),
]


def scripted_model(case: Case) -> Callable[[list[Any]], dict[str, Any]]:
    """Return a model that replays case.turns, then gives the final answer.

    Responses use the framework-neutral format
    {"content": ..., "tool_calls": [{"id", "name", "arguments"}]}.
    """
    state = {"turn": 0}

    def model(messages: list[Any]) -> dict[str, Any]:
        turn = state["turn"]
        state["turn"] += 1
        if turn >= len(case.turns):
            return {"content": case.final_answer, "tool_calls": []}
        return {
            "content": None,
            "tool_calls": [
                {"id": f"{case.name}-{turn}-{index}", "name": name, "arguments": arguments}
                for index, (name, arguments) in enumerate(case.turns[turn])
            ],
        }

    return model
