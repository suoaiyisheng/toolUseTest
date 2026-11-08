from __future__ import annotations

import json

from common.tools import get_course_detail, search_courses
from .agent import run_loop


def planner(messages):
    """Deterministic offline planner used in class before adding a real LLM."""
    if len(messages) == 1:
        return {"content": None, "tool_calls": [{"id": "call-1", "name": "search_courses", "arguments": {"keyword": "人工智能"}}]}
    result = json.loads(messages[-1]["content"])
    if isinstance(result, list) and result:
        first = result[0]
        return {"content": None, "tool_calls": [{"id": "call-2", "name": "get_course_detail", "arguments": {"course_id": first["id"]}}]}
    if isinstance(result, dict) and result.get("id"):
        return {
            "content": (
                f"推荐课程：{result['id']} {result['name']}，{result['credits']} 学分，"
                f"授课教师：{result['teacher']}。"
            ),
            "tool_calls": [],
        }
    return {"content": "没有找到课程。", "tool_calls": []}


if __name__ == "__main__":
    answer, trace = run_loop([{"role": "user", "content": "找一门人工智能课程并给出详情"}], planner)
    print("最终答案:", answer)
    print("调用轨迹:")
    for item in trace:
        print(json.dumps(item, ensure_ascii=False, indent=2))
