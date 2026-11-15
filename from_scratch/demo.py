from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.cases import CASES, Case, scripted_model
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


TRACE_PATH = Path(__file__).resolve().parent.parent / "traces" / "example_run.json"


def _short(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def print_history(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                arguments = json.dumps(call.get("arguments"), ensure_ascii=False)
                print(f"  assistant → 调用 {call.get('name')}({arguments})  [id={call['id']}]")
        elif role == "tool":
            print(f"  tool      ← {message['name']}: {_short(message['content'])}")
        else:
            print(f"  {role:<9} : {message['content']}")


def tool_outcomes(messages: list[dict[str, Any]]) -> list[str]:
    outcomes = []
    for message in messages:
        if message["role"] == "tool":
            output = json.loads(message["content"])
            outcomes.append(output["error"] if isinstance(output, dict) and "error" in output else "ok")
    return outcomes


def run_case(case: Case) -> bool:
    messages = [{"role": "user", "content": case.user}]
    answer, trace = run_loop(messages, scripted_model(case))
    outcomes = tool_outcomes(messages)
    passed = answer == case.final_answer and outcomes == case.outcomes

    print(f"=== {case.name}（{len(trace)} 轮）===")
    print_history(messages)
    print(f"  工具结果: {outcomes or '无工具调用'}，预期: {case.outcomes or '无工具调用'} → {'符合' if passed else '不符合'}")
    print()
    return passed


def run_planner_demo() -> None:
    messages = [{"role": "user", "content": "找一门人工智能课程并给出详情"}]
    answer, trace = run_loop(messages, planner)
    print("最终答案:", answer)
    print("调用轨迹:")
    for item in trace:
        print(json.dumps(item, ensure_ascii=False, indent=2))

    TRACE_PATH.parent.mkdir(exist_ok=True)
    TRACE_PATH.write_text(
        json.dumps(
            {"implementation": "from_scratch", "final_answer": answer, "messages": messages, "trace": trace},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"轨迹已保存到 {TRACE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part A 从零实现的 Tool-use Agent 演示")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", choices=[case.name for case in CASES], help="用 scripted_model 运行一个共享用例")
    group.add_argument("--all", action="store_true", help="依次运行全部 10 个共享用例")
    group.add_argument("--list", action="store_true", help="列出所有用例")
    args = parser.parse_args()

    if args.list:
        for case in CASES:
            print(f"{case.name:<20} {case.user}")
    elif args.case:
        run_case(next(case for case in CASES if case.name == args.case))
    elif args.all:
        results = [run_case(case) for case in CASES]
        print(f"共 {len(results)} 个用例，符合预期 {sum(results)} 个")
    else:
        run_planner_demo()
