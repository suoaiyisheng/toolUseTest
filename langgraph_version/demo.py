from __future__ import annotations

import argparse
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from common.cases import CASES, Case, scripted_model
from .agent import NeutralModel, offline_graph, run_agent


def _short(text: str, limit: int = 100) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def print_history(messages: list[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                arguments = json.dumps(call["args"], ensure_ascii=False)
                print(f"  AIMessage   → 调用 {call['name']}({arguments})  [id={call['id']}]")
        elif isinstance(message, ToolMessage):
            print(f"  ToolMessage ← {message.name} [{message.status}]: {_short(str(message.content))}")
        else:
            print(f"  {type(message).__name__:<11} : {message.content}")


def run_case(case: Case) -> bool:
    answer, messages = run_agent(case.user, NeutralModel(scripted_model(case)))
    outcomes = ["ok" if m.status == "success" else "error" for m in messages if isinstance(m, ToolMessage)]
    expected = ["ok" if outcome == "ok" else "error" for outcome in case.outcomes]
    passed = answer == case.final_answer and outcomes == expected

    print(f"=== {case.name} ===")
    print_history(messages)
    print(f"  工具结果: {outcomes or '无工具调用'}，预期: {expected or '无工具调用'} → {'符合' if passed else '不符合'}")
    print()
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part B LangGraph 版 Tool-use Agent 演示")
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
        result = offline_graph().invoke({"messages": [HumanMessage(content="找人工智能课程")]}, {"recursion_limit": 8})
        print_history(result["messages"])
