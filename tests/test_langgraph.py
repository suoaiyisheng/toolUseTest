import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from common.cases import CASES, scripted_model
from langgraph_version.agent import NeutralModel, offline_graph, run_agent


def test_langgraph_offline_smoke():
    result = offline_graph().invoke({"messages": [HumanMessage(content="搜索人工智能课程")]}, {"recursion_limit": 8})
    assert result["messages"][-1].content == "已完成工具查询。"


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_shared_cases(case):
    answer, messages = run_agent(case.user, NeutralModel(scripted_model(case)))

    assert answer == case.final_answer
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert ["ok" if m.status == "success" else "error" for m in tool_messages] == [
        "ok" if outcome == "ok" else "error" for outcome in case.outcomes
    ]
    successes = [m for m in tool_messages if m.status == "success"]
    for message, expected in zip(successes, case.output_contains):
        assert expected in message.content


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_every_tool_call_has_a_matching_tool_message(case):
    _, messages = run_agent(case.user, NeutralModel(scripted_model(case)))

    call_ids = [c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    result_ids = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    assert call_ids == result_ids
    assert isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls


def run_single(name, arguments):
    def model(messages):
        if len(messages) == 1:
            return {"content": None, "tool_calls": [{"id": "a", "name": name, "arguments": arguments}]}
        return {"content": "done", "tool_calls": []}

    _, messages = run_agent("x", NeutralModel(model))
    return next(m for m in messages if isinstance(m, ToolMessage))


def test_tool_exception_does_not_crash_graph():
    message = run_single("get_course_detail", {"course_id": "CS999"})
    assert message.status == "error"
    assert json.loads(message.content)["error"] == "LookupError"


def test_validation_error_is_returned_to_model():
    message = run_single("get_course_detail", {})
    assert message.status == "error"
    assert "course_id" in message.content


def test_unknown_tool_is_returned_to_model():
    message = run_single("delete_files", {})
    assert message.status == "error"
    assert "delete_files" in message.content


def test_recursion_limit_stops_infinite_loop():
    def looping_model(messages):
        return {"content": None, "tool_calls": [{"id": str(len(messages)), "name": "calculate", "arguments": {"expression": "1+1"}}]}

    with pytest.raises(GraphRecursionError):
        run_agent("x", NeutralModel(looping_model), max_iterations=3)
