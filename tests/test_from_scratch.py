import json

import pytest

from common.cases import CASES, scripted_model
from from_scratch.agent import run_loop


def test_loop_returns_final_answer_and_records_tool_messages():
    calls = {"n": 0}

    def model(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"content": None, "tool_calls": [{"id": "1", "name": "calculate", "arguments": {"expression": "2+3"}}]}
        assert messages[-1]["role"] == "tool"
        return {"content": "答案是 5。", "tool_calls": []}

    answer, trace = run_loop([{"role": "user", "content": "2+3"}], model)
    assert answer == "答案是 5。"
    assert len(trace) == 2


def tool_outputs(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def outcome_of(output):
    return output["error"] if isinstance(output, dict) and "error" in output else "ok"


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_shared_cases(case):
    messages = [{"role": "user", "content": case.user}]
    answer, trace = run_loop(messages, scripted_model(case))

    assert answer == case.final_answer
    assert len(trace) == len(case.turns) + 1

    outputs = tool_outputs(messages)
    assert [outcome_of(o) for o in outputs] == case.outcomes
    for output, expected in zip(outputs, case.output_contains):
        assert expected in json.dumps(output, ensure_ascii=False)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_every_tool_call_has_a_matching_tool_message(case):
    messages = [{"role": "user", "content": case.user}]
    run_loop(messages, scripted_model(case))

    call_ids = [c["id"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    result_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert call_ids == result_ids
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            assert messages[index - 1]["role"] in ("assistant", "tool")
    assert messages[-1] == {"role": "assistant", "content": case.final_answer}


def one_call_then_answer(raw_call):
    def model(messages):
        if len(messages) == 1:
            return {"content": None, "tool_calls": [raw_call]}
        return {"content": "done", "tool_calls": []}
    return model


def run_single(raw_call):
    messages = [{"role": "user", "content": "x"}]
    run_loop(messages, one_call_then_answer(raw_call))
    return tool_outputs(messages)[0]


def test_openai_format_with_json_string_arguments():
    output = run_single({"id": "a", "type": "function", "function": {"name": "search_courses", "arguments": '{"keyword": "Python"}'}})
    assert {c["id"] for c in output} == {"CS101", "DS210"}


def test_invalid_json_arguments():
    output = run_single({"id": "a", "name": "search_courses", "arguments": '{"keyword": "AI"'})
    assert output["error"] == "ToolCallParseError"


def test_wrong_argument_type():
    output = run_single({"id": "a", "name": "get_course_detail", "arguments": {"course_id": 101}})
    assert output["error"] == "TypeError"
    assert "course_id" in output["message"]


def test_extra_argument_rejected():
    output = run_single({"id": "a", "name": "search_courses", "arguments": {"keyword": "AI", "limit": 3}})
    assert output["error"] == "ValueError"
    assert "limit" in output["message"]


def test_missing_tool_name():
    output = run_single({"id": "a", "arguments": {}})
    assert output["error"] == "ToolCallParseError"


def test_model_can_self_correct_after_error():
    def model(messages):
        last = messages[-1]
        if last["role"] == "user":
            return {"content": None, "tool_calls": [{"id": "1", "name": "get_course_detail", "arguments": {}}]}
        if "error" in json.loads(last["content"]):
            return {"content": None, "tool_calls": [{"id": "2", "name": "get_course_detail", "arguments": {"course_id": "CS101"}}]}
        return {"content": json.loads(last["content"])["name"], "tool_calls": []}

    answer, trace = run_loop([{"role": "user", "content": "CS101"}], model)
    assert answer == "Python 编程"
    assert len(trace) == 3


def test_max_iterations_stops_infinite_loop():
    def looping_model(messages):
        return {"content": None, "tool_calls": [{"name": "calculate", "arguments": {"expression": "1+1"}}]}

    with pytest.raises(RuntimeError, match="max_iterations=3"):
        run_loop([{"role": "user", "content": "x"}], looping_model, max_iterations=3)
