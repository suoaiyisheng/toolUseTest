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

