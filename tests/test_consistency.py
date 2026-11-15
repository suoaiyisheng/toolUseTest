"""Both implementations must behave the same on every shared case."""
import json

import pytest
from langchain_core.messages import ToolMessage

from common.cases import CASES, scripted_model
from from_scratch.agent import run_loop
from langgraph_version.agent import NeutralModel, run_agent


def from_scratch_summary(case):
    messages = [{"role": "user", "content": case.user}]
    answer, _ = run_loop(messages, scripted_model(case))
    tools = []
    for m in messages:
        if m["role"] == "tool":
            output = json.loads(m["content"])
            failed = isinstance(output, dict) and "error" in output
            tools.append((m["name"], "error" if failed else "ok", None if failed else output))
    return answer, tools


def langgraph_summary(case):
    answer, messages = run_agent(case.user, NeutralModel(scripted_model(case)))
    tools = []
    for m in messages:
        if isinstance(m, ToolMessage):
            failed = m.status == "error"
            tools.append((m.name, "error" if failed else "ok", None if failed else json.loads(m.content)))
    return answer, tools


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_both_versions_agree(case):
    assert from_scratch_summary(case) == langgraph_summary(case)
