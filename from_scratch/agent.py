from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from common.tools import TOOL_REGISTRY


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Validate the name and execute a registered local tool."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"unknown tool: {name}")
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be an object")
    return TOOL_REGISTRY[name](**arguments)


def run_loop(
    messages: list[dict[str, Any]],
    request_model: Callable[[list[dict[str, Any]]], dict[str, Any]],
    *,
    max_iterations: int = 8,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the complete model → tool → model loop.

    The model adapter returns either {"content": str, "tool_calls": []} or
    a response with tool_calls containing id/name/arguments. This small adapter
    keeps the orchestration independent of a particular vendor SDK.
    """
    trace: list[dict[str, Any]] = []
    for iteration in range(max_iterations):
        response = request_model(messages)
        trace.append({"iteration": iteration, "response": response})
        tool_calls = response.get("tool_calls") or []
        if not tool_calls:
            return str(response.get("content", "")), trace

        messages.append({
            "role": "assistant",
            "content": response.get("content"),
            "tool_calls": tool_calls,
        })
        for raw_call in tool_calls:
            call = ToolCall(
                id=str(raw_call.get("id", f"call-{iteration}")),
                name=str(raw_call["name"]),
                arguments=raw_call.get("arguments", {}),
            )
            try:
                result = execute_tool(call.name, call.arguments)
            except Exception as exc:  # return errors to the model for self-correction
                result = {"error": type(exc).__name__, "message": str(exc)}
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })
    raise RuntimeError(f"tool loop exceeded max_iterations={max_iterations}")
