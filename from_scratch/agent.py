from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from common.tool_specs import TOOL_SPECS
from common.tools import TOOL_REGISTRY

SCHEMAS: dict[str, dict[str, Any]] = {
    spec["function"]["name"]: spec["function"]["parameters"] for spec in TOOL_SPECS
}

JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "null": type(None),
    "object": dict,
    "array": list,
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


def _json_type_name(value: Any) -> str:
    for json_name, py_type in JSON_TYPES.items():
        if json_name in ("integer", "number") and isinstance(value, bool):
            continue
        if isinstance(value, py_type):
            return json_name
    return type(value).__name__


def validate_arguments(name: str, arguments: dict[str, Any]) -> None:
    """Check arguments against the tool's JSON Schema before execution."""
    schema = SCHEMAS[name]
    properties = schema.get("properties", {})

    missing = [key for key in schema.get("required", []) if key not in arguments]
    if missing:
        raise ValueError(f"{name} 缺少必需参数: {', '.join(missing)}")

    if schema.get("additionalProperties") is False:
        extra = [key for key in arguments if key not in properties]
        if extra:
            allowed = ", ".join(properties) or "无"
            raise ValueError(f"{name} 不接受参数: {', '.join(extra)}（允许的参数: {allowed}）")

    for key, value in arguments.items():
        expected = properties.get(key, {}).get("type")
        if expected is None:
            continue
        expected_types = expected if isinstance(expected, list) else [expected]
        if _json_type_name(value) not in expected_types:
            raise TypeError(
                f"{name} 的参数 {key} 应为 {' 或 '.join(expected_types)}，"
                f"实际为 {_json_type_name(value)}（值: {value!r}）"
            )


class ToolCallParseError(ValueError):
    """The model produced a tool call that cannot be parsed."""


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Accept a dict or a JSON string (OpenAI format) and return a dict."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolCallParseError(f"arguments 不是合法 JSON: {exc.msg}（位置 {exc.pos}）: {raw!r}") from exc
    if not isinstance(raw, dict):
        raise ToolCallParseError(f"arguments 必须是 JSON 对象，实际为 {_json_type_name(raw)}")
    return raw


def _function_part(raw_call: dict[str, Any]) -> dict[str, Any]:
    function = raw_call.get("function")
    return function if isinstance(function, dict) else raw_call


def _tool_name(raw_call: dict[str, Any]) -> str | None:
    name = _function_part(raw_call).get("name")
    return name if isinstance(name, str) and name else None


def _ensure_id(raw_call: Any, fallback_id: str) -> dict[str, Any]:
    if not isinstance(raw_call, dict):
        raw_call = {"raw": raw_call}
    if not raw_call.get("id"):
        raw_call = {**raw_call, "id": fallback_id}
    return raw_call


def parse_tool_call(raw_call: dict[str, Any]) -> ToolCall:
    """Normalize flat {id, name, arguments} or OpenAI {id, function: {...}} calls."""
    name = _tool_name(raw_call)
    if name is None:
        raise ToolCallParseError(f"tool call 缺少工具名 name: {raw_call!r}")
    arguments = parse_arguments(_function_part(raw_call).get("arguments"))
    return ToolCall(id=str(raw_call["id"]), name=name, arguments=arguments)


def execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Validate the name and arguments, then execute a registered local tool."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"unknown tool: {name}（可用工具: {', '.join(TOOL_REGISTRY)}）")
    if not isinstance(arguments, dict):
        raise TypeError("tool arguments must be an object")
    validate_arguments(name, arguments)
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
        step: dict[str, Any] = {"iteration": iteration, "response": response, "tool_results": []}
        trace.append(step)
        if not isinstance(response, dict):
            raise TypeError(f"model adapter must return a dict, got {type(response).__name__}")
        tool_calls = response.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not tool_calls:
            answer = str(response.get("content") or "")
            messages.append({"role": "assistant", "content": answer})
            return answer, trace

        tool_calls = [_ensure_id(raw, f"call-{iteration}-{index}") for index, raw in enumerate(tool_calls)]
        messages.append({
            "role": "assistant",
            "content": response.get("content"),
            "tool_calls": tool_calls,
        })
        for raw_call in tool_calls:
            try:
                call = parse_tool_call(raw_call)
                result = execute_tool(call.name, call.arguments)
            except Exception as exc:  # return errors to the model for self-correction
                result = {"error": type(exc).__name__, "message": str(exc)}
            step["tool_results"].append({
                "tool_call_id": raw_call["id"],
                "name": _tool_name(raw_call) or "unknown",
                "output": result,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": raw_call["id"],
                "name": _tool_name(raw_call) or "unknown",
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })
    raise RuntimeError(f"tool loop exceeded max_iterations={max_iterations}")
