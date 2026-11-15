from __future__ import annotations

import json
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from common.tools import calculate as _calculate
from common.tools import filter_courses as _filter_courses
from common.tools import get_course_detail as _get_course_detail
from common.tools import search_courses as _search_courses


@tool
def search_courses(keyword: str):
    """根据关键词搜索课程。"""
    return _search_courses(keyword)


@tool
def get_course_detail(course_id: str):
    """根据课程编号查询课程详情。"""
    return _get_course_detail(course_id)


@tool
def filter_courses(department: str | None = None, min_credits: int | None = None):
    """按院系和最低学分筛选课程。"""
    return _filter_courses(department, min_credits)


@tool
def calculate(expression: str):
    """计算安全的四则运算表达式。"""
    return _calculate(expression)


TOOLS = [search_courses, get_course_detail, filter_courses, calculate]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def format_tool_error(error: Exception) -> str:
    """Return every tool error to the model in the same JSON shape as Part A.

    ToolNode's default handler only catches argument validation errors and
    re-raises exceptions thrown inside the tool, which would crash the graph.
    """
    return json.dumps({"error": type(error).__name__, "message": str(error)}, ensure_ascii=False)


def build_graph(model):
    """Build model → tools → model graph; model must implement bind_tools/invoke."""
    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: AgentState):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=format_tool_error))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


class NeutralModel:
    """Adapt a framework-neutral model function to LangGraph's bind_tools/invoke protocol.

    The wrapped function returns {"content": ..., "tool_calls": [{"id", "name", "arguments"}]},
    the same format Part A's run_loop consumes, so both versions can replay the same script.
    """

    def __init__(self, request_model: Callable[[list[BaseMessage]], dict[str, Any]]):
        self.request_model = request_model

    def bind_tools(self, tools):
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        response = self.request_model(messages)
        return AIMessage(
            content=response.get("content") or "",
            tool_calls=[
                {"id": call["id"], "name": call["name"], "args": call.get("arguments", {}), "type": "tool_call"}
                for call in response.get("tool_calls") or []
            ],
        )


def run_agent(user: str, model, *, max_iterations: int = 8) -> tuple[str, list[BaseMessage]]:
    """Run the graph; max_iterations counts model calls, like Part A's run_loop.

    Measured on this LangGraph version: N model calls need recursion_limit=2N
    (N model steps, N - 1 tool steps, plus one step for the initial input).
    """
    result = build_graph(model).invoke(
        {"messages": [HumanMessage(content=user)]},
        {"recursion_limit": 2 * max_iterations},
    )
    messages = result["messages"]
    return str(messages[-1].content), messages


def offline_graph():
    """A framework-only graph for smoke tests without a model provider.

    Replace this model with a real chat model in class. The deterministic model
    implements the same bind_tools/invoke protocol used by LangGraph.
    """

    def deterministic(messages: list[BaseMessage]) -> dict[str, Any]:
        if len(messages) == 1:
            return {"content": None, "tool_calls": [{"id": "lg-1", "name": "search_courses", "arguments": {"keyword": "人工智能"}}]}
        if isinstance(messages[-1], ToolMessage):
            return {"content": "已完成工具查询。", "tool_calls": []}
        return {"content": "已完成。", "tool_calls": []}

    return build_graph(NeutralModel(deterministic))
