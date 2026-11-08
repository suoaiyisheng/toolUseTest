from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
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


def build_graph(model):
    """Build model → tools → model graph; model must implement bind_tools/invoke."""
    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: AgentState):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


def offline_graph():
    """A framework-only graph for smoke tests without a model provider.

    Replace this model with a real chat model in class. The deterministic model
    implements the same bind_tools/invoke protocol used by LangGraph.
    """
    class DeterministicModel:
        def bind_tools(self, tools):
            self.tools = tools
            return self

        def invoke(self, messages):
            if len(messages) == 1:
                from langchain_core.messages import AIMessage
                return AIMessage(content="", tool_calls=[{"name": "search_courses", "args": {"keyword": "人工智能"}, "id": "lg-1", "type": "tool_call"}])
            last = messages[-1]
            if isinstance(last, ToolMessage):
                return __import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(content="已完成工具查询。")
            from langchain_core.messages import AIMessage
            return AIMessage(content="已完成。")

    return build_graph(DeterministicModel())
