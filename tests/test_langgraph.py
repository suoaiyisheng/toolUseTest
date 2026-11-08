from langchain_core.messages import HumanMessage

from langgraph_version.agent import offline_graph


def test_langgraph_offline_smoke():
    result = offline_graph().invoke({"messages": [HumanMessage(content="搜索人工智能课程")]}, {"recursion_limit": 8})
    assert result["messages"][-1].content == "已完成工具查询。"

