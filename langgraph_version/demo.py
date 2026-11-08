from langchain_core.messages import HumanMessage

from .agent import offline_graph


if __name__ == "__main__":
    result = offline_graph().invoke({"messages": [HumanMessage(content="找人工智能课程")]}, {"recursion_limit": 8})
    for message in result["messages"]:
        print(type(message).__name__, ":", message.content)
