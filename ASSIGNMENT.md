# 作业：构建一个 Tool-use Agent

## 背景

普通聊天模型只能生成文本。Tool-use 让模型输出结构化调用请求，由外部程序执行函数，再把结果放回上下文。本作业要求你用两种方式实现相同的校园课程助手：一次从零实现，一次使用 LangGraph。

## Part A：从零实现（60%）

### 必须完成

1. 在 `from_scratch/agent.py` 中完成工具注册、Schema、参数校验和动态执行。
2. 实现一个有 `max_iterations` 的 Agent Loop。
3. 正确保留 assistant tool-call 消息和对应的 tool result 消息。
4. 对未知工具、缺少参数、错误类型、非法 JSON 和工具异常返回可读错误。
5. 用 `from_scratch/demo.py` 记录至少一条两步调用轨迹。

### 禁止事项

Part A 不得导入 LangChain、LangGraph、AutoGen、CrewAI 等 Agent 框架。可以使用模型厂商的基础聊天 SDK，也可以使用本仓库的 deterministic planner 完成离线测试。

## Part B：LangGraph 实现（40%）

1. 在 `langgraph_version/agent.py` 使用 `@tool` 暴露相同工具。
2. 使用状态图表达 `model → tools → model` 循环。
3. 使用 `ToolNode` 或等价的框架节点执行工具。
4. 设置递归/迭代上限，并保留消息历史。
5. 在报告中把框架对象映射到 Part A 的手写代码。

## 报告问题

- 模型和工具分别负责什么？
- 为什么工具结果必须作为新消息回传？
- Tool call 与普通文本有什么区别？
- 如果工具描述缺少参数约束，会出现什么问题？
- 框架版隐藏了哪些步骤？调试时怎样观察这些步骤？
- 真实系统中，删除文件、支付、发消息等工具需要什么确认和权限控制？

## 评分细则

除 README 中的总表外，教师可按以下标准扣分：无限循环、执行未注册工具、吞掉错误、把 API Key 写入仓库、只提交最终答案而没有调用轨迹，均应扣除对应部分分数。

