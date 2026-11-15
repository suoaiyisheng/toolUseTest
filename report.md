# Tool-use Agent 机制分析报告

> 写作说明（提交前删除本段）：每节下的「要点」是提纲，请用自己的话改写成段落，
> 并尽量引用本仓库中的具体代码、测试或运行输出作为证据。「证据」一栏列出了可以引用的位置。

## 1. 实现概述

### Part A：从零实现（`from_scratch/agent.py`）

要点：
- 工具注册：`TOOL_REGISTRY`（名字 → 函数）与 `TOOL_SPECS`（JSON Schema）；同一份 Schema 既发给模型，也用于校验。
- 处理一个 tool call 的三个阶段：解析（`parse_tool_call` / `parse_arguments`）→ 校验（`execute_tool` 查注册表 + `validate_arguments`）→ 执行。
- `run_loop`：调用模型 → 无 tool call 则返回 → 否则追加 assistant 消息、逐个执行、追加 tool 消息 → 下一轮；`max_iterations` 防止无限循环。
- 错误分类：解析错误（`ToolCallParseError`）、校验错误（`ValueError` / `TypeError`）、工具执行错误（如 `LookupError`）都作为 tool 消息回传；模型适配器返回非字典属于程序错误，直接抛出。

### Part B：LangGraph 版（`langgraph_version/agent.py`）

要点：
- `@tool` 暴露同样的四个工具；`StateGraph` 的 `model` / `tools` 两个节点；`tools_condition` 决定继续还是结束。
- 用 `NeutralModel` 把与框架无关的模型函数适配成 `bind_tools` / `invoke` 协议，使两版能回放同一份用例脚本。
- `handle_tool_errors=format_tool_error`：让工具内部异常也回传给模型（为什么需要它，见第 4 节）。
- `recursion_limit = 2 * max_iterations`：实测得到的换算关系。

## 2. 框架对象与 Part A 代码的对应关系

| LangGraph | Part A |
|---|---|
| `@tool`（从函数签名生成 Schema） | `TOOL_SPECS` + `TOOL_REGISTRY` |
| `ToolNode` 内部的 Pydantic 参数校验 | `validate_arguments` |
| `ToolNode` 内部的工具名检查 | `execute_tool` 中的 `if name not in TOOL_REGISTRY` |
| `ToolNode(handle_tool_errors=...)` | `run_loop` 中的 `try/except` |
| `ToolMessage(status="error")` | `{"role": "tool", "content": {"error": ..., "message": ...}}` |
| `tools_condition` | `if not tool_calls: return answer, trace` |
| `add_edge("tools", "model")` | `for iteration in range(max_iterations)` 进入下一轮 |
| `add_messages` | `messages.append(...)` |
| `recursion_limit` | `max_iterations` |
| `AIMessage.tool_calls`（已解析的字典） | `parse_tool_call` / `parse_arguments` |

## 3. 报告问题

### 3.1 模型和工具分别负责什么？

要点：
- 模型：理解用户意图，决定是否调用工具、调用哪个、填什么参数，并把工具结果组织成回答。模型**不执行**任何代码。
- 工具（及调用它的程序）：确定性地执行操作、访问数据；程序负责校验与权限控制。
- 可以举 `calculate` 的例子：让模型心算不可靠，交给工具算；而且工具用 AST 只允许四则运算，不会执行任意代码。

证据：`common/tools.py` 的 `calculate`；`tests/test_tools.py::test_unknown_tool_and_safe_calculator`。

### 3.2 为什么工具结果必须作为新消息回传？

要点：
- 模型是无状态的，每次调用只能看到传入的消息历史；结果不写回，模型就"不知道"工具返回了什么。
- `tool_call_id` 把结果和请求一一对应，同一轮多个调用时尤其重要；OpenAI API 也要求每个 tool call 都有对应的 tool 消息。
- 错误也要回传，模型才能自我修正。

证据：`tests/test_from_scratch.py::test_model_can_self_correct_after_error`（第一次缺参数，看到错误后补上参数成功）；`test_every_tool_call_has_a_matching_tool_message`。

### 3.3 Tool call 与普通文本有什么区别？

要点：
- 普通文本给人看；tool call 是给程序看的结构化数据（工具名 + 参数），存放在独立字段 `tool_calls` 中，而不是混在 `content` 里。
- 程序可以对 tool call 做解析和校验，对文本则只能"猜"。
- 但 tool call 仍是模型生成的，可能出错：OpenAI 格式中参数是 JSON 字符串，可能不合法、缺字段、类型不对。

证据：`test_invalid_json_arguments`、`test_missing_tool_name`、`test_openai_format_with_json_string_arguments`。

### 3.4 如果工具描述缺少参数约束，会出现什么问题？

要点：
- 模型可能漏传必需参数、传错类型、编造不存在的参数（如给 `search_courses` 加 `limit`）。
- 程序端没有约束就无法校验：错误要么在工具内部以难懂的形式出现（举改进前的例子：`'int' object has no attribute 'strip'`），要么被静默接受（`min_credits=True` 被当成 1）。
- 本仓库的对比实验：同样传 `min_credits="3"`，Part A 拒绝（`TypeError`），LangGraph 的 Pydantic 宽松模式会自动转成 3 并执行成功。讨论：自动转换对模型更宽容，但对金额等参数可能危险。

证据：用 `python -m from_scratch.demo --case missing_argument` 和 `python -m langgraph_version.demo --case missing_argument` 的输出作对比（自行粘贴）。

### 3.5 框架版隐藏了哪些步骤？调试时怎样观察这些步骤？

要点：
- 隐藏的步骤：Schema 生成、参数 JSON 解析、参数校验、工具分发、错误包装、消息追加、循环判断（对照第 2 节的表）。
- 隐藏的默认行为可能出乎意料：本版本 `ToolNode` 默认只处理参数校验错误，工具内部异常（如查询 `CS999` 时的 `LookupError`）会让整个图崩溃；必须显式传入 `handle_tool_errors`。
- `recursion_limit` 统计的是图的步数而非模型调用次数，实测 N 次模型调用需要 2N 步，与直觉推算的 2N - 1 不同。
- 观察方法：打印完整的 `result["messages"]`；用 `graph.stream(...)` 逐节点查看；`get_graph().draw_mermaid()` 导出图结构；阅读框架源码（如 `langgraph/prebuilt/tool_node.py` 中的 `_default_handle_tool_errors`）；用测试固定行为。

证据：`tests/test_langgraph.py::test_tool_exception_does_not_crash_graph`；`langgraph_version/agent.py` 中 `run_agent` 的说明。可以把删掉 `handle_tool_errors` 后测试失败的输出贴进来。

### 3.6 真实系统中，删除文件、支付、发消息等工具需要什么确认和权限控制？

要点（请结合自己的思考展开）：
- 最小权限：只注册当前任务需要的工具；未注册工具一律拒绝（本仓库 `unknown_tool` 用例）。
- 区分只读工具和有副作用的工具；有副作用的工具在执行前需要用户明确确认（human-in-the-loop），例如 LangGraph 的 `interrupt`。
- 参数要严格校验，不做自动类型转换；设置金额、范围等上限。
- 权限由程序根据用户身份判断，不能依赖模型"自觉"；把工具结果当作不可信输入，防止提示注入诱导调用危险工具。
- 审计日志（本仓库的 trace 就是雏形）、幂等性、可撤销或延迟执行；限制调用次数（`max_iterations`）。

## 4. 两版实现的差异

| 情况 | Part A | Part B（LangGraph） |
|---|---|---|
| 缺少参数 | `ValueError: get_course_detail 缺少必需参数: course_id` | `ToolInvocationError: ... course_id: Field required` |
| 未知工具 | JSON 格式的 `ValueError` | 纯文本 `Error: delete_files is not a valid tool...` |
| `min_credits="3"` | 拒绝（`TypeError`） | 自动转为 3，执行成功 |
| 工具内部异常 | 回传给模型 | 默认会让图崩溃，需配置 `handle_tool_errors` |

要点：说明一致性测试比较的粒度（最终答案、工具名、成功/失败、成功时的输出），以及为什么不要求错误类型名完全相同。

## 5. 测试与调用轨迹

- 运行方式：`python -m pytest -q`，共 66 个测试：Part A 32 个，Part B 24 个，两版一致性 10 个。
- 10 个共享用例定义在 `common/cases.py`，两版都用 `scripted_model` 回放同一份脚本；`python -m from_scratch.demo --all` 和 `python -m langgraph_version.demo --all` 可以逐步展示。
- 测试检验的是 Agent 循环（分发、校验、错误回传、消息结构），而不是模型的决策质量；说明这样设计的理由和局限（`scripted_model` 不读取工具结果）。
- 调用轨迹：`traces/example_run.json`，包含完整消息历史和每轮的模型回复与工具结果。
- 流程图：`diagram.md`。
