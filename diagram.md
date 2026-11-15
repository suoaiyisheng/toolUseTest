# 流程图

## 1. Tool-use 闭环（两版共同的机制）

```mermaid
flowchart LR
    U[用户问题] --> H[(消息历史)]
    H --> M[模型]
    M -->|普通文本| A[最终回答]
    M -->|结构化 tool call| P[程序：解析与校验]
    P --> E[程序：执行 Python 工具]
    E -->|结果或错误作为 tool 消息| H
```

模型只决定"调用什么、传什么参数"，真正执行函数的是程序；工具结果必须写回消息历史，模型下一轮才能看到。

## 2. Part A：`from_scratch/agent.py` 中的 `run_loop`

```mermaid
flowchart TD
    S([开始]) --> C{"iteration < max_iterations?"}
    C -->|否| X([抛出 RuntimeError：超过最大迭代次数])
    C -->|是| R["request_model(messages)"]
    R --> T{"有 tool_calls?"}
    T -->|否| F["追加 assistant 最终回答"] --> D([返回 answer 和 trace])
    T -->|是| I["_ensure_id：补齐缺失的 id"]
    I --> AM["追加 assistant 消息（含 tool_calls）"]
    AM --> L["逐个处理 tool call"]
    L --> P1["parse_tool_call：取工具名、解析 JSON 参数"]
    P1 -->|失败| ERR["错误包装为 error 和 message"]
    P1 --> P2["execute_tool：工具名是否已注册"]
    P2 -->|未注册| ERR
    P2 --> P3["validate_arguments：按 TOOL_SPECS 校验"]
    P3 -->|不合法| ERR
    P3 --> P4["调用 TOOL_REGISTRY 中的函数"]
    P4 -->|工具抛异常| ERR
    P4 -->|成功| OK["结果转为 JSON"]
    ERR --> TM["追加 tool 消息（tool_call_id 对应）"]
    OK --> TM
    TM -->|还有调用| L
    TM -->|本轮处理完| C
```

每个 tool call 都会产生一条 tool 消息，无论成功还是失败；只有模型适配器返回非字典这类程序错误才会直接抛出。

## 3. Part B：`langgraph_version/agent.py` 中的状态图

由 `offline_graph().get_graph().draw_mermaid()` 从实际代码导出：

```mermaid
flowchart TD
    start([START]) --> model[model 节点：call_model]
    model -.->|tools_condition：有 tool_calls| tools["tools 节点：ToolNode(TOOLS, handle_tool_errors=format_tool_error)"]
    model -.->|tools_condition：无 tool_calls| finish([END])
    tools --> model
```

- 实线是固定边（`add_edge`），虚线是条件边（`add_conditional_edges` + `tools_condition`）。
- 每个节点返回的新消息由 `add_messages` 追加到 `AgentState.messages`。
- `recursion_limit = 2 * max_iterations` 限制总步数（实测 N 次模型调用需要 2N 步）。

## 4. 一次完整运行：用例 `search_then_detail`

```mermaid
sequenceDiagram
    participant U as 用户
    participant L as Agent 循环
    participant M as 模型
    participant T as 工具

    U->>L: 找一门人工智能课程并给出详情
    L->>M: 消息历史（1 条）
    M-->>L: tool call：search_courses(keyword=人工智能)
    L->>T: 校验后执行 search_courses
    T-->>L: [CS205 人工智能导论]
    L->>M: 消息历史（3 条）
    M-->>L: tool call：get_course_detail(course_id=CS205)
    L->>T: 校验后执行 get_course_detail
    T-->>L: CS205 详情（李老师，3 学分）
    L->>M: 消息历史（5 条）
    M-->>L: 普通文本：推荐 CS205 人工智能导论
    L-->>U: 最终回答
```

对应的完整消息历史见 `traces/example_run.json`。
