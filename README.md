# Tool-use Agent：从零实现到 LangGraph

这是一个面向大语言模型课程的两阶段作业仓库。学生需要先手写 Tool-use 的核心机制，再使用常用框架完成同一个校园课程助手，并比较两种实现。

## 学习目标

完成后，学生应能解释并实现以下闭环：

```text
工具 Schema → 模型选择工具 → 结构化 Tool Call → Python 执行工具
    ↑                                               ↓
    └────────────── Tool Result 回到消息历史 ────────┘
```

本仓库借鉴公开教学材料的任务形态和术语，包括 [sst-cs/ai-assignment-5-ref](https://github.com/sst-cs/ai-assignment-5-ref)、[DeepLearning.AI Functions, Tools and Agents with LangChain](https://www.deeplearning.ai/courses/functions-tools-agents-langchain/)、[Prompt Engineering Guide Function Calling](https://www.promptingguide.ai/agents/function-calling) 和 [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/en/unit1/bonus-unit1/introduction)。代码、数据和作业要求为本仓库重新编写；参考来源见 [REFERENCES.md](REFERENCES.md)。

## 两阶段作业

### Part A：从零实现

目录：`from_scratch/`

学生不能使用 LangChain、LangGraph 或其他 Agent 编排框架，需要自己完成：

- 工具注册表和 JSON Schema；
- 工具名称与参数校验；
- `execute_tool()` 动态分发；
- 模型返回普通文本或 tool call 的解析；
- 工具结果追加为 `tool` 消息；
- 最大迭代次数、未知工具、非法参数和工具异常处理。

运行本地确定性演示：

```bash
python -m from_scratch.demo
```

### Part B：使用框架

目录：`langgraph_version/`

学生使用 `@tool`、`ToolNode`、`tools_condition` 和 LangGraph 状态图，完成与 Part A 相同的测试任务：

```bash
python -m langgraph_version.demo
```

如果安装的 LangGraph 版本 API 有差异，先运行 `pytest -q`，再根据报错调整适配代码。框架版仍然要求学生画出 `model → tools → model` 的节点关系，并指出每个框架对象对应 Part A 的哪一段代码。

## 安装

Python 3.11 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

只有需要连接真实模型时才需要设置 API Key。默认演示和测试不联网、不产生费用。

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"
```

## 作业任务

两个版本必须使用同一组工具和用例：

1. 搜索人工智能相关课程；
2. 查询 `CS101` 的详情；
3. 筛选计算机学院且学分不少于 3 的课程；
4. 计算课程学分总和；
5. 先搜索，再查询其中一门课程详情；
6. 调用不存在的工具；
7. 缺少必需参数；
8. 触发工具异常；
9. 连续多次工具调用；
10. 不需要工具、直接回答的问题。

完整要求见 [ASSIGNMENT.md](ASSIGNMENT.md)，统一数据和测试见 `common/`、`tests/`。

## 提交结构

```text
student-submission/
├── from_scratch/
├── langgraph_version/
├── tests/
├── traces/
│   └── example_run.json
├── diagram.png 或 diagram.md
└── report.md
```

学生不得提交 API Key、`.env`、真实个人数据或生产系统凭据。

## 评分

| 项目 | 分值 |
|---|---:|
| 原理版 Schema、注册表和动态执行 | 20 |
| 原理版 Agent Loop 与消息历史 | 25 |
| 错误处理和安全限制 | 10 |
| LangGraph 版实现 | 20 |
| 两版测试行为一致 | 10 |
| 日志、流程图和自动测试 | 5 |
| 机制分析报告 | 10 |
| **总计** | **100** |

## 教师使用建议

建议 Part A 占 60%，Part B 占 40%。先要求学生阅读并运行确定性演示，再允许接入模型。课堂展示时重点追踪一条完整消息历史，而不是只看最终答案。

