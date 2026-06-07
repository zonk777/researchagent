# ResearchAgent 代码精读指南

> 按复杂度从低到高排列，建议逐文件阅读，每读完一个运行对应测试加深理解。

## 学习顺序

### 第一层：基础设施（读 3 个文件，约 200 行）

先从最简单的单文件模块开始，没有依赖，纯函数。

| 顺序 | 文件 | 行数 | 学什么 |
|------|------|------|--------|
| 1 | `src/researchagent/providers/openai_provider.py` | ~40 | `.env` 加载 → `ChatOpenAI` 创建的工厂模式 |
| 2 | `src/researchagent/core/state.py` | ~70 | `@dataclass` 做全局配置载体 |
| 3 | `pyproject.toml` | ~30 | 项目元数据、依赖声明、CLI 入口点 |

**动手：**
```bash
uv run researchagent test "你好"
```
打断点看 `create_model()` 如何从环境变量构造 LLM 实例。

---

### 第二层：工具系统（读 5 个文件，约 600 行）

每个工具是独立的纯函数，`registry.py` 把它们组装起来。

| 顺序 | 文件 | 行数 | 学什么 |
|------|------|------|--------|
| 4 | `src/researchagent/tools/calculator_tool.py` | ~85 | AST 安全求值，`simpleeval` 的使用 |
| 5 | `src/researchagent/tools/web_search_tool.py` | ~75 | Tavily API 封装 |
| 6 | `src/researchagent/tools/bash_tool.py` | ~260 | `subprocess` 执行、危险命令正则、平台适配 |
| 7 | `tests/test_calculator.py` | ~100 | pytest 测试模式 |
| 8 | `src/researchagent/tools/registry.py` | ~70 | `StructuredTool.from_function()` + lambda 闭包注入 `RuntimeState` |

**关键概念——闭包注入：**
```python
# registry.py 的核心模式
def build_tools(state: RuntimeState) -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="BashTool",
            # state 通过闭包注入，LLM 看不到
            func=lambda command, timeout=None: run_bash(state, command, timeout),
        ),
    ]
```

**动手：**
```bash
uv run researchagent tool-test -t calculator "sqrt(144)"
uv run researchagent tool-test -t bash "dir"
uv run researchagent tool-test -t search "Python"
uv run pytest tests/test_calculator.py tests/test_bash_tool.py -v
```

---

### 第三层：记忆系统（读 4 个文件，约 500 行）

短期记忆（纯 Python 列表操作）和长期记忆（向量数据库）。

| 顺序 | 文件 | 行数 | 学什么 |
|------|------|------|--------|
| 9 | `src/researchagent/memory/short_term.py` | ~125 | 对话缓冲、token 估计、LLM 摘要压缩 |
| 10 | `src/researchagent/memory/long_term.py` | ~175 | LanceDB 原生 API、BGE-M3 embedding、语义搜索 |
| 11 | `src/researchagent/memory/manager.py` | ~150 | 统一协调双层记忆 |
| 12 | `tests/test_short_term.py` | ~95 | Mock LLM（FakeLLM）测试模式 |

**动手：**
```bash
uv run python -c "
from pathlib import Path
from researchagent.memory.short_term import ConversationBuffer
# 创建一个 FakeLLM 测试短期记忆
class FakeLLM:
    def invoke(self, msgs):
        from langchain_core.messages import AIMessage
        return AIMessage(content='[摘要] 用户进行了对话')
buf = ConversationBuffer(FakeLLM(), max_tokens=100)
buf.add_message('user', '什么是 Python？')
buf.add_message('assistant', 'Python 是一门编程语言。')
print(buf.get_context())
"
```

---

### 第四层：Agent 图（读 4 个文件，约 500 行）⭐ 核心

这里是整个项目的核心——LangGraph 的状态管理和节点编排。

| 顺序 | 文件 | 行数 | 学什么 |
|------|------|------|--------|
| 13 | `src/researchagent/graph/state.py` | ~45 | `AgentState(TypedDict)` + `add_messages` 归约器 |
| 14 | `src/researchagent/prompts/agent.py` | ~70 | 三个 System Prompt（Agent/Planner/Reflector） |
| 15 | `src/researchagent/graph/workflow.py` | ~60 | `StateGraph` 的节点和边如何构建 |
| 16 | `src/researchagent/graph/nodes.py` | ~380 | 四个节点的完整实现 |

**先读 workflow.py 理解图的拓扑结构，再逐个节点深入：**

```python
# 图结构一目了然
START → planner → agent → {tool_calls? → tools → agent}
                           → {无 tool_calls → reflector}
                                     → {通过 → END}
                                     → {未通过 → agent 重试}
```

**agent_node 逐行解析：**

1. **获取状态**：`runtime = state["runtime"]; messages = state["messages"]`
2. **创建 LLM + 绑定工具**：`llm.bind_tools(tools)` — 这行让 LLM 能决定调用哪个工具
3. **构建记忆上下文**：从 RuntimeState 缓存中取 MemoryManager，构建增强 prompt
4. **首次调用注入 System Prompt**：`SystemMessage(AGENT_SYSTEM_PROMPT)`
5. **调用 LLM**：`model.invoke(messages)` → 返回 `AIMessage`（可能包含 `tool_calls`）
6. **返回结果**：`{"messages": [response], "iteration_count": +1}`

**动手：**
```bash
# 逐步跟踪执行过程
uv run researchagent run "计算 1+1"   # agent → CalculatorTool → agent → reflector
uv run researchagent run "你好"       # agent → reflector (无工具调用)
uv run pytest tests/test_graph.py -v   # 路由逻辑测试
```

---

### 第五层：CLI + Web UI（读 2 个文件，约 600 行）

| 顺序 | 文件 | 行数 | 学什么 |
|------|------|------|--------|
| 17 | `src/researchagent/cli/app.py` | ~400 | Typer CLI 设计、Rich 格式化、流式渲染 |
| 18 | `webui.py` | ~250 | Gradio Chatbot 组件、线程管理、进度条 |

---

### 第六层：辅助模块 + Benchmark（读 4 个文件，约 500 行）

| 顺序 | 文件 | 行数 | 学什么 |
|------|------|------|--------|
| 19 | `src/researchagent/core/retry.py` | ~70 | 装饰器实现指数退避重试 |
| 20 | `src/researchagent/core/tracker.py` | ~70 | Token 用量数据类设计 |
| 21 | `src/researchagent/core/logging_config.py` | ~30 | logging 模块配置 |
| 22 | `src/researchagent/core/llm_factory.py` | ~25 | `lru_cache` 做单例模式 |

---

## 阅读技巧

### 1. 每读完一个文件跑对应的测试

```bash
uv run pytest tests/test_calculator.py -v   # 读 calculator_tool.py 后
uv run pytest tests/test_graph.py -v        # 读 graph/ 后
```

### 2. 用 `--no-stream` 看完整执行流程

```bash
uv run researchagent run "计算 123*456" --no-stream
```
输出会展示每一步：planner → agent → tools → agent → reflector。

### 3. 打断点或者加 print

在任意节点加一行 `print(f"DEBUG: {变量}")` 观察运行时状态。

### 4. 从测试倒推代码逻辑

比如先看 `tests/test_graph.py::test_route_after_agent_with_tool_calls`，理解它测什么，再去看 `route_after_agent` 的实现。

---

## 学习路线图

```
第一周: 1-8  (基础设施 + 工具系统)     → 理解 LLM 调用和工具注册
第二周: 9-12 (记忆系统)                → 理解向量存储和语义搜索
第三周: 13-16 (Agent 图) ⭐           → 理解 LangGraph 状态管理和 ReAct 循环
第四周: 17-22 (CLI + 辅助 + Benchmark) → 整合理解整个项目
```
