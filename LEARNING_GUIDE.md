# ResearchAgent 代码精读指南

> 按复杂度从低到高排列。每读完一个文件跑对应测试加深理解。15 节课、30 个文件。

## 第一部分：地基（3 节课）

### 第 1 课：LLM 接口

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/providers/openai_provider.py` | ~50 | `.env` → `ChatOpenAI` 工厂模式，为什么用 OpenAI 客户端连 DeepSeek |
| `src/researchagent/core/state.py` | ~70 | `@dataclass RuntimeState` — 全局配置载体，缓存字段设计 |
| `pyproject.toml` | ~30 | 项目元数据、依赖声明、CLI 入口点 |

**动手**：`uv run researchagent test "你好"`

### 第 2 课：辅助模块

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/core/retry.py` | ~55 | 装饰器实现指数退避重试，动态导入 OpenAI SDK 异常类型 |
| `src/researchagent/core/llm_factory.py` | ~20 | `lru_cache` 单例模式，避免重复创建 ChatOpenAI 实例 |
| `src/researchagent/core/tracker.py` | ~80 | `TokenUsage` dataclass + `TokenTracker.summary()` 按节点聚合 |
| `src/researchagent/core/logging_config.py` | ~30 | stderr 日志配置 + 第三方库噪音抑制 |

---

## 第二部分：工具系统（3 节课）

### 第 3 课：通用工具

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/tools/calculator_tool.py` | ~80 | AST 白名单安全求值（`simpleeval.SimpleEval`） |
| `src/researchagent/tools/web_search_tool.py` | ~65 | Tavily API 封装，`os.getenv()` 读取密钥 |
| `src/researchagent/tools/bash_tool.py` | ~270 | `subprocess.run()` + 20 种危险命令正则 + Windows 平台适配 |

**动手**：
```bash
uv run researchagent tool-test -t calculator "sqrt(144)"
uv run researchagent tool-test -t bash "dir"
uv run researchagent tool-test -t search "Python"
```

### 第 4 课：工具注册

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/tools/registry.py` | ~70 | `StructuredTool.from_function()` + **lambda 闭包注入 RuntimeState** |

**关键模式**：`func=lambda command: run_bash(state, command)` — state 通过闭包传递，LLM 不可见。

### 第 5 课：学术搜索工具

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/research/tools/arxiv_tool.py` | ~70 | ArXiv API（免费无 Key），`arxiv.Search()` + `SortCriterion` |
| `src/researchagent/research/tools/semantic_scholar.py` | ~110 | Semantic Scholar API + `get_paper_details()` 含引用/被引 |
| `src/researchagent/research/tools/paper_db.py` | ~90 | LanceDB 论文向量存储（复用 BGE-M3） |
| `src/researchagent/research/tools/registry.py` | ~60 | `build_research_tools()` — 5 个学术专用工具集 |

---

## 第三部分：记忆系统（2 节课）

### 第 6 课：短期记忆

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/memory/short_term.py` | ~120 | 对话缓冲 + token 估计 + LLM 摘要压缩 |

### 第 7 课：长期记忆 + 管理器

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/memory/long_term.py` | ~170 | LanceDB 原生 API + BGE-M3 1024 维 + 离线缓存检测 |
| `src/researchagent/memory/manager.py` | ~130 | `MemoryManager` 统一协调 + `build_context()` 增强 prompt |

**动手**：
```bash
uv run pytest tests/test_short_term.py tests/test_long_term.py tests/test_manager.py -v
```

---

## 第四部分：Agent 图 ⭐（3 节课）

### 第 8 课：图结构

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/graph/state.py` | ~40 | `AgentState(TypedDict)` + `add_messages` 归约器 |
| `src/researchagent/graph/workflow.py` | ~60 | `StateGraph` 4 节点编排 |

```
START → planner → agent → {tool_calls → tools → agent}
                          → {无 tool_calls → reflector}
                                    → {pass → END}
                                    → {not pass → agent 重试}
```

### 第 9 课：四个节点

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/graph/nodes.py` | ~450 | `planner_node` + `agent_node` + `tools_node` + `reflector_node` + `route_after_agent` + `reflector_route` |

**agent_node 逐行解析**：
1. 获取状态 → 2. `llm.bind_tools(tools)` → 3. `memory.build_context(task)` → 4. `_invoke_with_retry()` → 5. 返回结果

**动手**：
```bash
uv run researchagent run "计算 1+1" --no-stream
uv run pytest tests/test_graph.py -v
```

### 第 10 课：提示词

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/prompts/agent.py` | ~35 | 3 个 System Prompt + 防注入规则 + 中文强制 + 不写文件约束 |

---

## 第五部分：分析引擎（2 节课）

### 第 11 课：论文分析

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/research/analysis/citation_graph.py` | ~80 | 被引统计 + H-index 估算 |
| `src/researchagent/research/analysis/method_comparison.py` | ~55 | 正则匹配 15 种 ML 方法名称 |
| `src/researchagent/research/analysis/trend_analyzer.py` | ~70 | 年份分布 + 趋势判断 + 关键词 |
| `src/researchagent/research/analysis/runner.py` | ~30 | `run_full_analysis()` 统一入口 |

### 第 12 课：输出模块

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/research/output/bib_manager.py` | ~60 | BibTeX 条目生成/保存/解析 |
| `src/researchagent/research/output/latex_renderer.py` | ~75 | Jinja2 模板 → LaTeX 论文草稿 |
| `src/researchagent/research/output/paper_tracker.py` | ~50 | 论文增量追踪 |

---

## 第六部分：交付（2 节课）

### 第 13 课：CLI

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `src/researchagent/cli/app.py` | ~580 | Typer 5 个命令 + Rich 流式 + TokenTracker 集成 |

### 第 14 课：Web UI

| 文件 | 行数 | 学什么 |
|------|------|--------|
| `webui.py` | ~330 | Gradio Chatbot + 6 快捷按钮 + 会话持久化 + 线程执行/取消 + 并发锁 + XSS 防护 |

---

## 学习路线图

```
第 1-2 课:  基础设施 (7 文件, ~340 行)   — LLM 调用 + 状态管理
第 3-5 课:  工具系统 (7 文件, ~820 行)   — 工具注册 + 学术搜索
第 6-7 课:  记忆系统 (3 文件, ~420 行)   — 向量存储 + 对话管理
第 8-10 课: Agent 图 (4 文件, ~590 行) ⭐ — LangGraph 编排 + ReAct 循环
第 11-12 课: 分析引擎 (7 文件, ~420 行)  — 论文分析 + LaTeX 输出
第 13-14 课: CLI+WebUI (2 文件, ~910 行) — 交互层
```

| 部分 | 节课 | 文件数 | 总行数 |
|------|------|--------|--------|
| 地基 | 1-2 | 7 | ~340 |
| 工具 | 3-5 | 7 | ~820 |
| 记忆 | 6-7 | 3 | ~420 |
| Agent 图 | 8-10 | 4 | ~590 |
| 分析引擎 | 11-12 | 7 | ~420 |
| 交付 | 13-14 | 2 | ~910 |
| **合计** | **14** | **30** | **~3,500** |
