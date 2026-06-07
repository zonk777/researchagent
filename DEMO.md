# ResearchAgent Demo 指南

## 环境准备

```bash
# 确保 .env 已配置
cp .env.example .env
# 编辑 .env: 填入 API_KEY, MODEL, BASE_URL, TAVILY_API_KEY

# 安装依赖
uv sync
```

## Demo 1: 数学计算

```bash
# CLI 方式
uv run researchagent run "计算 (123 + 456) * 789"

# 脚本方式
uv run python demos/demo_calculate.py
```

**预期效果：** Agent 调用 CalculatorTool 计算表达式，返回准确结果。

## Demo 2: 信息研究

```bash
# CLI 方式
uv run researchagent run "搜索 LangGraph 最新版本和主要特性，用中文总结"

# 脚本方式
uv run python demos/demo_research.py
```

**预期效果：** Agent 调用 WebSearchTool 搜索网络，返回结构化摘要。

## Demo 3: 文件操作

```bash
# CLI 方式
uv run researchagent run "列出当前项目目录下的所有 Python 文件"

# 脚本方式
uv run python demos/demo_file_ops.py
```

**预期效果：** Agent 调用 BashTool 执行 Shell 命令，返回目录列表。

## 多工具组合

```bash
uv run researchagent run "搜索 DeepSeek V4 的最新消息，然后用计算器算一下它发布距今多少天"
```

**预期效果：** Agent 先搜索（WebSearchTool），再计算（CalculatorTool），最后给出综合回答。

## 全量测试

```bash
uv run pytest tests/ -v
```

预期: 80+ 测试全部通过，耗时 < 90 秒。
