# ResearchAgent

从零构建的 LLM Agent 框架，基于 LangGraph 实现**规划-执行-反思**三阶段闭环。

## 架构

```
用户输入 → planner → agent ⇄ tools → reflector → {通过→输出, 未通过→重试}
```

- **Planner**: 拆解任务为子步骤 + 验收标准
- **Agent**: ReAct 循环，调用 LLM 决定使用哪个工具
- **Tools**: 计算器 (AST 安全求值)、Bash (Shell 执行)、WebSearch (Tavily)
- **Reflector**: 检查结果，未通过自动重试（最多 3 次）

## 快速开始

### 环境要求
- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
git clone https://github.com/zonk777/researchagent.git
cd researchagent
uv sync
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env` 填入 API 信息：

```
API_KEY=sk-xxxxxxxx
MODEL=deepseek-v4-pro
BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=tvly-xxxxxxxx   # 从 https://tavily.com 获取
```

### 运行

```bash
# 命令行
uv run researchagent run "计算 123 * 456"
uv run researchagent run "搜索 Python 最新版本并总结"

# Web 界面
uv run python webui.py          # 访问 http://localhost:7860

# 单独测试工具
uv run researchagent tool-test -t calculator "sqrt(144)"
uv run researchagent tool-test -t bash "dir"
uv run researchagent tool-test -t search "LangGraph"
```

### 测试

```bash
uv run pytest tests/ -v         # 89 个测试
```

### Docker

```bash
docker-compose up
```

## 工具

| 工具 | 能力 | 安全 |
|------|------|------|
| CalculatorTool | +-*/ sqrt sin cos log 等 | AST 白名单求值 |
| BashTool | Shell 命令执行 | 危险命令拦截 + 工作区隔离 |
| WebSearchTool | Tavily 联网搜索 | API Key 认证 |

## Benchmark

10 任务 / 4 难度等级 / 100% 通过率

```bash
uv run python benchmarks/e2e_bench.py      # 端到端
uv run python benchmarks/compare_langchain.py  # vs LangChain
uv run python benchmarks/ablation.py          # 消融实验
```

## 技术栈

Python 3.13 / LangGraph / LanceDB / BGE-M3 / DeepSeek API / Gradio / Docker

## 项目结构

```
src/researchagent/
├── cli/           # CLI (Typer + Rich)
├── core/          # 状态、日志、重试、Token追踪、LLM工厂
├── graph/         # LangGraph 4 节点 pipeline
├── memory/        # 短期缓冲 + 长期向量库 (LanceDB)
├── tools/         # Calculator / Bash / WebSearch + 工具注册
├── prompts/       # Agent / Planner / Reflector 提示词
└── providers/     # OpenAI 兼容 LLM 接口
```
