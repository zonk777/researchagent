"""工具注册中心 (Tool Registry)。

提供 `build_tools()` 和 `build_read_only_tools()` 两个工厂函数，
根据 RuntimeState 构建 LangChain StructuredTool 列表。

遵循 MokioAgent 的闭包注入模式：
    - RuntimeState 通过 lambda 闭包注入，对 LLM 不可见
    - StructuredTool.from_function() 使用 infer_schema=True (默认)
    - 工具的 JSON Schema 自动从函数参数类型注解推导
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from researchagent.core.state import RuntimeState
from researchagent.tools.bash_tool import bash_tool_description, run_bash
from researchagent.tools.arxiv_tool import search_arxiv
from researchagent.tools.calculator_tool import evaluate
from researchagent.tools.semantic_scholar import search_semantic_scholar
from researchagent.tools.web_search_tool import web_search


def build_tools(state: RuntimeState) -> list[StructuredTool]:
    """构建完整的工具集 (含写入能力)。

    包含:
        - ArxivSearchTool: ArXiv 学术论文搜索
        - SemanticScholarTool: Semantic Scholar 学术论文搜索
        - CalculatorTool: 安全数学计算
        - BashTool: Shell 命令执行
        - WebSearchTool: 网络搜索

    Args:
        state: 运行时状态，通过闭包注入到需要 state 的工具。

    Returns:
        StructuredTool 列表，可直接绑定到 LLM。
    """
    return [
        StructuredTool.from_function(
            name="ArxivSearchTool",
            func=search_arxiv,
            description=(
                "Search ArXiv for academic papers. "
                "Args: query (English keywords, supports boolean operators), "
                "max_results (1-20, default 5), "
                "sort_by (relevance/lastUpdatedDate/submittedDate). "
                "Returns papers with title, authors, summary, url, pdf_url, categories."
            ),
        ),
        StructuredTool.from_function(
            name="SemanticScholarTool",
            func=search_semantic_scholar,
            description=(
                "Search Semantic Scholar for published papers and conference proceedings. "
                "Broader coverage than ArXiv. "
                "Args: query (English keywords), max_results (1-20, default 5). "
                "Returns papers with title, authors, abstract, year, venue, citationCount, url."
            ),
        ),
        StructuredTool.from_function(
            name="CalculatorTool",
            func=lambda expression: evaluate(expression),
            description=(
                "Evaluate a mathematical expression safely. "
                "Supports +, -, *, /, **, //, % and functions: "
                "sqrt, sin, cos, tan, log, log10, log2, exp, "
                "ceil, floor, abs, round. Constants: pi, e. "
                "Input: a single math expression string like '2+3*4' or 'sqrt(144)'."
            ),
        ),
        StructuredTool.from_function(
            name="BashTool",
            func=lambda command, timeout_seconds=None: run_bash(
                state, command, timeout_seconds
            ),
            description=bash_tool_description(),
        ),
        StructuredTool.from_function(
            name="WebSearchTool",
            func=web_search,
            description=(
                "Search the web with Tavily and return structured results. "
                "Args: query (search string), max_results (1-10, default 5), "
                "include_answer (bool, default True for AI summary). "
                "Returns AI-generated answer plus result sources with title, url, content, and score."
            ),
        ),
    ]


def build_read_only_tools(state: RuntimeState) -> list[StructuredTool]:
    """构建只读工具集 (不含 BashTool 等写入类工具)。

    适用于验证、检查等不需要修改系统的场景。

    Args:
        state: 运行时状态。

    Returns:
        只读 StructuredTool 列表。
    """
    return [
        StructuredTool.from_function(
            name="ArxivSearchTool",
            func=search_arxiv,
            description="Search ArXiv for academic papers. Read-only.",
        ),
        StructuredTool.from_function(
            name="SemanticScholarTool",
            func=search_semantic_scholar,
            description="Search Semantic Scholar for papers. Read-only.",
        ),
        StructuredTool.from_function(
            name="CalculatorTool",
            func=lambda expression: evaluate(expression),
            description="Evaluate a mathematical expression safely.",
        ),
        StructuredTool.from_function(
            name="WebSearchTool",
            func=web_search,
            description="Search the web with Tavily.",
        ),
    ]
