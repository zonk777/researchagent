"""学术调研工具注册中心。

构建 research agent 使用的工具集，包含:
    - 学术工具 (ArXiv, Semantic Scholar)
    - 通用工具 (Calculator, Bash, WebSearch)
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from researchagent.core.state import RuntimeState
from researchagent.tools.arxiv_tool import search_arxiv
from researchagent.tools.bash_tool import bash_tool_description, run_bash
from researchagent.tools.calculator_tool import evaluate
from researchagent.tools.semantic_scholar import search_semantic_scholar
from researchagent.tools.web_search_tool import web_search


def build_research_tools(state: RuntimeState) -> list[StructuredTool]:
    """构建学术调研 Agent 的完整工具集。

    包含通用工具（计算器/Bash/搜索）和学术专用工具。
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
            description="Evaluate a mathematical expression safely. Input: math expression string.",
        ),
        StructuredTool.from_function(
            name="BashTool",
            func=lambda command, timeout_seconds=None: run_bash(state, command, timeout_seconds),
            description=bash_tool_description(),
        ),
        StructuredTool.from_function(
            name="WebSearchTool",
            func=web_search,
            description="Search the web with Tavily. Use for non-academic information.",
        ),
    ]
