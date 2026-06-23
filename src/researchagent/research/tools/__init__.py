"""学术调研工具集。

提供学术论文搜索和存储能力：
    - ArxivSearchTool: ArXiv 预印本搜索（免费，无需 API Key）
    - SemanticScholarTool: Semantic Scholar 已发表论文搜索
    - PaperDB: 论文向量数据库（LanceDB + BGE-M3）
"""

from researchagent.research.tools.paper_db import PaperDB
from researchagent.research.tools.registry import build_research_tools

__all__ = ["build_research_tools", "PaperDB"]
