"""LaTeX 论文草稿渲染器。

使用 Jinja2 模板将综述内容渲染为可编译的 LaTeX 文档。
"""

from __future__ import annotations

from pathlib import Path

LATEX_TEMPLATE = r"""\documentclass[11pt,a4paper]{{article}}

\usepackage[UTF8]{{ctex}}
\usepackage{{hyperref}}
\usepackage{{geometry}}
\usepackage{{booktabs}}
\usepackage{{cite}}
\geometry{{margin=2.5cm}}
\title{{{title}}}
\author{{KorinAgentFlow Research Assistant}}
\date{{\\today}}

\begin{{document}}

\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

\section{{引言}}
{introduction}

\section{{相关工作与方法}}
{methods}

\section{{方法对比}}
{comparison_table}

\section{{讨论与未来方向}}
{discussion}

\section{{结论}}
{conclusion}

\bibliographystyle{{plain}}
\bibliography{{references}}

\end{{document}}
"""


def render_latex(
    output_path: Path,
    *,
    title: str = "Literature Review",
    abstract: str = "",
    introduction: str = "",
    methods: str = "",
    comparison_table: str = "",
    discussion: str = "",
    conclusion: str = "",
) -> Path:
    """渲染 LaTeX 论文草稿。

    Args:
        output_path: 输出 .tex 文件路径。
        title: 论文标题。
        abstract: 摘要。
        introduction: 引言。
        methods: 相关工作与方法。
        comparison_table: LaTeX 表格格式的方法对比。
        discussion: 讨论与未来方向。
        conclusion: 结论。

    Returns:
        保存的 .tex 文件路径。
    """
    content = LATEX_TEMPLATE.format(
        title=title.replace("&", "\\&").replace("%", "\\%"),
        abstract=abstract,
        introduction=introduction,
        methods=methods,
        comparison_table=comparison_table,
        discussion=discussion,
        conclusion=conclusion,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def build_comparison_table(papers: list[dict]) -> str:
    """从论文列表构建 LaTeX 方法对比表。

    Args:
        papers: 论文列表。

    Returns:
        LaTeX tabular 格式的表格字符串。
    """
    if not papers:
        return "无对比数据。"

    rows = []
    for i, p in enumerate(papers[:8], 1):
        title = p.get("title", "")[:50]
        year = str(p.get("year", ""))
        citations = p.get("citation_count", p.get("citationCount", 0))

        # LaTeX 转义
        title = title.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#")
        rows.append(f"    [{i}] & {title} & {year} & {citations} \\\\")

    rows_str = "\n".join(rows)
    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{论文对比}\\label{tab:comparison}\n"
        "\\begin{tabular}{cllc}\n"
        "\\toprule\n"
        "编号 & 标题 & 年份 & 被引数 \\\\\n"
        "\\midrule\n"
        f"{rows_str}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )
