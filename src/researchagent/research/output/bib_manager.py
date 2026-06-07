"""BibTeX 引用管理器。

生成和管理 .bib 格式的参考文献。
"""

from __future__ import annotations

import re
from pathlib import Path


def generate_bib_entry(paper: dict, index: int) -> str:
    """为单篇论文生成 BibTeX 条目。

    Args:
        paper: 论文信息 dict，含 title, authors, year, url 等。
        index: 引用编号。

    Returns:
        BibTeX 格式的引用条目字符串。
    """
    title = paper.get("title", "Unknown Title")
    authors = paper.get("authors", ["Unknown Author"])
    year = paper.get("year") or paper.get("published", "")[:4] if paper.get("published") else ""
    url = paper.get("url", "")

    # 生成引用 key: first_author_lastnameYEAR_first_word
    first_author = authors[0] if isinstance(authors, list) else str(authors)
    last_name = first_author.split()[-1] if " " in first_author else first_author
    first_word = re.sub(r"[^a-zA-Z]", "", title.split()[0].lower()) if title.split() else "unknown"
    key = f"{last_name}{year}{first_word}"

    # 转义 LaTeX 特殊字符
    title = title.replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")

    author_str = " and ".join(authors[:5]) if isinstance(authors, list) else str(authors)
    if isinstance(authors, list) and len(authors) > 5:
        author_str += " and others"

    lines = [
        f"@article{{{key},",
        f"  title = {{{title}}},",
        f"  author = {{{author_str}}},",
    ]
    if year:
        lines.append(f"  year = {{{year}}},")
    lines.append(f"  journal = {{CoRR}},")
    if url:
        lines.append(f"  url = {{{url}}},")
    lines.append("}")
    return "\n".join(lines)


def save_bib_file(entries: list[str], output_path: Path) -> Path:
    """保存 .bib 文件。

    Args:
        entries: BibTeX 条目列表。
        output_path: 输出文件路径。

    Returns:
        保存的文件路径。
    """
    content = "\n\n".join(entries)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def parse_existing_bib(file_path: Path) -> list[str]:
    """解析已有 .bib 文件，返回条目列表（去重用）。"""
    if not file_path.exists():
        return []
    content = file_path.read_text(encoding="utf-8")
    # 按 @article 分割
    entries = re.split(r"\n(?=@)", content)
    return [e.strip() for e in entries if e.strip()]
