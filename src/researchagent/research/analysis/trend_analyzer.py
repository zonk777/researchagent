"""研究趋势分析。

分析论文发表年份分布和关键词趋势。
"""

from __future__ import annotations

from collections import Counter


def analyze_year_distribution(papers: list[dict]) -> dict:
    """分析论文年份分布。

    Args:
        papers: 论文列表。

    Returns:
        {"years": {year: count}, "range": (min, max), "trend": "growing/stable/declining"}
    """
    year_counter: Counter = Counter()
    for p in papers:
        year = str(p.get("year", ""))[:4]
        if year and year.isdigit():
            year_counter[year] += 1

    years = dict(sorted(year_counter.items()))
    if not years:
        return {"years": {}, "range": (0, 0), "trend": "insufficient data"}

    year_list = sorted(int(y) for y in years)
    return {
        "years": years,
        "range": (year_list[0], year_list[-1]),
        "trend": _classify_trend(years),
    }


def _classify_trend(years: dict) -> str:
    """判断发表趋势。"""
    if len(years) < 2:
        return "insufficient data"
    values = list(years.values())
    first_half = sum(list(values)[:len(values)//2])
    second_half = sum(list(values)[len(values)//2:])
    if second_half > first_half * 1.2:
        return "growing"
    elif second_half < first_half * 0.8:
        return "declining"
    return "stable"


def analyze_keywords(papers: list[dict], top_n: int = 10) -> dict:
    """提取论文标题中的高频关键词。"""
    stop_words = {"the", "a", "an", "for", "of", "in", "on", "to", "and", "with",
                  "using", "via", "from", "by", "is", "are", "was", "were", "be",
                  "based", "new", "towards", "toward", "through", "its"}

    word_counter: Counter = Counter()
    for p in papers:
        title = p.get("title", "").lower()
        # 提取单词（>=3 个字母）
        words = [w.strip(",:;.!?()[]{}") for w in title.split()
                 if len(w.strip(",:;.!?()[]{}")) >= 3
                 and w.strip(",:;.!?()[]{}") not in stop_words]
        word_counter.update(words)

    return {"keywords": dict(word_counter.most_common(top_n))}


def format_trend_report(year_data: dict, keywords: dict) -> str:
    """生成趋势分析报告。"""
    lines = ["## 研究趋势", ""]

    # 年份分布
    years = year_data.get("years", {})
    if years:
        lines.append("### 发表年份分布")
        lines.append("")
        max_count = max(years.values())
        for year, count in years.items():
            bar_len = int(count / max_count * 20)
            bar = "█" * bar_len
            lines.append(f"- {year}: {count} 篇 {bar}")
        lines.append(f"\n趋势: **{year_data.get('trend', 'N/A')}**")
        lines.append("")

    # 关键词
    kw = keywords.get("keywords", {})
    if kw:
        lines.append("### 高频关键词")
        lines.append("")
        for word, count in kw.items():
            lines.append(f"- **{word}**: {count} 次")
        lines.append("")

    return "\n".join(lines)
