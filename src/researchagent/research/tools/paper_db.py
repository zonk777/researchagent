"""论文向量数据库。

基于 LanceDB 存储论文的 embedding 向量和元数据，
支持语义搜索、去重和相关度评分。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)


class PaperDB:
    """论文向量数据库。

    复用与 long_term.py 相同的 LanceDB + BGE-M3 技术栈。
    """

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self._embeddings = None
        self._table: Any = None
        self._initialized = False

    def add_paper(self, paper: dict) -> bool:
        """存储一篇论文到向量库。

        Args:
            paper: 论文 dict，含 title, authors, summary/abstract, year, url 等。
        """
        self._ensure_initialized()
        try:
            text = f"{paper.get('title', '')} {paper.get('summary', paper.get('abstract', ''))}"
            vector = self._embeddings.embed_query(text)
            self._table.add([{
                "vector": vector,
                "title": paper.get("title", ""),
                "authors": ", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), list) else str(paper.get("authors", "")),
                "summary": paper.get("summary", paper.get("abstract", ""))[:1000],
                "year": str(paper.get("year", paper.get("published", "")[:4])),
                "url": paper.get("url", ""),
                "citation_count": int(paper.get("citationCount", paper.get("citation_count", 0))),
                "source": paper.get("source", "unknown"),
                "added_at": datetime.now(timezone.utc).isoformat(),
            }])
            return True
        except Exception as e:
            logger.warning("Failed to add paper '%s': %s", paper.get("title", "")[:50], e)
            return False

    def search_similar(self, query: str, k: int = 5) -> list[dict]:
        """语义搜索相关论文。"""
        self._ensure_initialized()
        try:
            vector = self._embeddings.embed_query(query)
            results = self._table.search(vector).limit(k).to_list()
            return [{"title": r["title"], "authors": r["authors"],
                     "summary": r["summary"][:300], "year": r["year"],
                     "url": r["url"], "citation_count": r["citation_count"]}
                    for r in results]
        except Exception as e:
            logger.warning("Paper search failed: %s", e)
            return []

    def count(self) -> int:
        self._ensure_initialized()
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    def get_all_papers(self) -> list[dict]:
        """返回所有论文的元数据（用于趋势分析）。"""
        self._ensure_initialized()
        try:
            return self._table.to_pandas().to_dict("records")
        except Exception:
            return []

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        _cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
        if (_cache_dir / "snapshots").is_dir():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from langchain_huggingface import HuggingFaceEmbeddings

        self._embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.store_path.mkdir(parents=True, exist_ok=True)
        conn = lancedb.connect(str(self.store_path))
        try:
            self._table = conn.open_table("papers")
        except Exception:
            self._table = conn.create_table("papers", schema=pa.schema([
                pa.field("vector", pa.list_(pa.float32(), 1024)),
                pa.field("title", pa.string()),
                pa.field("authors", pa.string()),
                pa.field("summary", pa.string()),
                pa.field("year", pa.string()),
                pa.field("url", pa.string()),
                pa.field("citation_count", pa.int32()),
                pa.field("source", pa.string()),
                pa.field("added_at", pa.string()),
            ]))
        self._initialized = True
