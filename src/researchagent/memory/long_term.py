"""长期记忆：向量语义搜索 (LanceDB + BGE-M3)。

LongTermMemory 提供基于语义的长期记忆存储和检索。
使用 LanceDB 原生 API（服务端内嵌、无需外部进程），
BGE-M3 作为 embedding 模型（1024维、中英文多语言）。

存储结构 (LanceDB 表):
    vector: float[1024]     # BGE-M3 embedding
    text: str               # 原始文本
    role: str               # user/assistant/system
    category: str           # interaction/fact/preference
    timestamp: str          # ISO 8601 格式
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import logging

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

# 在 huggingface 库被懒加载导入之前，检测本地缓存并设置离线模式
# huggingface 相关库在 _ensure_initialized() 中延迟导入，
# 因此此处设置的 HF_HUB_OFFLINE 将对其生效
_cache_base = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3"
if (_cache_base / "snapshots").is_dir():
    os.environ["HF_HUB_OFFLINE"] = "1"


class LongTermMemory:
    """向量长期记忆。

    使用 LanceDB 原生 API 存储和语义检索对话记录。

    使用方式:
        ltm = LongTermMemory(store_path=Path("./memory_store"))
        ltm.add_interaction("user", "我喜欢 Python", category="preference")
        results = ltm.search("用户喜欢什么编程语言？", k=3)
    """

    def __init__(
        self,
        store_path: Path,
        table_name: str = "long_term_memory",
    ) -> None:
        """初始化长期记忆。

        Args:
            store_path: LanceDB 数据持久化目录。
            table_name: LanceDB 表名。
        """
        self.store_path = store_path
        self.table_name = table_name
        self._connection: Any = None     # lancedb connection
        self._table: Any = None          # lancedb table
        self._embeddings = None          # HuggingFaceEmbeddings
        self._initialized = False

    # ---- 公共方法 ----

    def add_interaction(
        self,
        text: str,
        role: str = "user",
        category: str = "interaction",
    ) -> bool:
        """存储一条对话记录到长期记忆。

        Args:
            text: 对话文本内容。
            role: 消息角色。
            category: 分类标签 (interaction / fact / preference)。

        Returns:
            True 表示存储成功。
        """
        self._ensure_initialized()

        try:
            vector = self._embed_text(text)
            self._table.add([{
                "vector": vector,
                "text": text,
                "role": role,
                "category": category,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }])
            return True
        except Exception as e:
            logger.warning(
                "Failed to add interaction: %s (text=%r, role=%s)",
                e, text[:200], role,
            )
            return False

    def add_fact(self, fact: str) -> bool:
        """存储一条重要事实。"""
        return self.add_interaction(fact, role="system", category="fact")

    def add_preference(self, preference: str) -> bool:
        """存储一条用户偏好。"""
        return self.add_interaction(preference, role="user", category="preference")

    def search(
        self,
        query: str,
        k: int = 5,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """语义搜索长期记忆。

        Args:
            query: 搜索查询文本。
            k: 最多返回结果数。
            category: 可选的分类过滤。

        Returns:
            搜索结果列表，每项包含 text, role, category, timestamp。
        """
        self._ensure_initialized()

        try:
            vector = self._embed_text(query)

            builder = self._table.search(vector).limit(k)
            if category:
                builder = builder.where(f"category = '{category}'", prefilter=True)

            results = builder.to_list()

            return [
                {
                    "text": r.get("text", ""),
                    "role": r.get("role", ""),
                    "category": r.get("category", ""),
                    "timestamp": r.get("timestamp", ""),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(
                "Failed to search memory: %s (query=%r, k=%d)", e, query[:200], k,
            )
            return []

    def count(self) -> int:
        """返回存储的记忆条数。"""
        self._ensure_initialized()
        try:
            return self._table.count_rows()
        except Exception:
            return 0

    # ---- 内部方法 ----

    def _ensure_initialized(self) -> None:
        """懒加载: 首次使用前初始化 embedding 模型和 LanceDB 连接。"""
        if self._initialized:
            return

        from langchain_huggingface import HuggingFaceEmbeddings

        # BGE-M3: 多语言 (100+)、1024维、8192 token 上下文
        # 若模型未缓存，需先设置 HF_ENDPOINT 环境变量指向可用镜像下载
        self._embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # LanceDB 原生连接
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._connection = lancedb.connect(str(self.store_path))

        # 尝试打开或创建表
        try:
            self._table = self._connection.open_table(self.table_name)
        except Exception:
            # 创建空表 (首次使用)
            self._table = self._connection.create_table(
                self.table_name,
                schema=pa.schema([
                    pa.field("vector", pa.list_(pa.float32(), 1024)),
                    pa.field("text", pa.string()),
                    pa.field("role", pa.string()),
                    pa.field("category", pa.string()),
                    pa.field("timestamp", pa.string()),
                ]),
            )

        self._initialized = True

    def _embed_text(self, text: str) -> list[float]:
        """将文本转换为 embedding 向量。"""
        # HuggingFaceEmbeddings.embed_query 返回 list[float]
        return self._embeddings.embed_query(text)
