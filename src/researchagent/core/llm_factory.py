"""LLM 实例工厂，提供单例模式的模型创建。

避免 graph 中每个节点重复创建 ChatOpenAI 实例。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from researchagent.providers.openai_provider import create_model


@lru_cache(maxsize=4)
def get_cached_model(temperature: float = 0.0) -> ChatOpenAI:
    """获取缓存的 LLM 实例。

    使用 lru_cache 确保同一 temperature 参数下只创建一个实例。
    LangChain ChatOpenAI 不是线程安全的，单线程 Agent 无问题。
    """
    return create_model(temperature=temperature)


def get_model_for_node(node_name: str = "agent") -> ChatOpenAI:
    """根据节点名称获取模型实例。

    所有节点当前使用相同 temperature=0.0，由 lru_cache 保证复用。
    """
    return get_cached_model(temperature=0.0)
