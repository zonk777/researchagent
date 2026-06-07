"""OpenAI 兼容 API 提供者。

本模块提供 `create_model()` 工厂函数，根据 `.env` 配置文件创建
langchain `ChatOpenAI` 实例。参考 MokioAgent 的 `providers/openai_provider.py`。

环境变量:
    API_KEY (必需): API 密钥
    MODEL (必需): 模型名称
    BASE_URL (必需): API 端点地址

用法:
    from researchagent.providers.openai_provider import create_model

    model = create_model()
    response = model.invoke("你好")
    print(response.content)
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def create_model(
    temperature: float = 0.0,
) -> ChatOpenAI:
    """从 .env 配置文件创建 ChatOpenAI 实例。

    自动加载 `.env` 文件中的配置，读取 API_KEY、MODEL、BASE_URL
    三个环境变量来初始化 ChatOpenAI 客户端。

    Args:
        temperature: 采样温度，范围 [0, 2]。默认 0.0 表示确定性输出，
            适合工具调用和任务执行场景。调高可增加输出多样性。

    Returns:
        配置完成的 ChatOpenAI 实例，可直接调用 `.invoke()` 进行推理。

    Raises:
        RuntimeError: 当 API_KEY、MODEL 或 BASE_URL 任一环境变量
            未设置时抛出，错误消息包含缺失的变量名列表。
    """
    load_dotenv()

    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL")
    base_url = os.getenv("BASE_URL")

    # 检查必需的配置项
    missing = [
        name
        for name, value in {
            "API_KEY": api_key,
            "MODEL": model,
            "BASE_URL": base_url,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"缺少必需的 .env 配置项: {', '.join(missing)}。"
            f"请复制 .env.example 为 .env 并填入你的 API 信息。"
        )

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )
