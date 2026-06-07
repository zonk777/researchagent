"""LLM 调用重试装饰器。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

_BUILTIN_RETRYABLE = (ConnectionError, TimeoutError, OSError)

# OpenAI SDK 可重试异常（不强制依赖 openai 包）
_OPENAI_RETRYABLE: tuple = ()
try:
    import openai  # noqa: F811
    _OPENAI_RETRYABLE = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )
except ImportError:
    pass

RETRYABLE_EXCEPTIONS = _BUILTIN_RETRYABLE + _OPENAI_RETRYABLE


def retry_llm(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
) -> Callable[[F], F]:
    """LLM 调用重试装饰器。

    对可重试的网络异常（ConnectionError、TimeoutError、OSError）
    使用指数退避策略自动重试。不可重试的异常直接抛出。

    Args:
        max_retries: 最大重试次数。
        base_delay: 初始等待秒数。
        backoff_factor: 退避倍数。
        max_delay: 最大等待秒数。
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"LLM call failed after {max_retries} retries: {e}"
                        )
                        raise
                    logger.warning(
                        f"LLM call attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                except Exception as e:
                    logger.error(f"Unretryable LLM error: {type(e).__name__}: {e}")
                    raise

            if last_exception:
                raise last_exception
            raise RuntimeError(
                "Unexpected: retry loop completed without return or exception"
            )

        return wrapper  # type: ignore[return-value]

    return decorator
