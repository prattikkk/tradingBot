"""
AlphaBot Retry Utility — exponential backoff decorator.
Max 3 retries for all external API calls.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Callable, Type, Tuple

from loguru import logger


def retry_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Async retry decorator with exponential backoff.

    Usage:
        @retry_async(max_retries=3, exceptions=(ConnectionError, TimeoutError))
        async def fetch_data():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"[RETRY] {func.__name__} failed after {max_retries} attempts: {e}"
                        )
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(
                        f"[RETRY] {func.__name__} attempt {attempt}/{max_retries} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
            raise last_exception  # Should not reach here
        return wrapper
    return decorator


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """Synchronous retry decorator with exponential backoff."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"[RETRY] {func.__name__} failed after {max_retries} attempts: {e}"
                        )
                        raise
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"[RETRY] {func.__name__} attempt {attempt}/{max_retries} "
                        f"failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator
