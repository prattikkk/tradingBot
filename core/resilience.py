"""Shared resilience primitives for API-bound components."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass


class TokenBucketLimiter:
    """Thread-safe token bucket limiter with sync and async acquisition."""

    def __init__(self, rate_per_minute: int, capacity: float | None = None):
        safe_rate = max(1, int(rate_per_minute))
        self._rate_per_second = safe_rate / 60.0
        self._capacity = float(capacity if capacity is not None else safe_rate)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self, now: float) -> None:
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)
        self._last_refill = now

    def _wait_time_locked(self, tokens: float) -> float:
        now = time.monotonic()
        self._refill_locked(now)
        if self._tokens >= tokens:
            self._tokens -= tokens
            return 0.0
        missing = tokens - self._tokens
        return missing / self._rate_per_second

    def acquire(self, tokens: float = 1.0) -> None:
        needed = max(0.0, tokens)
        while True:
            with self._lock:
                wait_for = self._wait_time_locked(needed)
            if wait_for <= 0:
                return
            time.sleep(min(wait_for, 1.0))

    async def acquire_async(self, tokens: float = 1.0) -> None:
        needed = max(0.0, tokens)
        while True:
            with self._lock:
                wait_for = self._wait_time_locked(needed)
            if wait_for <= 0:
                return
            await asyncio.sleep(min(wait_for, 1.0))


@dataclass
class _CircuitState:
    failures: int = 0
    open_until: float = 0.0


class CircuitBreaker:
    """Simple endpoint-level circuit breaker."""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: int = 30):
        self._failure_threshold = max(1, int(failure_threshold))
        self._cooldown_seconds = max(1, int(cooldown_seconds))
        self._states: dict[str, _CircuitState] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            state = self._states.setdefault(key, _CircuitState())
            if state.open_until > now:
                return False
            if state.open_until and state.open_until <= now:
                state.open_until = 0.0
                state.failures = 0
            return True

    def record_success(self, key: str) -> None:
        with self._lock:
            state = self._states.setdefault(key, _CircuitState())
            state.failures = 0
            state.open_until = 0.0

    def record_failure(self, key: str) -> None:
        with self._lock:
            state = self._states.setdefault(key, _CircuitState())
            state.failures += 1
            if state.failures >= self._failure_threshold:
                state.open_until = time.monotonic() + self._cooldown_seconds
                state.failures = 0


def retry_delay_seconds(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with bounded ceiling."""
    base = max(base, 0.0)
    cap = max(cap, base)
    return min(base * (2 ** min(max(0, attempt), 30)), cap)
