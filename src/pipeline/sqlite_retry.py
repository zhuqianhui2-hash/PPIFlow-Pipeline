from __future__ import annotations

import sqlite3
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def lock_retry_deadline(
    *,
    busy_timeout_ms: int,
    minimum_seconds: float = 30.0,
    multiplier: float = 3.0,
) -> float:
    return time.time() + max(float(minimum_seconds), (float(busy_timeout_ms) / 1000.0) * float(multiplier))


def is_retryable_lock_error(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def run_with_lock_retry(
    fn: Callable[[], T],
    *,
    busy_timeout_ms: int,
    minimum_seconds: float = 30.0,
    multiplier: float = 3.0,
    initial_sleep_s: float = 0.05,
    max_sleep_s: float = 0.5,
    jitter: bool = True,
    jitter_scale: float = 0.25,
) -> T:
    deadline = lock_retry_deadline(
        busy_timeout_ms=busy_timeout_ms,
        minimum_seconds=minimum_seconds,
        multiplier=multiplier,
    )
    sleep_s = max(float(initial_sleep_s), 0.001)
    while True:
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if is_retryable_lock_error(exc) and time.time() < deadline:
                wait_s = sleep_s
                if jitter:
                    span = max(0.001, wait_s * float(jitter_scale))
                    wait_s = random.uniform(max(0.001, wait_s - span), wait_s + span)
                time.sleep(wait_s)
                sleep_s = min(sleep_s * 2.0, float(max_sleep_s))
                continue
            raise
