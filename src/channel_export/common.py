"""Shared plumbing: retrying request execution, call/quota accounting, raw JSON storage."""

from __future__ import annotations

import http.client
import json
import logging
import random
import socket
import ssl
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, TypeVar

from googleapiclient.errors import HttpError

log = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
# 403s that are rate limits (retry) as opposed to real permission/quota problems (don't).
RETRYABLE_403_REASONS = frozenset(
    {"rateLimitExceeded", "userRateLimitExceeded", "backendError", "internalError"}
)
QUOTA_EXHAUSTED_REASONS = frozenset({"quotaExceeded", "dailyLimitExceeded"})
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    socket.timeout,
    ssl.SSLError,
    http.client.HTTPException,
)

def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield consecutive slices of ``items`` with at most ``size`` elements."""
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def http_status(err: BaseException) -> int | None:
    """HTTP status code of an ``HttpError`` (None for other exceptions)."""
    if not isinstance(err, HttpError):
        return None
    try:
        return int(err.resp.status)
    except (AttributeError, TypeError, ValueError):
        return None


def _error_payload(err: HttpError) -> dict[str, Any]:
    content = getattr(err, "content", b"") or b""
    try:
        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        payload = json.loads(text)
    except (ValueError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def error_reason(err: BaseException) -> str | None:
    """First ``reason`` in a Google API error body, e.g. ``quotaExceeded`` or ``commentsDisabled``."""
    if not isinstance(err, HttpError):
        return None
    error = _error_payload(err).get("error")
    if not isinstance(error, dict):
        return None
    for item in error.get("errors") or []:
        if isinstance(item, dict) and item.get("reason"):
            return str(item["reason"])
    status = error.get("status")
    return str(status) if status else None


def error_message(err: BaseException) -> str:
    """Human-readable message for any exception, preferring the API's own message."""
    if isinstance(err, HttpError):
        error = _error_payload(err).get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        reason = getattr(err, "reason", None)
        if reason:
            return str(reason)
    return f"{type(err).__name__}: {err}"


def describe_error(err: BaseException) -> dict[str, Any]:
    """Compact, JSON-serialisable description of an exception for the manifest."""
    return {
        "type": type(err).__name__,
        "http_status": http_status(err),
        "reason": error_reason(err),
        "message": error_message(err),
    }


def is_retryable(err: BaseException) -> bool:
    """True for rate limits, 5xx responses and transient network failures."""
    if isinstance(err, HttpError):
        status = http_status(err)
        if status in RETRYABLE_STATUSES:
            return True
        return status == 403 and error_reason(err) in RETRYABLE_403_REASONS
    return isinstance(err, TRANSIENT_ERRORS)


def is_quota_exhausted(err: BaseException) -> bool:
    """True when the daily project quota is used up (retrying won't help until it resets)."""
    return isinstance(err, HttpError) and error_reason(err) in QUOTA_EXHAUSTED_REASONS


def execute_with_retry(
    request: Any,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 64.0,
    sleep: Callable[[float], None] = time.sleep,
    on_attempt: Callable[[], None] | None = None,
) -> Any:
    """Call ``request.execute()`` with exponential backoff (plus jitter) on 429/5xx/network errors."""
    attempt = 0
    while True:
        if on_attempt is not None:
            on_attempt()
        try:
            return request.execute()
        except Exception as err:  # noqa: BLE001 - filtered by is_retryable below
            if attempt >= max_retries or not is_retryable(err):
                raise
            delay = min(max_delay, base_delay * (2**attempt)) + random.uniform(0, base_delay)
            log.warning(
                "Transient error (%s); retry %d/%d in %.1fs",
                error_message(err),
                attempt + 1,
                max_retries,
                delay,
            )
            sleep(delay)
            attempt += 1
