"""Shared Zep client, request limits, mode selection, and retry policy.

Deployment configuration:

* ``ZEP_MODE`` — ``cloud`` (default) or ``local``. Selects the ingestion
  protocol: Zep Cloud's Batch API (``client.batch.*``) or the local OpenZep
  ``/graph-batch`` direct-POST path. ``local`` must be chosen explicitly.
* ``ZEP_BASE_URL`` — the API base URL. Defaults to Zep Cloud
  (``https://api.getzep.com/api/v2``). ``ZEP_MODE=local`` *requires* an
  explicit ``ZEP_BASE_URL`` (e.g. ``http://localhost:8000/api/v2``) so a
  deployment can never silently fall back to localhost.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Callable, Literal, TypeVar

import httpx
from zep_cloud.client import Zep
from zep_cloud.core.api_error import ApiError as ZepApiError

from ..config import Config
from .logger import get_logger

logger = get_logger("mirofish.zep")

T = TypeVar("T")

# Safe default: Zep Cloud. Any other endpoint must be configured explicitly
# via ZEP_BASE_URL; local OpenZep additionally requires ZEP_MODE=local.
ZEP_CLOUD_BASE_URL = "https://api.getzep.com/api/v2"
ZEP_MODE_ENV = "ZEP_MODE"
ZEP_BASE_URL_ENV = "ZEP_BASE_URL"
ZepMode = Literal["cloud", "local"]
# Keep request behavior aligned with the zep-cloud 3.25.0 SDK default that
# MiroFish used before introducing the shared client. This is an internal
# integration policy, not a deployment setting users need to tune.
ZEP_HTTP_REQUEST_TIMEOUT_SECONDS = 60.0
# Zep ingestion is asynchronous and may take several minutes. Preserve the
# original GraphBuilder deadline while keeping it separate from HTTP timeout.
ZEP_INGESTION_WAIT_TIMEOUT_SECONDS = 7200
MAX_ZEP_SEARCH_QUERY_CHARS = 400
MAX_ZEP_SEARCH_RESULTS = 50


def get_zep_mode() -> ZepMode:
    """Return the configured Zep integration mode (default: ``cloud``)."""

    raw = (os.environ.get(ZEP_MODE_ENV) or "cloud").strip().lower()
    if raw not in {"cloud", "local"}:
        raise ValueError(
            f"{ZEP_MODE_ENV} must be 'cloud' or 'local', got {raw!r}"
        )
    return raw  # type: ignore[return-value]


def is_local_zep_mode() -> bool:
    """Return whether the explicit local OpenZep protocol is selected."""

    return get_zep_mode() == "local"


def get_zep_base_url() -> str:
    """Return the configured Zep base URL without a trailing slash.

    Zep Cloud is the default. ``ZEP_MODE=local`` has no default URL: it must
    point at an explicitly configured OpenZep server.
    """

    raw = (os.environ.get(ZEP_BASE_URL_ENV) or "").strip()
    if is_local_zep_mode():
        if not raw:
            raise ValueError(
                f"{ZEP_MODE_ENV}=local requires {ZEP_BASE_URL_ENV} to point at "
                "the local OpenZep server (e.g. http://localhost:8000/api/v2)"
            )
    elif not raw:
        raw = ZEP_CLOUD_BASE_URL
    return raw.rstrip("/")


def normalize_zep_search_query(query: Any) -> str:
    """Return a non-empty query within Zep Cloud's endpoint limit."""

    if not isinstance(query, str):
        raise ValueError("Zep search query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("Zep search query must not be empty")
    return normalized[:MAX_ZEP_SEARCH_QUERY_CHARS]


def normalize_zep_search_limit(limit: Any) -> int:
    """Clamp a search result limit to the current Zep Cloud contract."""

    try:
        normalized = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Zep search limit must be an integer") from exc
    if normalized < 1:
        raise ValueError("Zep search limit must be at least 1")
    return min(normalized, MAX_ZEP_SEARCH_RESULTS)


@lru_cache(maxsize=4)
def _cached_zep_client(api_key: str, timeout: float, base_url: str) -> Zep:
    return Zep(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


def get_zep_client(api_key: str | None = None, timeout: float | None = None) -> Zep:
    """Return a process-shared, explicitly configured Zep client."""

    # zep-cloud gives ZEP_API_URL precedence even when base_url is explicit.
    # Reject it so the SDK cannot silently override the configured
    # ZEP_BASE_URL / Zep Cloud default.
    if os.environ.get("ZEP_API_URL"):
        raise ValueError(
            "ZEP_API_URL is unsupported (the zep-cloud SDK gives it silent "
            "precedence); configure ZEP_BASE_URL instead"
        )

    normalized_key = (api_key or Config.ZEP_API_KEY or "").strip()
    if not normalized_key:
        raise ValueError("ZEP_API_KEY 未配置")

    request_timeout = float(
        timeout if timeout is not None else ZEP_HTTP_REQUEST_TIMEOUT_SECONDS
    )
    if request_timeout <= 0:
        raise ValueError("Zep request timeout must be greater than 0")
    return _cached_zep_client(
        normalized_key, request_timeout, get_zep_base_url()
    )


def clear_zep_client_cache() -> None:
    """Clear cached clients. Intended for tests and controlled reconfiguration."""

    _cached_zep_client.cache_clear()


def is_retryable_zep_error(error: BaseException) -> bool:
    """Return whether a failed *read* is safe and useful to retry."""

    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    if isinstance(error, ZepApiError):
        status_code = error.status_code
        return status_code in {408, 429} or (
            status_code is not None and 500 <= status_code <= 599
        )
    return False


def _retry_after_seconds(error: BaseException) -> float | None:
    if not isinstance(error, ZepApiError) or not error.headers:
        return None
    value = next(
        (
            header_value
            for header_name, header_value in error.headers.items()
            if header_name.lower() == "retry-after"
        ),
        None,
    )
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def call_zep_read_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry a safe Zep read only for transport, 408, 429, or 5xx errors."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or not is_retryable_zep_error(error):
                raise

            retry_after = _retry_after_seconds(error)
            delay = min(
                retry_after if retry_after is not None else initial_delay * (2 ** (attempt - 1)),
                max_delay,
            )
            logger.warning(
                "Zep %s attempt %s/%s failed (%s); retrying in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                type(error).__name__,
                delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")
