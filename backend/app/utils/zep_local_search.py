"""Local OpenZep graph-search adapter (session_id-scoped ``/graph/search``).

Why this module exists
----------------------
MiroFish's Zep Cloud search path calls ``zep_client.graph.search(
query=..., graph_id=..., scope=...)``. The *local OpenZep compatibility
server* implements ``POST /graph/search`` with a different request model
(``GraphSearchRequest``: ``query`` / ``user_id`` / ``session_id`` /
``limit`` / ``min_score``): it ignores ``graph_id`` and ``scope`` entirely
and only scopes its underlying ``graphiti.search`` call when ``session_id``
is set — without it the search runs across **every** group stored on the
server. Calling the SDK method against the local server therefore returned
facts from all graphs on the server, contaminating fresh personas with
facts from older, unrelated graphs.

In ``ZEP_MODE=local`` only, MiroFish therefore POSTs the local contract
directly — ``{"query": ..., "session_id": <graph_id>, "limit": ...}`` — so
the search is scoped to exactly one graph. The request keeps the same
policies as the SDK path: the same API-key authentication, the shared
``ZEP_HTTP_REQUEST_TIMEOUT_SECONDS`` request timeout, the shared
``call_zep_read_with_retry`` retry policy (applied by callers), and the
shared ``normalize_zep_search_query`` / ``normalize_zep_search_limit``
query and result caps.

Zep Cloud mode never uses this module and keeps the SDK call exactly as it
was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx
from zep_cloud.core.api_error import ApiError as ZepApiError

from ..config import Config
from .logger import get_logger
from .zep import (
    ZEP_HTTP_REQUEST_TIMEOUT_SECONDS,
    get_zep_base_url,
    is_local_zep_mode,
    normalize_zep_search_limit,
    normalize_zep_search_query,
)

logger = get_logger("mirofish.zep_local_search")

# The local OpenZep compatibility endpoint (relative to ZEP_BASE_URL).
LOCAL_GRAPH_SEARCH_ENDPOINT = "/graph/search"

# Signature of the injectable transport. Tests use this seam to capture the
# request instead of monkeypatching httpx internals; production always uses
# the default httpx transport below.
PostJson = Callable[..., Any]


@dataclass(frozen=True)
class LocalGraphSearchResults:
    """Result of one local ``/graph/search`` request.

    Mirrors what the zep-cloud SDK exposes when it receives the same local
    payload (``GraphSearchResults`` carries the ``results`` key as an extra
    attribute): ``.results`` is a list of plain fact dicts
    (``{"uuid", "fact", "score", "metadata"}``). The local contract has no
    ``edges``/``nodes`` scopes — every result is an edge fact.
    """

    results: List[Dict[str, Any]] = field(default_factory=list)


def build_local_graph_search_payload(
    *, query: str, graph_id: str, limit: int
) -> Dict[str, Any]:
    """Build the local ``/graph/search`` request body.

    The scoping contract is deliberate: ``session_id`` carries the graph id
    (the only field the local server turns into a graphiti group filter),
    and ``graph_id`` / ``scope`` are **not** sent — the local server would
    silently ignore them.
    """

    if not isinstance(graph_id, str) or not graph_id.strip():
        raise ValueError("graph_id must be a non-empty string")
    return {
        "query": normalize_zep_search_query(query),
        "session_id": graph_id.strip(),
        "limit": normalize_zep_search_limit(limit),
    }


def _extract_results(body: Any) -> List[Dict[str, Any]]:
    """Normalize a local ``/graph/search`` response into fact dicts.

    The local server returns ``{"results": [{"uuid", "fact", "score",
    "metadata"}, ...]}``. Dict items are kept as-is (the callers' existing
    dict result handling reads ``fact`` from them); non-dict items are
    skipped defensively. A missing ``results`` key yields an empty list.
    """

    results = body.get("results") if isinstance(body, dict) else getattr(body, "results", None)
    if results is None:
        return []
    if not isinstance(results, list):
        raise ValueError(
            f"OpenZep /graph/search returned an unexpected 'results' shape: "
            f"{type(results).__name__}"
        )
    return [item for item in results if isinstance(item, dict)]


def _httpx_post_json(
    *,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> Any:
    """Default transport: one direct POST, raising on HTTP error statuses.

    HTTP error responses are raised as ``ZepApiError`` so the shared
    ``is_retryable_zep_error`` policy classifies them identically to SDK
    failures (408/429/5xx retryable, everything else fatal).
    """

    response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    if response.status_code >= 400:
        raise ZepApiError(
            headers={name: value for name, value in response.headers.items()},
            status_code=response.status_code,
            body=response.text,
        )
    return response.json()


def local_graph_search(
    *,
    query: str,
    graph_id: str,
    limit: int,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    post_json: Optional[PostJson] = None,
) -> LocalGraphSearchResults:
    """Run one session_id-scoped ``/graph/search`` request against the local
    OpenZep server (local mode only).

    Auth, request timeout, query/result limits, and the result dict shape
    all match the shared SDK-path policies; retrying is the caller's job —
    wrap this callable in ``call_zep_read_with_retry`` exactly like the SDK
    reads, so transport/408/429/5xx failures retry under the same budget.
    ``post_json`` is an injection seam for tests; production resolves it
    lazily so tests can monkeypatch the module's httpx transport.
    """

    if not is_local_zep_mode():
        raise RuntimeError(
            "local_graph_search must only be used in ZEP_MODE=local; "
            "Zep Cloud searches must use the zep-cloud SDK (graph_id scope)"
        )

    transport = post_json or _httpx_post_json

    normalized_key = (api_key or Config.ZEP_API_KEY or "").strip()
    if not normalized_key:
        raise ValueError("ZEP_API_KEY 未配置")

    effective_base_url = (base_url or get_zep_base_url()).rstrip("/")
    request_timeout = float(
        timeout if timeout is not None else ZEP_HTTP_REQUEST_TIMEOUT_SECONDS
    )
    if request_timeout <= 0:
        raise ValueError("Zep request timeout must be greater than 0")

    payload = build_local_graph_search_payload(
        query=query, graph_id=graph_id, limit=limit
    )
    body = transport(
        url=effective_base_url + LOCAL_GRAPH_SEARCH_ENDPOINT,
        payload=payload,
        # Same authentication the zep-cloud SDK performs on its requests.
        headers={
            "Authorization": f"Api-Key {normalized_key}",
            "Content-Type": "application/json",
        },
        timeout=request_timeout,
    )
    results = _extract_results(body)
    logger.debug(
        "OpenZep本地 /graph/search 完成: session_id=%s, 返回 %d 条事实",
        payload["session_id"],
        len(results),
    )
    return LocalGraphSearchResults(results=results)
