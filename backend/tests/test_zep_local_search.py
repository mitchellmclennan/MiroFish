"""Local OpenZep /graph/search scoping tests.

The local OpenZep compatibility server's ``GraphSearchRequest`` accepts
``session_id`` and ignores ``graph_id``/``scope``; without ``session_id``
its graphiti search runs across *every* graph on the server. These tests
prove that in ``ZEP_MODE=local`` MiroFish sends the local contract —
``session_id=<graph_id>`` and never ``graph_id``/``scope`` — with the same
auth, timeout and limit policies as the SDK path, that local payloads are
parsed into facts, and that Zep Cloud mode keeps the SDK call exactly as it
was.
"""

import pytest
import httpx
from types import SimpleNamespace
from zep_cloud.core.api_error import ApiError as ZepApiError

from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.zep_entity_reader import EntityNode
from app.services.zep_tools import ZepToolsService
from app.utils import zep_local_search
from app.utils.zep import ZEP_HTTP_REQUEST_TIMEOUT_SECONDS
from app.utils.zep_local_search import (
    build_local_graph_search_payload,
    local_graph_search,
)

LOCAL_BASE_URL = "http://localhost:8000/api/v2"
GRAPH_ID = "graph-test"
ENTITY_UUID = "entity-uuid-1"


def _local_mode(monkeypatch):
    monkeypatch.setenv("ZEP_MODE", "local")
    monkeypatch.setenv("ZEP_BASE_URL", LOCAL_BASE_URL)


def _cloud_mode(monkeypatch):
    monkeypatch.setenv("ZEP_MODE", "cloud")
    monkeypatch.delenv("ZEP_BASE_URL", raising=False)


class RecordingTransport:
    """Transport double capturing every request instead of hitting httpx."""

    def __init__(self, body=None, error=None):
        self.requests = []
        self._body = body
        self._error = error

    def __call__(self, *, url, payload, headers, timeout):
        self.requests.append(
            {
                "url": url,
                "payload": payload,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if self._error is not None:
            raise self._error
        return self._body


class ForbiddenSdkGraph:
    """SDK double proving the SDK search is never called in local mode."""

    class _Graph:
        def search(self, **kwargs):
            raise AssertionError(
                f"zep-cloud SDK graph.search must not be called in local "
                f"mode; got kwargs={kwargs}"
            )

    def __init__(self):
        self.graph = self._Graph()


class RecordingSdkGraph:
    """SDK double recording the exact kwargs of every search call."""

    def __init__(self, results=None):
        self.calls = []
        self._results = results or SimpleNamespace(edges=[], nodes=[], results=None)

        class _Graph:
            def search(_self, **kwargs):
                self.calls.append(kwargs)
                return self._results

        self.graph = _Graph()


def _entity(name="Marcus R."):
    return EntityNode(
        uuid=ENTITY_UUID,
        name=name,
        labels=["Person", "Entity"],
        summary="An order was drafted.",
        attributes={},
        related_edges=[],
        related_nodes=[],
    )


def _make_generator(monkeypatch, zep_client):
    monkeypatch.setenv("MIROFISH_LLM_LANGUAGE", "en")
    generator = OasisProfileGenerator(
        api_key="test-key",
        zep_api_key="test-zep-key",
        graph_id=GRAPH_ID,
    )
    generator.zep_client = zep_client
    return generator


def _install_transport(monkeypatch, transport):
    """Route the adapter through the recording transport for this test."""
    monkeypatch.setattr(
        zep_local_search, "_httpx_post_json", transport, raising=True
    )


# ── payload contract ─────────────────────────────────────────────────────────


def test_payload_carries_session_id_not_graph_id_or_scope():
    payload = build_local_graph_search_payload(
        query="All information about NeoLife", graph_id=GRAPH_ID, limit=30
    )

    # The local server only scopes its search when session_id is set
    assert payload["session_id"] == GRAPH_ID
    # graph_id/scope would be silently ignored by the local server
    assert "graph_id" not in payload
    assert "scope" not in payload
    assert payload["query"] == "All information about NeoLife"
    assert payload["limit"] == 30


def test_payload_applies_shared_query_and_limit_policies():
    long_query = "q" * 401

    payload = build_local_graph_search_payload(
        query="  padded query  ", graph_id=GRAPH_ID, limit=999
    )
    assert payload["query"] == "padded query"

    payload = build_local_graph_search_payload(
        query=long_query, graph_id=GRAPH_ID, limit=999
    )
    assert payload["query"] == long_query[:400]
    assert payload["limit"] == 50  # MAX_ZEP_SEARCH_RESULTS


@pytest.mark.parametrize("bad_graph_id", ["", "   ", None])
def test_payload_requires_a_real_graph_id(bad_graph_id):
    with pytest.raises(ValueError):
        build_local_graph_search_payload(
            query="q", graph_id=bad_graph_id, limit=10
        )


# ── adapter request behavior ──────────────────────────────────────────────────


def test_local_search_sends_api_key_auth_and_shared_timeout(monkeypatch):
    _local_mode(monkeypatch)
    transport = RecordingTransport(body={"results": []})
    _install_transport(monkeypatch, transport)

    local_graph_search(
        query="q",
        graph_id=GRAPH_ID,
        limit=10,
        api_key="the-key",
    )

    request = transport.requests[0]
    assert request["url"] == LOCAL_BASE_URL + "/graph/search"
    assert request["headers"]["Authorization"] == "Api-Key the-key"
    assert request["headers"]["Content-Type"] == "application/json"
    # Same request timeout policy as the shared SDK client
    assert request["timeout"] == ZEP_HTTP_REQUEST_TIMEOUT_SECONDS


def test_local_search_respects_explicit_base_url_and_timeout(monkeypatch):
    _local_mode(monkeypatch)
    transport = RecordingTransport(body={"results": []})
    _install_transport(monkeypatch, transport)

    local_graph_search(
        query="q",
        graph_id=GRAPH_ID,
        limit=10,
        api_key="the-key",
        base_url="http://elsewhere:9000/api/v2/",
        timeout=12.5,
    )

    request = transport.requests[0]
    assert request["url"] == "http://elsewhere:9000/api/v2/graph/search"
    assert request["timeout"] == 12.5


def test_local_search_requires_local_mode(monkeypatch):
    _cloud_mode(monkeypatch)
    transport = RecordingTransport(body={"results": []})
    _install_transport(monkeypatch, transport)

    with pytest.raises(RuntimeError, match="ZEP_MODE=local"):
        local_graph_search(query="q", graph_id=GRAPH_ID, limit=10)


def test_local_search_requires_an_api_key(monkeypatch):
    _local_mode(monkeypatch)
    monkeypatch.setattr(
        "app.utils.zep_local_search.Config", SimpleNamespace(ZEP_API_KEY=None)
    )

    with pytest.raises(ValueError, match="ZEP_API_KEY"):
        local_graph_search(query="q", graph_id=GRAPH_ID, limit=10)


def test_local_search_rejects_non_positive_timeout(monkeypatch):
    _local_mode(monkeypatch)
    transport = RecordingTransport(body={"results": []})
    _install_transport(monkeypatch, transport)

    with pytest.raises(ValueError, match="timeout"):
        local_graph_search(
            query="q", graph_id=GRAPH_ID, limit=10, api_key="k", timeout=0
        )


def test_local_search_raises_zep_api_error_on_http_error_status(monkeypatch):
    _local_mode(monkeypatch)
    transport = RecordingTransport(
        body={"results": []},
        error=ZepApiError(status_code=429, body="rate limited"),
    )
    _install_transport(monkeypatch, transport)

    with pytest.raises(ZepApiError) as error:
        local_graph_search(query="q", graph_id=GRAPH_ID, limit=10, api_key="k")
    # retry classification must see the status (429/5xx are retryable)
    assert error.value.status_code == 429


# ── adapter result parsing ─────────────────────────────────────────────────────


def test_local_search_parses_fact_dicts_from_results_key(monkeypatch):
    _local_mode(monkeypatch)
    body = {
        "results": [
            {"uuid": "f1", "fact": "neolife offers Starter at $699/mo.", "score": 0.9},
            {"uuid": "f2", "fact": "  ", "score": 0.4},
            "not-a-dict",
        ]
    }
    transport = RecordingTransport(body=body)
    _install_transport(monkeypatch, transport)

    result = local_graph_search(
        query="q", graph_id=GRAPH_ID, limit=10, api_key="k"
    )

    # The adapter keeps dict items verbatim (dict result handling is the
    # callers' job: empty facts are filtered there); non-dict junk is
    # dropped defensively.
    assert result.results == [
        {"uuid": "f1", "fact": "neolife offers Starter at $699/mo.", "score": 0.9},
        {"uuid": "f2", "fact": "  ", "score": 0.4},
    ]


def test_local_search_tolerates_missing_results_key(monkeypatch):
    _local_mode(monkeypatch)
    transport = RecordingTransport(body={"unexpected": True})
    _install_transport(monkeypatch, transport)

    result = local_graph_search(
        query="q", graph_id=GRAPH_ID, limit=10, api_key="k"
    )
    assert result.results == []


# ── profile generation: local mode ────────────────────────────────────────────


def test_profile_local_searches_carry_session_id_not_graph_id_or_scope(monkeypatch):
    _local_mode(monkeypatch)
    transport = RecordingTransport(
        body={"results": [{"uuid": "f1", "fact": "neolife ships orders."}]}
    )
    _install_transport(monkeypatch, transport)
    generator = _make_generator(monkeypatch, ForbiddenSdkGraph())

    results = generator._search_zep_for_entity(_entity())

    assert len(transport.requests) == 2  # edge search + node search
    for request in transport.requests:
        assert request["payload"]["session_id"] == GRAPH_ID
        assert "graph_id" not in request["payload"]
        assert "scope" not in request["payload"]
    # both searches still run, with the same result limits as before
    assert sorted(r["payload"]["limit"] for r in transport.requests) == [20, 30]
    # local fact dicts are parsed into the grounding results
    assert results["facts"] == ["neolife ships orders."]
    assert results["attempted"] is True


def test_profile_local_search_grounds_persona_with_scoped_facts(monkeypatch):
    _local_mode(monkeypatch)
    transport = RecordingTransport(
        body={
            "results": [
                {"uuid": "f1", "fact": "Starter pricing is $699/mo + $4.00/order."}
            ]
        }
    )
    _install_transport(monkeypatch, transport)
    generator = _make_generator(monkeypatch, ForbiddenSdkGraph())

    grounding = generator._build_grounding_context(_entity())

    facts = [f["text"] for f in grounding.facts]
    assert "Starter pricing is $699/mo + $4.00/order." in facts
    search_facts = [f for f in grounding.facts if f["source"] == "zep_search"]
    assert search_facts


def test_profile_local_search_retries_transient_transport_errors(monkeypatch):
    _local_mode(monkeypatch)
    import threading

    lock = threading.Lock()
    calls = []

    def flaky_transport(*, url, payload, headers, timeout):
        with lock:
            calls.append(payload)
            # Each of the two parallel searches retries its first
            # (transient) failure once: the first two calls fail, the
            # remaining calls succeed.
            should_fail = len(calls) <= 2
        if should_fail:
            raise httpx.ConnectError("connection refused")
        return {"results": [{"uuid": "f1", "fact": "recovered fact"}]}

    _install_transport(monkeypatch, flaky_transport)
    monkeypatch.setattr("app.utils.zep.time.sleep", lambda _s: None)
    generator = _make_generator(monkeypatch, ForbiddenSdkGraph())

    results = generator._search_zep_for_entity(_entity())

    assert len(calls) == 4  # 2 searches × 2 attempts (1 retry each)
    assert "recovered fact" in results["facts"]


def test_profile_local_search_fails_fast_on_auth_errors(monkeypatch):
    _local_mode(monkeypatch)
    calls = []

    def unauthorized_transport(*, url, payload, headers, timeout):
        calls.append(payload)
        raise ZepApiError(status_code=401, body="unauthorized")

    _install_transport(monkeypatch, unauthorized_transport)
    generator = _make_generator(monkeypatch, ForbiddenSdkGraph())

    with pytest.raises(ZepApiError):
        generator._search_zep_for_entity(_entity())

    # 401 is not retryable: exactly one attempt per search operation —
    # two searches total, never the 3-attempt retry budget.
    assert len(calls) == 2


# ── profile generation: Cloud SDK contract unchanged ───────────────────────────


def test_profile_cloud_mode_keeps_sdk_call_exactly_as_before(monkeypatch):
    _cloud_mode(monkeypatch)
    sdk = RecordingSdkGraph()
    transport = RecordingTransport(body={"results": []})
    _install_transport(monkeypatch, transport)
    generator = _make_generator(monkeypatch, sdk)

    generator._search_zep_for_entity(_entity())

    assert len(sdk.calls) == 2
    assert sdk.calls[0] == {
        "query": "All information, activities, events, relationships and background about Marcus R.",
        "graph_id": GRAPH_ID,
        "limit": 30,
        "scope": "edges",
        "reranker": "rrf",
    }
    assert sdk.calls[1] == {
        "query": "All information, activities, events, relationships and background about Marcus R.",
        "graph_id": GRAPH_ID,
        "limit": 20,
        "scope": "nodes",
        "reranker": "rrf",
    }
    # the local adapter must never be used in cloud mode
    assert transport.requests == []


# ── report tools: local mode ───────────────────────────────────────────────────


def test_report_search_local_mode_scopes_and_parses_local_results(monkeypatch):
    _local_mode(monkeypatch)
    transport = RecordingTransport(
        body={"results": [{"uuid": "f1", "fact": "neolife offers Starter."}]}
    )
    _install_transport(monkeypatch, transport)

    service = object.__new__(ZepToolsService)
    service.client = ForbiddenSdkGraph()

    result = service.search_graph(GRAPH_ID, "pricing", limit=10)

    request = transport.requests[0]
    assert request["payload"]["session_id"] == GRAPH_ID
    assert "graph_id" not in request["payload"]
    assert "scope" not in request["payload"]
    # local fact dicts are parsed instead of being silently dropped
    assert result.facts == ["neolife offers Starter."]
    assert result.edges == [
        {
            "uuid": "f1",
            "name": "",
            "fact": "neolife offers Starter.",
            "source_node_uuid": "",
            "target_node_uuid": "",
        }
    ]
    assert result.total_count == 1


def test_report_search_cloud_mode_keeps_sdk_call_exactly_as_before(monkeypatch):
    _cloud_mode(monkeypatch)
    sdk = RecordingSdkGraph()
    transport = RecordingTransport(body={"results": []})
    _install_transport(monkeypatch, transport)

    service = object.__new__(ZepToolsService)
    service.client = sdk

    service.search_graph(GRAPH_ID, "pricing", limit=10, scope="nodes")

    assert sdk.calls == [
        {
            "graph_id": GRAPH_ID,
            "query": "pricing",
            "limit": 10,
            "scope": "nodes",
            "reranker": "cross_encoder",
        }
    ]
    assert transport.requests == []
