from types import SimpleNamespace
import io
import json
import socket
import urllib.error

import httpx
import pytest
from zep_cloud import Zep
from zep_cloud.core.api_error import ApiError as ZepApiError

from app.services import graph_builder as graph_builder_module
from app.services.graph_builder import (
    BatchSubmission,
    GraphBuilderService,
    PermanentEpisodePollError,
)
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.zep_entity_reader import EntityNode, ZepEntityReader
from app.services.zep_tools import ZepToolsService

LOCAL_BASE_URL = "http://localhost:8000/api/v2"


def _local_mode(monkeypatch):
    """Select the explicit local OpenZep protocol with a configured URL."""

    monkeypatch.setenv("ZEP_MODE", "local")
    monkeypatch.setenv("ZEP_BASE_URL", LOCAL_BASE_URL)


def _cloud_mode(monkeypatch):
    """Select the default Zep Cloud Batch API protocol."""

    monkeypatch.setenv("ZEP_MODE", "cloud")
    monkeypatch.delenv("ZEP_BASE_URL", raising=False)


class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://zep.test", status, "error", None, io.BytesIO(b"")
    )


def test_report_search_caps_the_query_sent_to_zep():
    calls = []

    class GraphApi:
        def search(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(edges=[], nodes=[])

    service = object.__new__(ZepToolsService)
    service.client = SimpleNamespace(graph=GraphApi())

    original_query = "q" * 401
    result = service.search_graph("graph-id", original_query)

    assert calls[0]["query"] == original_query[:400]
    assert result.query == original_query


def test_profile_context_search_caps_both_queries_sent_to_zep():
    calls = []

    class GraphApi:
        def search(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(edges=[], nodes=[])

    generator = object.__new__(OasisProfileGenerator)
    generator.zep_client = SimpleNamespace(graph=GraphApi())
    generator.graph_id = "graph-id"

    entity = EntityNode(
        uuid="node-id",
        name="n" * 500,
        labels=["Entity", "Person"],
        summary="",
        attributes={},
    )
    generator._search_zep_for_entity(entity)

    assert len(calls) == 2
    assert all(0 < len(call["query"]) <= 400 for call in calls)


def test_entity_context_includes_incoming_edges_from_the_full_graph():
    incoming = {
        "uuid": "edge-in",
        "name": "WORKS_AT",
        "fact": "Alice works at Acme",
        "source_node_uuid": "alice",
        "target_node_uuid": "acme",
        "attributes": {},
    }
    outgoing = {
        "uuid": "edge-out",
        "name": "BUILDS",
        "fact": "Acme builds Product",
        "source_node_uuid": "acme",
        "target_node_uuid": "product",
        "attributes": {},
    }
    unrelated = {
        "uuid": "edge-unrelated",
        "name": "LOCATED_IN",
        "fact": "OtherCo is located in Paris",
        "source_node_uuid": "other-company",
        "target_node_uuid": "paris",
        "attributes": {},
    }

    reader = object.__new__(ZepEntityReader)
    reader.client = SimpleNamespace(
        graph=SimpleNamespace(
            node=SimpleNamespace(
                get=lambda **_kwargs: SimpleNamespace(
                    uuid_="acme",
                    name="Acme",
                    labels=["Company"],
                    summary="",
                    attributes={},
                ),
                # Real Cloud 3.25 omits incoming edges here.
                get_edges=lambda **_kwargs: [SimpleNamespace(**outgoing)],
            )
        )
    )
    reader.get_all_edges = lambda _graph_id: [incoming, outgoing, unrelated]
    reader.get_all_nodes = lambda _graph_id: [
        {"uuid": "alice", "name": "Alice", "labels": ["Person"], "summary": ""},
        {"uuid": "acme", "name": "Acme", "labels": ["Company"], "summary": ""},
        {"uuid": "product", "name": "Product", "labels": ["Product"], "summary": ""},
    ]

    entity = reader.get_entity_with_context("graph-id", "acme")

    assert entity is not None
    assert len(entity.related_edges) == 2
    assert {edge["edge_name"] for edge in entity.related_edges} == {
        "WORKS_AT",
        "BUILDS",
    }
    assert {edge["direction"] for edge in entity.related_edges} == {
        "incoming",
        "outgoing",
    }
    assert {node["name"] for node in entity.related_nodes} == {"Alice", "Product"}


def test_entity_reader_does_not_turn_auth_failure_into_missing_entity():
    def unauthorized(**_kwargs):
        raise ZepApiError(status_code=401, body={"message": "unauthorized"})

    reader = object.__new__(ZepEntityReader)
    reader.client = SimpleNamespace(
        graph=SimpleNamespace(node=SimpleNamespace(get=unauthorized))
    )

    with pytest.raises(ZepApiError) as error:
        reader.get_entity_with_context("graph-id", "node-id")

    assert error.value.status_code == 401


def test_entity_reader_does_not_turn_edge_failure_into_empty_data():
    def forbidden(**_kwargs):
        raise ZepApiError(status_code=403, body={"message": "forbidden"})

    reader = object.__new__(ZepEntityReader)
    reader.client = SimpleNamespace(
        graph=SimpleNamespace(
            node=SimpleNamespace(get_edges=forbidden),
        )
    )

    with pytest.raises(ZepApiError) as error:
        reader.get_node_edges("node-id")

    assert error.value.status_code == 403


def test_report_tools_do_not_turn_zep_read_failures_into_empty_data():
    def unauthorized(**_kwargs):
        raise ZepApiError(status_code=401, body={"message": "unauthorized"})

    service = object.__new__(ZepToolsService)
    service.client = SimpleNamespace(
        graph=SimpleNamespace(node=SimpleNamespace(get=unauthorized))
    )

    with pytest.raises(ZepApiError):
        service.get_node_detail("node-id")

    service.get_all_edges = lambda _graph_id: (_ for _ in ()).throw(
        ZepApiError(status_code=503, body={"message": "unavailable"})
    )
    with pytest.raises(ZepApiError):
        service.get_node_edges("graph-id", "node-id")


def test_episode_processing_timeout_fails_instead_of_reporting_success(monkeypatch):
    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(
        graph=SimpleNamespace(
            episode=SimpleNamespace(
                get=lambda **_kwargs: SimpleNamespace(processed=False)
            )
        )
    )

    timestamps = iter([0.0, 2.0])
    monkeypatch.setattr(graph_builder_module.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="episode"):
        builder._wait_for_episodes(["episode-1"], timeout=1)


def _local_builder() -> GraphBuilderService:
    builder = object.__new__(GraphBuilderService)
    builder.api_key = "test-key"
    return builder


def test_document_ingestion_uses_openzep_graph_batch_and_persists_identity(monkeypatch):
    """OpenZep本地模式：chunk直接POST到/graph-batch并跟踪episode uuid。

    Zep Cloud的client.batch.*不被本地OpenZep服务实现；该测试锁定
    /graph-batch直传路径的关键契约（显式local模式、鉴权头、
    确定性operation id、回调、episode uuid回收与数量校验）。
    """
    _local_mode(monkeypatch)
    requests_seen = []

    def fake_urlopen(req, timeout=None):
        requests_seen.append(req)
        episodes = [
            {"uuid_": f"episode-{index}", "name": f"chunk-{index}"}
            for index in range(len(json.loads(req.data)["episodes"]))
        ]
        return _FakeResponse({"episodes": episodes})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    builder = _local_builder()
    persisted = []

    submission = builder.add_text_batches(
        "graph-id",
        ["chunk one", "chunk two"],
        batch_created_callback=lambda batch_id, operation_id: persisted.append(
            (batch_id, operation_id)
        ),
    )

    assert len(requests_seen) == 1
    request = requests_seen[0]
    assert request.full_url == LOCAL_BASE_URL + "/graph-batch"
    payload = json.loads(request.data)
    assert payload["graph_id"] == "graph-id"
    assert [episode["data"] for episode in payload["episodes"]] == [
        "chunk one",
        "chunk two",
    ]
    assert all(episode["type"] == "text" for episode in payload["episodes"])

    assert submission.batch_id == f"openzep-{submission.operation_id}"
    assert submission.item_count == 2
    assert submission.episode_uuids == ["episode-0", "episode-1"]
    assert len(submission.operation_id) == 64
    # 新路径在POST前用确定性ID一次性journal，便于事后对账
    assert persisted == [
        (f"openzep-{submission.operation_id}", submission.operation_id)
    ]


def test_local_graph_batch_post_sends_bearer_auth(monkeypatch):
    """M1：直传路径必须携带与SDK路径一致的Bearer鉴权头。"""

    _local_mode(monkeypatch)
    requests_seen = []

    def fake_urlopen(req, timeout=None):
        requests_seen.append(req)
        return _FakeResponse({"episodes": [{"uuid_": "episode-0"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    builder = _local_builder()
    builder.add_text_batches("graph-id", ["chunk one"])

    request = requests_seen[0]
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["Content-type"] == "application/json"


def test_local_graph_batch_post_requires_an_api_key(monkeypatch):
    """M1：直传路径缺少API key时必须显式失败，而不是静默无鉴权。"""

    _local_mode(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: None)
    monkeypatch.setattr(graph_builder_module.Config, "ZEP_API_KEY", None)

    builder = object.__new__(GraphBuilderService)

    with pytest.raises(ValueError, match="ZEP_API_KEY"):
        builder.add_text_batches("graph-id", ["chunk one"])


@pytest.mark.parametrize(
    "undelivered_reason",
    [ConnectionRefusedError(), socket.gaierror("name or service not known")],
)
def test_local_graph_batch_post_retries_when_request_never_delivered(
    monkeypatch, undelivered_reason
):
    """M1：请求确定未被服务端接收时，才允许有界重放。"""

    _local_mode(monkeypatch)
    calls = []
    sleeps = []

    def fake_urlopen(req, timeout=None):
        calls.append(True)
        if len(calls) == 1:
            raise urllib.error.URLError(undelivered_reason)
        return _FakeResponse({"episodes": [{"uuid_": "episode-0"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", sleeps.append)

    builder = _local_builder()
    submission = builder.add_text_batches("graph-id", ["chunk one"])

    assert submission.episode_uuids == ["episode-0"]
    assert len(calls) == 2
    assert sleeps == [graph_builder_module.ZEP_GRAPH_BATCH_INITIAL_DELAY_SECONDS]


def test_local_graph_batch_post_retries_rate_limit_then_succeeds(monkeypatch):
    """M1：429表示服务端明确拒绝未处理，属于可安全重放的失败。"""

    _local_mode(monkeypatch)
    calls = []
    sleeps = []

    def fake_urlopen(req, timeout=None):
        calls.append(True)
        if len(calls) == 1:
            raise _http_error(429)
        return _FakeResponse({"episodes": [{"uuid_": "episode-0"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", sleeps.append)

    builder = _local_builder()
    submission = builder.add_text_batches("graph-id", ["chunk one"])

    assert submission.episode_uuids == ["episode-0"]
    assert len(calls) == 2
    assert sleeps == [2.0]


@pytest.mark.parametrize("ambiguous_error", [TimeoutError("read timed out"), _http_error(500), _http_error(503)])
def test_local_graph_batch_post_fails_fast_on_ambiguous_errors(
    monkeypatch, ambiguous_error
):
    """M1：读超时/5xx等歧义失败不得重放，必须立即失败防止重复episode。"""

    _local_mode(monkeypatch)
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(True)
        raise ambiguous_error

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _s: None)

    builder = _local_builder()

    with pytest.raises(RuntimeError, match="NOT replayed"):
        builder.add_text_batches("graph-id", ["chunk one"])

    assert len(calls) == 1


def test_local_graph_batch_post_retry_is_bounded(monkeypatch):
    """M1：即便都是可安全重放的失败，重试也必须有界。"""

    _local_mode(monkeypatch)
    calls = []
    sleeps = []

    def fake_urlopen(req, timeout=None):
        calls.append(True)
        raise urllib.error.URLError(ConnectionRefusedError())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", sleeps.append)

    builder = _local_builder()

    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        builder.add_text_batches("graph-id", ["chunk one"])

    assert len(calls) == graph_builder_module.ZEP_GRAPH_BATCH_MAX_ATTEMPTS
    assert sleeps == [2.0, 4.0]


def test_document_ingestion_rejects_missing_openzep_episode_uuids(monkeypatch):
    """OpenZep返回的episode uuid数量不足时必须显式失败，不能静默成功。"""

    _local_mode(monkeypatch)

    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"episodes": [{"uuid_": "episode-0"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    builder = _local_builder()

    with pytest.raises(RuntimeError, match="episode uuids"):
        builder.add_text_batches("graph-id", ["chunk one", "chunk two"])


def test_document_ingestion_uses_current_batch_api_and_persists_identity(monkeypatch):
    """B2：Cloud模式保留Batch API路径（client.batch.create/add/process）。"""

    _cloud_mode(monkeypatch)
    calls = []

    class BatchApi:
        def create(self, **kwargs):
            calls.append(("create", kwargs))
            return SimpleNamespace(batch_id="batch-1")

        def add(self, **kwargs):
            calls.append(("add", kwargs))
            return [
                SimpleNamespace(episode_uuid=f"episode-{index}")
                for index, _item in enumerate(kwargs["items"])
            ]

        def process(self, **kwargs):
            calls.append(("process", kwargs))
            return SimpleNamespace(status="queued")

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(batch=BatchApi())
    persisted = []

    submission = builder.add_text_batches(
        "graph-id",
        ["chunk one", "chunk two"],
        batch_created_callback=lambda batch_id, operation_id: persisted.append(
            (batch_id, operation_id)
        ),
    )

    assert submission.batch_id == "batch-1"
    assert submission.item_count == 2
    assert len(submission.operation_id) == 64
    assert persisted == [
        (None, submission.operation_id),
        ("batch-1", submission.operation_id),
    ]
    assert [name for name, _kwargs in calls] == ["create", "add", "process"]
    items = calls[1][1]["items"]
    assert [item.type for item in items] == ["graph_episode", "graph_episode"]
    assert all(item.graph_id == "graph-id" for item in items)
    assert all(item.data_type == "text" for item in items)


def test_local_mode_ingestion_never_touches_the_batch_api(monkeypatch):
    """B1/B2：显式local模式下，client.batch绝不能被调用。"""

    _local_mode(monkeypatch)

    class ForbiddenBatchApi:
        def __getattr__(self, name):
            raise AssertionError(f"local mode must not call batch.{name}")

    builder = object.__new__(GraphBuilderService)
    builder.api_key = "test-key"
    builder.client = SimpleNamespace(batch=ForbiddenBatchApi())

    def fake_urlopen(req, timeout=None):
        episodes = [
            {"uuid_": f"episode-{index}"}
            for index in range(len(json.loads(req.data)["episodes"]))
        ]
        return _FakeResponse({"episodes": episodes})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    submission = builder.add_text_batches("graph-id", ["chunk one"])

    assert submission.batch_id.startswith("openzep-")


def test_batch_create_timeout_is_reconciled_by_operation_metadata(monkeypatch):
    """B2：Cloud batch create超时后按operation元数据对账，绝不盲目重放。"""

    _cloud_mode(monkeypatch)
    calls = []
    list_count = 0

    class BatchApi:
        def create(self, **_kwargs):
            calls.append("create")
            raise TimeoutError("response lost")

        def list(self, **_kwargs):
            nonlocal list_count
            calls.append("list")
            list_count += 1
            if list_count == 1:
                return SimpleNamespace(batches=[], next_cursor=None)
            return SimpleNamespace(
                batches=[SimpleNamespace(
                    batch_id="batch-recovered",
                    metadata={
                        "mirofish_operation_id": GraphBuilderService.build_operation_id(
                            "graph-id", ["chunk"]
                        ),
                        "graph_id": "graph-id",
                    },
                )],
                next_cursor=None,
            )

        def add(self, **kwargs):
            calls.append("add")
            return [SimpleNamespace(episode_uuid="episode-1")]

        def process(self, **_kwargs):
            calls.append("process")
            return SimpleNamespace(status="queued")

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(batch=BatchApi())
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    submission = builder.add_text_batches("graph-id", ["chunk"])

    assert submission.batch_id == "batch-recovered"
    assert calls == ["create", "list", "list", "add", "process"]


def test_batch_add_timeout_recovers_a_fully_accepted_group_without_replay(monkeypatch):
    """B2：Cloud batch add超时后，仅在服务端确认完整接收时恢复，不重放。"""

    _cloud_mode(monkeypatch)
    add_calls = []
    list_calls = []

    class BatchApi:
        def create(self, **_kwargs):
            return SimpleNamespace(batch_id="batch-1")

        def add(self, **_kwargs):
            add_calls.append(True)
            raise TimeoutError("response lost")

        def list_items(self, **_kwargs):
            list_calls.append(True)
            if len(list_calls) == 1:
                return SimpleNamespace(items=[], next_cursor=None)
            return SimpleNamespace(
                items=[
                    SimpleNamespace(sequence_index=0, episode_uuid="episode-1"),
                    SimpleNamespace(sequence_index=1, episode_uuid="episode-2"),
                ],
                next_cursor=None,
            )

        def process(self, **_kwargs):
            return SimpleNamespace(status="queued")

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(batch=BatchApi())
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    submission = builder.add_text_batches(
        "graph-id", ["chunk one", "chunk two"]
    )

    assert submission.item_count == 2
    assert add_calls == [True]
    assert len(list_calls) == 2


def test_graph_create_persists_identity_before_post_and_reconciles_timeout():
    events = []

    class GraphApi:
        def create(self, **kwargs):
            events.append(("create", kwargs["graph_id"]))
            raise TimeoutError("response lost")

        def get(self, graph_id):
            events.append(("get", graph_id))
            return SimpleNamespace(graph_id=graph_id)

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(graph=GraphApi())

    graph_id = builder.create_graph(
        "Graph",
        graph_id="known-id",
        graph_id_callback=lambda value: events.append(("persist", value)),
    )

    assert graph_id == "known-id"
    assert events == [
        ("persist", "known-id"),
        ("create", "known-id"),
        ("get", "known-id"),
    ]


def test_batch_wait_collects_processed_episodes_and_keeps_polling(monkeypatch):
    """OpenZep本地模式：_wait_for_batch轮询/graph/episodes/{uuid}直到processed。"""
    _local_mode(monkeypatch)
    poll_paths = []
    outcomes = {
        "episode-1": iter([False, True]),
        "episode-2": iter([True]),
    }

    def fake_urlopen(req, timeout=None):
        path = req.full_url.rsplit("/", 1)[-1]
        poll_paths.append(path)
        # episode-1首轮未完成，第二轮完成；episode-2首轮即完成
        done = next(outcomes[path])
        return _FakeResponse({"processed": done})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    builder = _local_builder()
    submission = BatchSubmission("batch-1", "operation", ["episode-1", "episode-2"], 2)

    # 返回顺序为完成顺序：episode-2首轮完成，episode-1第二轮完成
    assert builder._wait_for_batch(submission, timeout=30) == ["episode-2", "episode-1"]
    # pending是set，轮询顺序不定，只断言次数
    assert poll_paths.count("episode-1") == 2
    assert poll_paths.count("episode-2") == 1
    assert len(poll_paths) == 3


def test_batch_wait_times_out_while_episodes_stay_unprocessed(monkeypatch):
    _local_mode(monkeypatch)

    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"processed": False})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    timestamps = iter([0.0, 2.0])
    monkeypatch.setattr(graph_builder_module.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    builder = _local_builder()

    with pytest.raises(TimeoutError, match="batch-1"):
        builder._wait_for_batch(
            BatchSubmission("batch-1", "operation", ["episode-1"], 1),
            timeout=1,
        )


@pytest.mark.parametrize("status", [404, 410])
def test_episode_poll_permanent_404_410_fails_fast(monkeypatch, status):
    """M2：episode轮询遇到永久性404/410必须立即失败，不得空转到超时。"""
    _local_mode(monkeypatch)
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        raise _http_error(status)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    builder = _local_builder()

    with pytest.raises(PermanentEpisodePollError, match="episode-gone") as error:
        builder._wait_for_batch(
            BatchSubmission("batch-1", "operation", ["episode-gone"], 1),
            timeout=7200,
        )

    assert str(status) in str(error.value)
    assert calls == [LOCAL_BASE_URL + "/graph/episodes/episode-gone"]


def test_episode_poll_swallows_transient_errors_and_keeps_polling(monkeypatch):
    """M2：瞬时失败（连接拒绝、5xx、408/429）仍保持轮询，不被误判为永久。"""
    _local_mode(monkeypatch)
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(True)
        if len(calls) == 1:
            raise urllib.error.URLError(ConnectionRefusedError())
        return _FakeResponse({"processed": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    builder = _local_builder()

    done = builder._wait_for_batch(
        BatchSubmission("batch-1", "operation", ["episode-1"], 1),
        timeout=30,
    )

    assert done == ["episode-1"]
    assert len(calls) == 2


def test_episode_poll_missing_api_key_fails_fast(monkeypatch):
    """M1/M2：轮询前缺少API key必须立即失败，不得被瞬时错误处理吞掉后空转。"""
    _local_mode(monkeypatch)
    monkeypatch.setattr(graph_builder_module.Config, "ZEP_API_KEY", None)
    monkeypatch.setattr(
        graph_builder_module.time,
        "sleep",
        lambda _seconds: pytest.fail("poll loop must not start"),
    )

    builder = object.__new__(GraphBuilderService)

    with pytest.raises(ValueError, match="ZEP_API_KEY"):
        builder._wait_for_batch(
            BatchSubmission("batch-1", "operation", ["episode-1"], 1),
            timeout=7200,
        )


def test_local_wire_contract_for_ingestion_and_wait(monkeypatch):
    """锁定OpenZep本地模式的线上契约：/graph-batch直传 + episode轮询。"""
    _local_mode(monkeypatch)
    requests_seen = []

    def fake_urlopen(req, timeout=None):
        method = req.get_method()
        path = req.full_url.replace(LOCAL_BASE_URL, "")
        requests_seen.append((method, path, req.data))

        if method == "POST" and path == "/graph-batch":
            return _FakeResponse({"episodes": [{"uuid_": "episode-1"}]})
        if method == "GET" and path == "/graph/episodes/episode-1":
            return _FakeResponse({"processed": True})
        raise AssertionError(f"Unexpected request: {method} {path}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    builder = _local_builder()

    submission = builder.add_text_batches("graph-id", ["source chunk"])
    assert builder._wait_for_batch(submission, timeout=1) == ["episode-1"]

    assert [(method, path) for method, path, _body in requests_seen] == [
        ("POST", "/graph-batch"),
        ("GET", "/graph/episodes/episode-1"),
    ]
    batch_payload = json.loads(requests_seen[0][2])
    assert batch_payload == {
        "graph_id": "graph-id",
        "episodes": [
            {
                "name": f"{submission.operation_id}-0",
                "data": "source chunk",
                "type": "text",
                "source_description": "MiroFish source document chunk",
            }
        ],
    }


def test_batch_wait_validates_terminal_items_and_opaque_zero_cursor(monkeypatch):
    """B2：Cloud模式等待batch终态并逐项校验（含不透明0游标）。"""

    _cloud_mode(monkeypatch)
    list_calls = []

    class BatchApi:
        def get(self, **_kwargs):
            return SimpleNamespace(
                status="succeeded",
                progress=SimpleNamespace(
                    percent_complete=100,
                    succeeded_items=2,
                ),
            )

        def list_items(self, **kwargs):
            list_calls.append(kwargs)
            if kwargs["cursor"] is None:
                return SimpleNamespace(
                    items=[SimpleNamespace(
                        sequence_index=0,
                        status="succeeded",
                        episode_uuid="episode-1",
                        source_uuid="episode-1",
                    )],
                    next_cursor=0,
                )
            return SimpleNamespace(
                items=[SimpleNamespace(
                    sequence_index=1,
                    status="succeeded",
                    episode_uuid="episode-2",
                    source_uuid="episode-2",
                )],
                next_cursor=None,
            )

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(batch=BatchApi())
    submission = BatchSubmission("batch-1", "operation", [], 2)

    assert builder._wait_for_batch(submission, timeout=1) == [
        "episode-1",
        "episode-2",
    ]
    assert [call["cursor"] for call in list_calls] == [None, 0]


@pytest.mark.parametrize("status", ["partial", "failed", "invalid", "canceled"])
def test_batch_non_success_terminal_states_fail(monkeypatch, status):
    """B2：Cloud batch非成功终态必须显式失败。"""

    _cloud_mode(monkeypatch)
    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(
        batch=SimpleNamespace(
            get=lambda **_kwargs: SimpleNamespace(status=status, progress=None),
            list_items=lambda **_kwargs: SimpleNamespace(
                items=[SimpleNamespace(status="failed", error={"message": "bad"})],
                next_cursor=None,
            ),
        )
    )

    with pytest.raises(RuntimeError, match=status):
        builder._wait_for_batch(
            BatchSubmission("batch-1", "operation", [], 1),
            timeout=1,
        )


def test_batch_wait_times_out_while_status_remains_nonterminal(monkeypatch):
    """B2：Cloud batch长时间非终态时按超时失败。"""

    _cloud_mode(monkeypatch)
    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(
        batch=SimpleNamespace(
            get=lambda **_kwargs: SimpleNamespace(status="processing", progress=None)
        )
    )
    timestamps = iter([0.0, 2.0])
    monkeypatch.setattr(graph_builder_module.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="batch-1"):
        builder._wait_for_batch(
            BatchSubmission("batch-1", "operation", [], 1),
            timeout=1,
        )


def test_installed_sdk_serializes_the_batch_325_contract(monkeypatch):
    """B2：真实SDK的Batch API 3.25线上契约（Cloud模式，模拟传输层）。"""

    _cloud_mode(monkeypatch)
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path, request.content))
        path = request.url.path
        if path.endswith("/batches") and request.method == "POST":
            return httpx.Response(
                200,
                json={"batch_id": "batch-1", "status": "draft", "item_count": 0},
            )
        if path.endswith("/batches/batch-1/items") and request.method == "POST":
            return httpx.Response(200, json=[{
                "item_id": "item-1",
                "sequence_index": 0,
                "status": "pending",
                "episode_uuid": "episode-1",
                "source_uuid": "episode-1",
            }])
        if path.endswith("/batches/batch-1/process"):
            return httpx.Response(
                200,
                json={"batch_id": "batch-1", "status": "queued", "item_count": 1},
            )
        if path.endswith("/batches/batch-1"):
            return httpx.Response(200, json={
                "batch_id": "batch-1",
                "status": "succeeded",
                "item_count": 1,
                "progress": {"percent_complete": 100, "succeeded_items": 1},
            })
        if path.endswith("/batches/batch-1/items") and request.method == "GET":
            return httpx.Response(200, json={
                "items": [{
                    "item_id": "item-1",
                    "sequence_index": 0,
                    "status": "succeeded",
                    "episode_uuid": "episode-1",
                    "source_uuid": "episode-1",
                }],
                "next_cursor": None,
            })
        raise AssertionError(f"Unexpected request: {request.method} {path}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport_client:
        builder = object.__new__(GraphBuilderService)
        builder.client = Zep(api_key="test-key", httpx_client=transport_client)
        submission = builder.add_text_batches("graph-id", ["source chunk"])
        assert builder._wait_for_batch(submission, timeout=1) == ["episode-1"]

    assert [(method, path) for method, path, _body in requests] == [
        ("POST", "/api/v2/batches"),
        ("POST", "/api/v2/batches/batch-1/items"),
        ("POST", "/api/v2/batches/batch-1/process"),
        ("GET", "/api/v2/batches/batch-1"),
        ("GET", "/api/v2/batches/batch-1/items"),
    ]
    add_payload = json.loads(requests[1][2])
    assert add_payload["items"][0] == {
        "data": "source chunk",
        "data_type": "text",
        "graph_id": "graph-id",
        "metadata": add_payload["items"][0]["metadata"],
        "source_description": "MiroFish source document chunk",
        "type": "graph_episode",
    }
