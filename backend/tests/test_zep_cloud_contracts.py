from types import SimpleNamespace
import json

import pytest
from zep_cloud.core.api_error import ApiError as ZepApiError

from app.services import graph_builder as graph_builder_module
from app.services.graph_builder import BatchSubmission, GraphBuilderService
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.zep_entity_reader import EntityNode, ZepEntityReader
from app.services.zep_tools import ZepToolsService


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


def test_document_ingestion_uses_openzep_graph_batch_and_persists_identity(monkeypatch):
    """OpenZep本地试验路径：chunk直接POST到/graph-batch并跟踪episode uuid。

    Zep Cloud的client.batch.*不被本地OpenZep服务实现；该测试锁定
    /graph-batch直传路径的关键契约（确定性operation id、回调、
    episode uuid回收与数量校验）。
    """
    requests_seen = []

    class _FakeResponse:
        def __init__(self, body):
            self._body = body

        def read(self):
            return json.dumps(self._body).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(req, timeout=None):
        requests_seen.append(req)
        episodes = [
            {"uuid_": f"episode-{index}", "name": f"chunk-{index}"}
            for index in range(len(json.loads(req.data)["episodes"]))
        ]
        return _FakeResponse({"episodes": episodes})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(batch=SimpleNamespace())
    persisted = []

    submission = builder.add_text_batches(
        "graph-id",
        ["chunk one", "chunk two"],
        batch_created_callback=lambda batch_id, operation_id: persisted.append(
            (batch_id, operation_id)
        ),
    )

    from app.utils.zep import ZEP_CLOUD_BASE_URL

    assert len(requests_seen) == 1
    request = requests_seen[0]
    assert request.full_url == ZEP_CLOUD_BASE_URL + "/graph-batch"
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


def test_document_ingestion_rejects_missing_openzep_episode_uuids(monkeypatch):
    """OpenZep返回的episode uuid数量不足时必须显式失败，不能静默成功。"""

    class _FakeResponse:
        def __init__(self, body):
            self._body = body

        def read(self):
            return json.dumps(self._body).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"episodes": [{"uuid_": "episode-0"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(batch=SimpleNamespace())

    with pytest.raises(RuntimeError, match="episode uuids"):
        builder.add_text_batches("graph-id", ["chunk one", "chunk two"])


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
    """OpenZep本地路径：_wait_for_batch轮询/graph/episodes/{uuid}直到processed。"""
    poll_paths = []
    outcomes = {
        "episode-1": iter([False, True]),
        "episode-2": iter([True]),
    }

    class _FakeResponse:
        def __init__(self, body):
            self._body = json.dumps(body).encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(req, timeout=None):
        path = req.full_url.rsplit("/", 1)[-1]
        poll_paths.append(path)
        # episode-1首轮未完成，第二轮完成；episode-2首轮即完成
        done = next(outcomes[path])
        return _FakeResponse({"processed": done})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(batch=SimpleNamespace())
    submission = BatchSubmission("batch-1", "operation", ["episode-1", "episode-2"], 2)

    # 返回顺序为完成顺序：episode-2首轮完成，episode-1第二轮完成
    assert builder._wait_for_batch(submission, timeout=30) == ["episode-2", "episode-1"]
    # pending是set，轮询顺序不定，只断言次数
    assert poll_paths.count("episode-1") == 2
    assert poll_paths.count("episode-2") == 1
    assert len(poll_paths) == 3


def test_batch_wait_times_out_while_episodes_stay_unprocessed(monkeypatch):
    class _FakeResponse:
        def __init__(self, body):
            self._body = json.dumps(body).encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"processed": False})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    timestamps = iter([0.0, 2.0])
    monkeypatch.setattr(graph_builder_module.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(batch=SimpleNamespace())

    with pytest.raises(TimeoutError, match="batch-1"):
        builder._wait_for_batch(
            BatchSubmission("batch-1", "operation", ["episode-1"], 1),
            timeout=1,
        )


def test_openzep_trial_wire_contract_for_ingestion_and_wait(monkeypatch):
    """锁定OpenZep本地试验路径的线上契约：/graph-batch直传 + episode轮询。"""
    from app.utils.zep import ZEP_CLOUD_BASE_URL

    requests_seen = []

    class _FakeResponse:
        def __init__(self, body):
            self._body = json.dumps(body).encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_urlopen(req, timeout=None):
        method = req.get_method()
        path = req.full_url.replace(ZEP_CLOUD_BASE_URL, "")
        requests_seen.append((method, path, req.data))

        if method == "POST" and path == "/graph-batch":
            return _FakeResponse({"episodes": [{"uuid_": "episode-1"}]})
        if method == "GET" and path == "/graph/episodes/episode-1":
            return _FakeResponse({"processed": True})
        raise AssertionError(f"Unexpected request: {method} {path}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(graph_builder_module.time, "sleep", lambda _seconds: None)

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(batch=SimpleNamespace())

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
