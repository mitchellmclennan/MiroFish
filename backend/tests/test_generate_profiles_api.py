"""generate-profiles API contract tests (review finding L3).

The route must fail clearly (400 + entity_quality report) when the quality
filter drops every entity — never a silent `success: true, count: 0` — and
must report the post-filter entity types that actually produced profiles.
The route also applies the prepare-level duplicate-speaker guard and must
surface its merge audit (entity_quality.merged) in the response.
"""

import pytest

from app import create_app
from app.api import simulation as simulation_api_module
from app.services.zep_entity_reader import EntityNode, FilteredEntities


def _entity(name, labels, summary=""):
    return EntityNode(
        uuid=f"uuid-{name}",
        name=name,
        labels=labels,
        summary=summary,
        attributes={},
        related_edges=[],
        related_nodes=[],
    )


def _stub_reader(entities):
    class StubReader:
        def filter_defined_entities(self, **kwargs):
            return FilteredEntities(
                entities=list(entities),
                entity_types={e.get_entity_type() or "Unknown" for e in entities},
                total_count=len(entities),
                filtered_count=len(entities),
            )

    return StubReader


def _stub_generator():
    class StubProfile:
        def to_reddit_format(self):
            return {"user_id": 0, "username": "u", "name": "Marcus R."}

        def to_twitter_format(self):
            return dict(self.to_reddit_format())

        def to_dict(self):
            return dict(self.to_reddit_format())

    class StubGenerator:
        def __init__(self, *args, **kwargs):
            pass

        def generate_profiles_from_entities(self, entities, use_llm=True, **kwargs):
            self.entities = list(entities)
            return [StubProfile() for _ in entities]

    return StubGenerator


@pytest.fixture()
def client(monkeypatch):
    app = create_app()
    app.config.update(TESTING=True)
    monkeypatch.delenv("MIROFISH_ENTITY_QUALITY_FILTER", raising=False)
    return app.test_client()


def test_generate_profiles_fails_when_quality_filter_drops_all(client, monkeypatch):
    monkeypatch.setattr(
        simulation_api_module,
        "ZepEntityReader",
        _stub_reader(
            [
                _entity("Anti-Kickback Statute", ["ExtractedEntity", "Entity"], "legal"),
                _entity("intake_2026-06-30.pdf", ["ExtractedEntity", "Entity"], "doc"),
            ]
        ),
    )

    response = client.post("/api/simulation/generate-profiles", json={"graph_id": "g"})

    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False
    assert "entity_quality" in body["data"]
    quality = body["data"]["entity_quality"]
    assert quality["kept_count"] == 0
    assert quality["dropped_count"] == 2
    assert quality["relabeled_count"] == 0
    reasons = {d["reason"] for d in quality["dropped"]}
    assert reasons == {"statute", "filename_fragment"}


def test_generate_profiles_reports_post_filter_entity_types(client, monkeypatch):
    """entity_types必须是过滤后实际生成人设的类型，而不是pre-filter集合。"""
    StubGenerator = _stub_generator()
    monkeypatch.setattr(
        simulation_api_module,
        "ZepEntityReader",
        _stub_reader(
            [
                _entity("Marcus R.", ["Person", "Entity"], "An order was drafted."),
                _entity("EKRA", ["ExtractedEntity", "Entity"], "legal"),
            ]
        ),
    )
    monkeypatch.setattr(simulation_api_module, "OasisProfileGenerator", StubGenerator)

    response = client.post("/api/simulation/generate-profiles", json={"graph_id": "g"})

    assert response.status_code == 200
    data = response.get_json()["data"]
    # 只有post-filter的Person进入了人设生成
    assert data["entity_types"] == ["Person"]
    assert data["count"] == 1
    assert data["profiles"][0]["name"] == "Marcus R."
    quality = data["entity_quality"]
    assert quality["kept_count"] == 1
    assert quality["dropped_count"] == 1
    assert quality["dropped"][0]["entity_name"] == "EKRA"
    assert quality["dropped"][0]["reason"] == "statute"


def test_generate_profiles_still_fails_on_zero_pre_filter_entities(client, monkeypatch):
    monkeypatch.setattr(
        simulation_api_module,
        "ZepEntityReader",
        _stub_reader([]),
    )

    response = client.post("/api/simulation/generate-profiles", json={"graph_id": "g"})

    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False


def test_generate_profiles_dedupes_duplicate_speakers_and_audits_merges(
    client, monkeypatch
):
    """重复发言主体守卫（prepare级）：同一归一化身份（"NeoLife"与
    "NeoLife Official"）只生成一个人设，合并在entity_quality.merged审计。"""
    StubGenerator = _stub_generator()
    monkeypatch.setattr(
        simulation_api_module,
        "ZepEntityReader",
        _stub_reader(
            [
                _entity(
                    "NeoLife",
                    ["Organization", "Entity"],
                    "Fulfillment infrastructure.",
                ),
                _entity(
                    "NeoLife Official",
                    ["Organization", "Entity"],
                    "Official account.",
                ),
            ]
        ),
    )
    monkeypatch.setattr(simulation_api_module, "OasisProfileGenerator", StubGenerator)

    response = client.post(
        "/api/simulation/generate-profiles", json={"graph_id": "g"}
    )

    assert response.status_code == 200
    data = response.get_json()["data"]
    # 同一身份只保留一个发言人：只有一个人设
    assert data["count"] == 1
    quality = data["entity_quality"]
    assert quality["kept_count"] == 1
    assert quality["merged_count"] == 1
    merged = quality["merged"][0]
    assert merged["entity_name"] == "NeoLife Official"
    assert merged["kept_name"] == "NeoLife"
    assert merged["action"] == "merge"
    assert merged["reason"] == "duplicate_speaker_identity"
