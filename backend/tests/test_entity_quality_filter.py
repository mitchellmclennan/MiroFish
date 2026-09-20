"""Junk-entity filtering tests (applied before profile generation).

Statutes, products/companies mislabeled as people, boilerplate fragments,
empty names and low-information entities must not become personas.
Legitimate companies stay as organizations (relabelled when they were
mislabeled as people).
"""

import pytest

from app.services.entity_quality_filter import (
    EntityQualityFilter,
    filter_entities_for_profiles,
    is_entity_quality_filter_enabled,
)
from app.services.zep_entity_reader import EntityNode


def _entity(name, labels, summary="", attributes=None, related_edges=None, uuid=None):
    return EntityNode(
        uuid=uuid or f"uuid-{name}",
        name=name,
        labels=labels,
        summary=summary,
        attributes=attributes or {},
        related_edges=related_edges or [],
        related_nodes=[],
    )


def _decisions_by_name(report):
    return {d.entity_name: d for d in report.dropped + report.relabeled}


class TestStatuteFiltering:
    def test_statute_label_is_dropped(self):
        entity = _entity("Anti-Kickback Statute", ["Statute", "Entity"], summary="Regulates fees.")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept == []
        assert report.dropped[0].reason == "statute"

    def test_statute_named_entity_is_dropped_regardless_of_label(self):
        entity = _entity(
            "Dietary Supplement Health and Education Act of 1994",
            ["ExtractedEntity", "Entity"],
            summary="A law about supplements.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.dropped[0].reason == "statute"

    def test_real_world_named_acts_are_dropped(self):
        names = ["Anti-Kickback Statute", "EKRA", "AKS/EKRA", "HIPAA", "21 U.S.C. § 1798"]
        entities = [_entity(n, ["ExtractedEntity", "Entity"], summary="legal") for n in names]
        report = EntityQualityFilter().filter_entities(entities)
        assert len(report.dropped) == len(names)
        assert all(d.reason == "statute" for d in report.dropped)

    def test_summary_mentioning_statutes_does_not_drop_company(self):
        entity = _entity(
            "neolife",
            ["Organization", "Entity"],
            summary="Fees comply with AKS and EKRA requirements.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert entity.name in [e.name for e in report.kept]


class TestCompanyRelabeling:
    def test_company_mislabeled_as_person_is_relabelled(self):
        entity = _entity(
            "Neolife Inc",
            ["Person", "Entity"],
            summary="A company that ships orders.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert len(report.kept) == 1
        kept = report.kept[0]
        assert kept.get_entity_type() == "Organization"
        assert report.relabeled[0].reason == "company_mislabeled_as_person"
        assert report.relabeled[0].relabeled_to == "Organization"
        # Relabelling must not lose graph facts
        entity_with_edges = _entity(
            "Acme Corp",
            ["Person", "Entity"],
            summary="",
            related_edges=[{"fact": "Acme Corp partners with neolife.", "direction": "incoming"}],
        )
        report2 = EntityQualityFilter().filter_entities([entity_with_edges])
        assert report2.kept[0].get_entity_type() == "Organization"
        assert report2.kept[0].related_edges == entity_with_edges.related_edges

    def test_legitimate_company_stays_as_organization(self):
        entity = _entity(
            "NVIDIA Corporation",
            ["Organization", "Entity"],
            summary="NVIDIA Corporation owns trademarks.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].get_entity_type() == "Organization"
        assert not report.relabeled


class TestProductFiltering:
    def test_product_mislabeled_as_person_is_dropped(self):
        entity = _entity(
            "Protein Shake",
            ["Person", "Entity"],
            summary="Protein Shake is a supplement sold to clinics.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept == []
        assert report.dropped[0].reason == "product_mislabeled_as_person"

    def test_dosage_form_names_are_dropped_for_generic_entities(self):
        names = [
            "tretinoin 0.05% cream",
            "Sexual health Troches",
            "Compounded finasteride + minoxidil",
            "ODT",
        ]
        entities = [
            _entity(n, ["ExtractedEntity", "Entity"], summary="product")
            for n in names
        ]
        report = EntityQualityFilter().filter_entities(entities)
        assert len(report.dropped) == len(names)

    def test_company_with_product_words_in_name_is_kept(self):
        """组织标签优先豁免：合法公司名含产品词不被误伤。"""
        entity = _entity(
            "Troches Manufacturing Company",
            ["Company", "Entity"],
            summary="A manufacturer.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "Troches Manufacturing Company"


class TestFragmentFiltering:
    @pytest.mark.parametrize(
        "name,reason",
        [
            ("", "empty_name"),
            ("   ", "empty_name"),
            ("N/A", "placeholder_name"),
            ("Unknown", "placeholder_name"),
            ("Dr.", "honorific_title_only"),
            ("CEO", "honorific_title_only"),
            ("intake_2026-06-30.pdf", "filename_fragment"),
            ("https://neolife.example.com", "url_or_email_fragment"),
            ("ops@neolife.example", "url_or_email_fragment"),
            ("§5", "no_alphabetic_characters"),
            ("300 orders/mo", "number_led_fragment"),
            ("per-clinic controls", "generic_lowercase_fragment"),
            ("provider approval", "generic_lowercase_fragment"),
        ],
    )
    def test_junk_names_are_dropped(self, name, reason):
        entity = _entity(name, ["ExtractedEntity", "Entity"], summary="some context")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.dropped[0].reason == reason

    def test_lowercase_generic_fragments_from_neolife_graph(self):
        names = [
            "roll-up billing",
            "white-label intake",
            "cold_chain_overnight",
            "dollar amount per order",
            "brick-and-mortar clinics",
            "custom E&O structuring",
        ]
        entities = [_entity(n, ["ExtractedEntity", "Entity"], summary="") for n in names]
        report = EntityQualityFilter().filter_entities(entities)
        assert len(report.dropped) == len(names)
        assert all(d.reason == "generic_lowercase_fragment" for d in report.dropped)

    def test_lowercase_organization_is_kept(self):
        """小写名称但类型明确（如neolife标为Organization）时保留。"""
        entity = _entity("neolife", ["Organization", "Entity"], summary="infrastructure")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "neolife"

    def test_lowercase_person_with_real_name_is_kept(self):
        entity = _entity("marcus r.", ["Person", "Entity"], summary="An order was drafted for marcus r.")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "marcus r."


class TestLowInformationFiltering:
    def test_entity_without_any_information_is_dropped(self):
        entity = _entity("Orphan Entity", ["Person", "Entity"], summary="", attributes={})
        report = EntityQualityFilter().filter_entities([entity])
        assert report.dropped[0].reason == "low_information_entity"

    def test_entity_with_edges_is_kept(self):
        entity = _entity(
            "Connected Entity",
            ["Person", "Entity"],
            summary="",
            related_edges=[{"fact": "Connected Entity ships orders.", "direction": "outgoing"}],
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept

    def test_entity_with_summary_is_kept(self):
        entity = _entity("Summarized", ["Person", "Entity"], summary="A person with a story.")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept

    def test_entity_with_attributes_only_is_kept(self):
        entity = _entity("Attr Entity", ["Person", "Entity"], summary="", attributes={"role": "coordinator"})
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept


class TestFilterIntegrationContract:
    def test_neolife_style_corpus_is_filtered_correctly(self):
        """以真实NeoLife图谱中的实体形态做回归验证。"""
        entities = [
            _entity("neolife", ["Organization", "Entity"], summary="neolife is the fulfillment infrastructure."),
            _entity("NVIDIA Corporation", ["Organization", "Entity"], summary="NVIDIA Corporation owns trademarks."),
            _entity("Marcus R.", ["Person", "Entity"], summary="An order was drafted for Marcus R."),
            _entity("Dana R.", ["Person", "Entity"], summary="A coordinator processes orders."),
            _entity("Dr.", ["Person", "Entity"], summary="Dr. approved orders using a streamlined interface."),
            _entity("Anti-Kickback Statute", ["ExtractedEntity", "Entity"], summary="A fee structure remains constant."),
            _entity("EKRA", ["ExtractedEntity", "Entity"], summary="Fees under EKRA are regulated."),
            _entity("tretinoin 0.05% cream", ["ExtractedEntity", "Entity"], summary="A refill order is being processed."),
            _entity("Sexual health Troches", ["ExtractedEntity", "Entity"], summary=""),
            _entity("intake_2026-06-30.pdf", ["ExtractedEntity", "Entity"], summary="The document contains 3 pages."),
            _entity("300 orders/mo", ["ExtractedEntity", "Entity"], summary="Clinics must upgrade when they clear 300 orders."),
            _entity("per-clinic controls", ["ExtractedEntity", "Entity"], summary="Per-clinic controls enable management."),
            _entity("carrier", ["ExtractedEntity", "Entity"], summary=""),
            _entity("Crunchbase", ["MediaOutlet", "Entity"], summary="Crunchbase follows neolife for updates."),
            _entity("partner negotiators", ["PartnerNegotiator", "Entity"], summary="Stress-test synthetic personas."),
        ]
        report = EntityQualityFilter().filter_entities(entities)

        kept_names = {e.name for e in report.kept}
        assert "neolife" in kept_names
        assert "NVIDIA Corporation" in kept_names
        assert "Marcus R." in kept_names
        assert "Dana R." in kept_names
        assert "Crunchbase" in kept_names
        assert "partner negotiators" in kept_names

        dropped_names = {d.entity_name for d in report.dropped}
        assert "Dr." in dropped_names
        assert "Anti-Kickback Statute" in dropped_names
        assert "EKRA" in dropped_names
        assert "tretinoin 0.05% cream" in dropped_names
        assert "Sexual health Troches" in dropped_names
        assert "intake_2026-06-30.pdf" in dropped_names
        assert "300 orders/mo" in dropped_names
        assert "per-clinic controls" in dropped_names
        assert "carrier" in dropped_names

    def test_report_dict_is_json_serializable(self):
        import json

        entities = [
            _entity("EKRA", ["ExtractedEntity", "Entity"], summary=""),
            _entity("neolife", ["Organization", "Entity"], summary="infrastructure"),
        ]
        report = EntityQualityFilter().filter_entities(entities)
        payload = json.dumps(report.to_dict(), ensure_ascii=False)
        assert "dropped" in payload and "kept" in payload

    def test_env_kill_switch_disables_filtering(self, monkeypatch):
        monkeypatch.setenv("MIROFISH_ENTITY_QUALITY_FILTER", "0")
        assert not is_entity_quality_filter_enabled()

        entities = [
            _entity("EKRA", ["ExtractedEntity", "Entity"], summary="legal"),
            _entity("", ["Person", "Entity"], summary=""),
        ]
        report = filter_entities_for_profiles(entities)
        assert len(report.kept) == 2
        assert report.dropped == []
        assert report.relabeled == []

    def test_filtering_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MIROFISH_ENTITY_QUALITY_FILTER", raising=False)
        assert is_entity_quality_filter_enabled()
