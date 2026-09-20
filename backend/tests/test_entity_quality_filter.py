"""Junk-entity filtering tests (applied before profile generation).

Statutes, products/companies mislabeled as people, boilerplate fragments,
empty names and low-information entities must not become personas.
Legitimate companies stay as organizations (relabelled when they were
mislabeled as people).

The prepare-level pipeline (``filter_entities_for_profiles``) additionally
dedupes duplicate speakers: one agent persona per normalized entity
identity ("NeoLife" + "NeoLife Official" extracted as two entities must
not become two speakers), while deliberately never merging unrelated
aliases.
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


class TestRealEntityPreservation:
    """评审H2回归：真实组织/人物不得被名称形状规则误删。"""

    def test_usc_organization_is_kept(self):
        """\bu.?s.?c\b曾把南加州大学缩写当法规引证删除；无点号缩写不再是引证。"""
        entity = _entity("USC", ["Organization", "Entity"], summary="The university.")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "USC"

    def test_dotted_statute_citations_are_still_dropped(self):
        """带点号的U.S.C./C.F.R.引证仍然必须删除。"""
        for name in ("U.S.C.", "C.F.R.", "21 U.S.C. § 1798"):
            entity = _entity(name, ["ExtractedEntity", "Entity"], summary="legal")
            report = EntityQualityFilter().filter_entities([entity])
            assert report.dropped and report.dropped[0].reason == "statute", name

    def test_short_and_number_led_org_names_are_kept(self):
        """3M/7-Eleven是真实机构名；过短/数字开头规则豁免组织标签。"""
        for name in ("3M", "7-Eleven"):
            entity = _entity(name, ["Organization", "Entity"], summary="A real company.")
            report = EntityQualityFilter().filter_entities([entity])
            assert report.kept and report.kept[0].name == name, name

    def test_number_led_generic_fragments_are_still_dropped(self):
        entity = _entity("300 orders/mo", ["ExtractedEntity", "Entity"], summary="")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.dropped[0].reason == "number_led_fragment"

    def test_lowercase_person_with_generic_label_is_kept(self):
        """评审H2：marcus r.被提取器打上默认ExtractedEntity标签，不得因小写被删。"""
        entity = _entity(
            "marcus r.",
            ["ExtractedEntity", "Entity"],
            summary="An order was drafted for marcus r.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "marcus r."

    def test_lowercase_person_shapes_with_multiple_initials_are_kept(self):
        entity = _entity(
            "dana r. j.",
            ["ExtractedEntity", "Entity"],
            summary="A coordinator.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "dana r. j."

    def test_lowercase_concept_fragments_are_still_dropped(self):
        """人名形状豁免不得放开概念碎片：无缩写的小写片段仍应被删。"""
        names = ["provider approval", "roll-up billing", "carrier", "olife"]
        entities = [_entity(n, ["ExtractedEntity", "Entity"], summary="") for n in names]
        report = EntityQualityFilter().filter_entities(entities)
        assert {d.entity_name for d in report.dropped} == set(names)
        assert all(d.reason == "generic_lowercase_fragment" for d in report.dropped)


class TestPricingTierFiltering:
    """评审H2回归：定价套餐档位不得成为社媒发言主体。"""

    def test_bare_tier_names_are_dropped_for_generic_labels(self):
        entities = [
            _entity(n, ["ExtractedEntity", "Entity"], summary="A plan tier.")
            for n in ("Starter", "Growth", "Scale")
        ]
        report = EntityQualityFilter().filter_entities(entities)
        assert len(report.dropped) == 3
        assert all(d.reason == "pricing_tier_not_a_speaker" for d in report.dropped)

    def test_tier_pricing_summaries_are_dropped(self):
        """真实trial图谱形态："Growth plan costs $1,999/month plus $3.50 per order."。"""
        entity = _entity(
            "Growth",
            ["ExtractedEntity", "Entity"],
            summary="Growth plan costs $1,999/month plus $3.50 per order.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept == []
        assert report.dropped[0].reason == "pricing_tier_not_a_speaker"

    def test_price_signature_names_are_dropped(self):
        for name in ("Instant + $3.00 / order", "Priority + $1.50 / order"):
            entity = _entity(name, ["ExtractedEntity", "Entity"], summary="")
            report = EntityQualityFilter().filter_entities([entity])
            assert report.dropped and report.dropped[0].reason == "pricing_tier_not_a_speaker", name

    def test_tier_word_fragments_are_dropped(self):
        for name in ("Starter Intake to pharmacy", "Standard Included Approved orders"):
            entity = _entity(name, ["ExtractedEntity", "Entity"], summary="")
            report = EntityQualityFilter().filter_entities([entity])
            assert report.dropped and report.dropped[0].reason == "pricing_tier_not_a_speaker", name

    def test_person_summary_mentioning_pricing_is_not_dropped(self):
        """反误伤：人物摘要提到自己付了套餐价，不是定价档位。"""
        entity = _entity(
            "Marcus R.",
            ["Person", "Entity"],
            summary="Marcus R. pays for the Starter plan at $699/mo.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "Marcus R."

    def test_person_summary_with_verb_plans_is_not_dropped(self):
        """反误伤："Marcus plans to attend…"里的plans是动词，不是套餐描述。"""
        entity = _entity(
            "Marcus",
            ["Person", "Entity"],
            summary="Marcus plans monthly payments of $699 with his accountant.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "Marcus"

    def test_tier_word_prefixed_non_pricing_names_are_kept(self):
        """反误伤："Growth team"不含套餐/计费特征词，不得按档位删除。"""
        entity = _entity("Growth team", ["ExtractedEntity", "Entity"], summary="A team.")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "Growth team"

    def test_organization_labeled_tier_word_is_kept(self):
        """组织标签豁免：仅凭名称无法区分"Enterprise"是真机构还是套餐档位。"""
        entity = _entity(
            "Enterprise",
            ["Organization", "Entity"],
            summary="Enterprise offers a control plane for managing multiple clinics.",
        )
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "Enterprise"


class TestDrugPseudoEntityFiltering:
    """评审H2回归：药物/激素类伪实体不得成为发言主体（大小写无关）。"""

    def test_drug_names_are_dropped_for_generic_labels(self):
        names = [
            "Testosterone",
            "Peptides",
            "Testosterone Cypionate",
            "progesterone",  # 评审：小写同类被删而大写逃逸——现在大小写都删
            "estradiol",
            "Hormone therapy HRT",
            "Hormone therapy HRT: estradiol, progesterone",
            "TRT Men's hormones",
            "Tretinoin and derm compounds",
            "Low-dose naltrexone LDN protocols",
        ]
        entities = [_entity(n, ["ExtractedEntity", "Entity"], summary="") for n in names]
        report = EntityQualityFilter().filter_entities(entities)
        dropped = {d.entity_name for d in report.dropped}
        for name in names:
            assert name in dropped, name
            assert all(
                d.reason in ("product_not_a_speaker", "generic_lowercase_fragment")
                for d in report.dropped
                if d.entity_name == name
            )

    def test_drug_mislabeled_as_person_is_dropped(self):
        entity = _entity("Testosterone", ["Person", "Entity"], summary="A hormone.")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.dropped[0].reason == "product_mislabeled_as_person"

    def test_organization_with_drug_words_is_kept(self):
        entity = _entity("Hormone Health Clinic", ["Clinic", "Entity"], summary="A clinic.")
        report = EntityQualityFilter().filter_entities([entity])
        assert report.kept and report.kept[0].name == "Hormone Health Clinic"


class TestFilterIntegrationContract:
    def test_neolife_style_corpus_is_filtered_correctly(self):
        """以真实NeoLife图谱中的实体形态做回归验证。"""
        entities = [
            _entity("neolife", ["Organization", "Entity"], summary="neolife is the fulfillment infrastructure."),
            _entity("NVIDIA Corporation", ["Organization", "Entity"], summary="NVIDIA Corporation owns trademarks."),
            _entity("7-Eleven", ["Organization", "Entity"], summary="A convenience store chain."),
            _entity("Marcus R.", ["Person", "Entity"], summary="An order was drafted for Marcus R."),
            _entity("marcus r.", ["ExtractedEntity", "Entity"], summary="An order was drafted for marcus r."),
            _entity("Dana R.", ["Person", "Entity"], summary="A coordinator processes orders."),
            _entity("Dr.", ["Person", "Entity"], summary="Dr. approved orders using a streamlined interface."),
            _entity("Anti-Kickback Statute", ["ExtractedEntity", "Entity"], summary="A fee structure remains constant."),
            _entity("EKRA", ["ExtractedEntity", "Entity"], summary="Fees under EKRA are regulated."),
            _entity("tretinoin 0.05% cream", ["ExtractedEntity", "Entity"], summary="A refill order is being processed."),
            _entity("Testosterone", ["ExtractedEntity", "Entity"], summary="A hormone used in therapy."),
            _entity("Growth", ["ExtractedEntity", "Entity"], summary="Growth plan costs $1,999/month plus $3.50 per order."),
            _entity("Instant + $3.00 / order", ["ExtractedEntity", "Entity"], summary=""),
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
        assert "7-Eleven" in kept_names
        assert "Marcus R." in kept_names
        assert "marcus r." in kept_names
        assert "Dana R." in kept_names
        assert "Crunchbase" in kept_names
        assert "partner negotiators" in kept_names

        dropped_names = {d.entity_name for d in report.dropped}
        assert "Dr." in dropped_names
        assert "Anti-Kickback Statute" in dropped_names
        assert "EKRA" in dropped_names
        assert "tretinoin 0.05% cream" in dropped_names
        assert "Testosterone" in dropped_names
        assert "Growth" in dropped_names
        assert "Instant + $3.00 / order" in dropped_names
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

    def test_pure_quality_filter_does_not_dedupe(self):
        """边界：EntityQualityFilter只做质量决策；"Marcus R."与"marcus r."
        两个实体都保留（去重属于prepare级管道，filter_entities_for_profiles）。
        """
        entities = [
            _entity("Marcus R.", ["Person", "Entity"], summary="An order was drafted."),
            _entity("marcus r.", ["ExtractedEntity", "Entity"], summary="An order was drafted."),
        ]
        report = EntityQualityFilter().filter_entities(entities)
        assert {e.name for e in report.kept} == {"Marcus R.", "marcus r."}
        assert report.merged == []


class TestDuplicateSpeakerDedupe:
    """prepare级重复发言主体守卫：同一归一化身份只保留一个发言主体。

    真实NeoLife图谱把同一组织抽成了"NeoLife"与"NeoLife Official"两个
    实体，各自生成了一个Agent人设，导致同一现实主体在模拟中重复发言。
    """

    def test_neolife_and_neolife_official_merge_into_one_speaker(self):
        entities = [
            _entity(
                "NeoLife",
                ["Organization", "Entity"],
                summary="Fulfillment infrastructure.",
                related_edges=[{"fact": "NeoLife ships orders.", "direction": "outgoing"}],
            ),
            _entity(
                "NeoLife Official",
                ["Organization", "Entity"],
                summary="Official account.",
                related_edges=[],
            ),
        ]
        report = filter_entities_for_profiles(entities)

        assert len(report.kept) == 1
        assert report.kept[0].name == "NeoLife"
        assert len(report.merged) == 1
        merged = report.merged[0]
        assert merged.entity_name == "NeoLife Official"
        assert merged.kept_name == "NeoLife"
        assert merged.identity == "neolife"
        assert merged.reason == "duplicate_speaker_identity"

    def test_case_and_whitespace_folded_duplicates_merge(self):
        entities = [
            _entity("neolife", ["Organization", "Entity"], summary="a"),
            _entity("NeoLife", ["Organization", "Entity"], summary="b"),
            _entity(" NeoLife  ", ["Organization", "Entity"], summary="c"),
        ]
        report = filter_entities_for_profiles(entities)

        assert len(report.kept) == 1
        assert len(report.merged) == 2

    def test_richest_entity_is_kept_regardless_of_input_order(self):
        thin = _entity(
            "NeoLife",
            ["Organization", "Entity"],
            # 有摘要才会通过低信息过滤——这里隔离测试的是去重的选择规则
            summary="Some context.",
        )
        rich = _entity(
            "NeoLife Official",
            ["Organization", "Entity"],
            summary="Official account of NeoLife.",
            related_edges=[
                {"fact": "NeoLife Official posts pricing updates.", "direction": "outgoing"},
                {"fact": "NeoLife Official replies to clinics.", "direction": "outgoing"},
            ],
        )
        report = filter_entities_for_profiles([thin, rich])
        assert report.kept[0].name == "NeoLife Official"
        assert report.merged[0].entity_name == "NeoLife"

        # 信息量相同时平局取输入序最前者（完全确定）
        equal_one = _entity("NeoLife", ["Organization", "Entity"], summary="same")
        equal_two = _entity("NeoLife Official", ["Organization", "Entity"], summary="same")
        report = filter_entities_for_profiles([equal_one, equal_two])
        assert report.kept[0].name == "NeoLife"
        assert report.merged[0].entity_name == "NeoLife Official"

    def test_kept_order_follows_input_order(self):
        entities = [
            _entity("Marcus R.", ["Person", "Entity"], summary="a"),
            _entity("NeoLife", ["Organization", "Entity"], summary="b"),
            _entity("marcus r.", ["ExtractedEntity", "Entity"], summary="c"),
        ]
        report = filter_entities_for_profiles(entities)
        assert [e.name for e in report.kept] == ["Marcus R.", "NeoLife"]

    def test_unrelated_aliases_are_never_merged(self):
        """身份合并刻意保守：不同品牌拼写/后缀词不得混为一谈。"""
        entities = [
            _entity("NeoLife", ["Organization", "Entity"], summary="a"),
            _entity("Neo", ["Organization", "Entity"], summary="b"),
            _entity("NeoLife Labs", ["Organization", "Entity"], summary="c"),
            _entity("Neo-Life", ["Organization", "Entity"], summary="d"),
            _entity("NeoLife Team", ["Organization", "Entity"], summary="e"),
            _entity("BarOfficial", ["Organization", "Entity"], summary="f"),
        ]
        report = filter_entities_for_profiles(entities)

        assert len(report.kept) == len(entities)
        assert report.merged == []

    def test_glued_official_suffix_is_not_stripped(self):
        """后缀必须是独立结尾词："BarOfficial"（粘连词）不剥离、不与
        "Bar"合并；"X Official"（独立词）才剥离。"""
        entities = [
            _entity("Bar", ["Organization", "Entity"], summary="a"),
            _entity("BarOfficial", ["Organization", "Entity"], summary="b"),
        ]
        report = filter_entities_for_profiles(entities)
        assert len(report.kept) == 2
        assert report.merged == []

        entities = [
            _entity("Bar", ["Organization", "Entity"], summary="a"),
            _entity("Bar Official", ["Organization", "Entity"], summary="b"),
        ]
        report = filter_entities_for_profiles(entities)
        assert len(report.kept) == 1
        assert report.merged[0].entity_name == "Bar Official"

    def test_official_account_and_page_suffixes_all_group(self):
        entities = [
            _entity("NeoLife", ["Organization", "Entity"], summary="a"),
            _entity("NeoLife Official", ["Organization", "Entity"], summary="b"),
            _entity("NeoLife Official Account", ["Organization", "Entity"], summary="c"),
            _entity("NeoLife Official Page", ["Organization", "Entity"], summary="d"),
        ]
        report = filter_entities_for_profiles(entities)
        assert len(report.kept) == 1
        assert len(report.merged) == 3

    def test_suffix_only_name_is_not_merged_with_unrelated_entity(self):
        """纯后缀名"Official"保持原样：不与"Bar"合并，只与同名实体合并。"""
        entities = [
            _entity("Bar", ["Organization", "Entity"], summary="a"),
            _entity("Official", ["Organization", "Entity"], summary="b"),
        ]
        report = filter_entities_for_profiles(entities)
        assert len(report.kept) == 2
        assert report.merged == []

    def test_empty_names_pass_through_without_collapsing(self, monkeypatch):
        """空名称无身份意义：不参与合并分组、不互相折叠。质量过滤开启
        时空名称会被先行剔除，这里关闭过滤以隔离去重行为。"""
        monkeypatch.setenv("MIROFISH_ENTITY_QUALITY_FILTER", "0")
        entities = [
            _entity("", ["Person", "Entity"], summary=""),
            _entity("", ["Person", "Entity"], summary=""),
        ]
        report = filter_entities_for_profiles(entities)
        assert len(report.kept) == 2
        assert report.merged == []

    def test_dedupe_runs_even_when_quality_filter_is_disabled(self, monkeypatch):
        """去重是正确性守卫：质量过滤开关不影响同一身份只保留一个发言人。"""
        monkeypatch.setenv("MIROFISH_ENTITY_QUALITY_FILTER", "0")

        entities = [
            _entity("NeoLife", ["Organization", "Entity"], summary="a"),
            _entity("NeoLife Official", ["Organization", "Entity"], summary="b"),
        ]
        report = filter_entities_for_profiles(entities)

        assert len(report.kept) == 1
        assert report.dropped == []
        assert len(report.merged) == 1

    def test_report_dict_includes_merged_block(self):
        import json

        entities = [
            _entity("NeoLife", ["Organization", "Entity"], summary="a"),
            _entity("NeoLife Official", ["Organization", "Entity"], summary="b"),
        ]
        report = filter_entities_for_profiles(entities)
        payload = report.to_dict()

        assert payload["kept_count"] == 1
        assert payload["merged_count"] == 1
        assert payload["total_input"] == 2
        assert payload["merged"][0]["entity_name"] == "NeoLife Official"
        assert payload["merged"][0]["action"] == "merge"
        assert payload["merged"][0]["kept_name"] == "NeoLife"
        # 审计块必须可序列化（写入entity_quality_report.json）
        json.dumps(payload, ensure_ascii=False)

    def test_dedupe_drops_no_entities_when_all_identities_unique(self):
        entities = [
            _entity("NeoLife", ["Organization", "Entity"], summary="a"),
            _entity("NVIDIA Corporation", ["Organization", "Entity"], summary="b"),
            _entity("Marcus R.", ["Person", "Entity"], summary="c"),
        ]
        report = filter_entities_for_profiles(entities)
        assert [e.name for e in report.kept] == [
            "NeoLife",
            "NVIDIA Corporation",
            "Marcus R.",
        ]
        assert report.merged == []
