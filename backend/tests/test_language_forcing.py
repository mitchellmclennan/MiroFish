"""Run-level English forcing contract tests.

MIROFISH_LLM_LANGUAGE=en must force every MiroFish-controlled LLM prompt
site (ontology, personas, action episode text, report, simulation config)
to emit English output instructions, without changing default behavior.
"""

import json
import re
from types import SimpleNamespace

import pytest

from app.utils.locale import (
    FORCED_ENGLISH_LLM_INSTRUCTION,
    get_language_instruction,
    is_english_forced,
)


@pytest.fixture(autouse=True)
def _no_forced_language_by_default(monkeypatch):
    monkeypatch.delenv("MIROFISH_LLM_LANGUAGE", raising=False)


def _force_english(monkeypatch):
    monkeypatch.setenv("MIROFISH_LLM_LANGUAGE", "en")


def test_default_language_instruction_stays_chinese(monkeypatch):
    monkeypatch.delenv("MIROFISH_LLM_LANGUAGE", raising=False)

    assert not is_english_forced()
    instruction = get_language_instruction()
    assert "中文" in instruction
    assert "English" not in instruction


def test_forced_english_instruction_wins_over_locale(monkeypatch):
    _force_english(monkeypatch)
    assert is_english_forced()
    instruction = get_language_instruction()
    assert instruction == FORCED_ENGLISH_LLM_INSTRUCTION
    assert "English" in instruction
    assert "JSON string value" in instruction


def test_forced_english_helper_toggles_with_env(monkeypatch):
    monkeypatch.delenv("MIROFISH_LLM_LANGUAGE", raising=False)
    assert not is_english_forced()
    monkeypatch.setenv("MIROFISH_LLM_LANGUAGE", "en")
    assert is_english_forced()
    monkeypatch.setenv("MIROFISH_LLM_LANGUAGE", "EN")
    assert is_english_forced()


def _make_generator():
    from app.services.oasis_profile_generator import OasisProfileGenerator

    return OasisProfileGenerator(
        api_key="test-key",
        zep_api_key="test-zep-key",
        graph_id="graph-test",
    )


def test_persona_system_prompt_is_english_when_forced(monkeypatch):
    _force_english(monkeypatch)
    generator = _make_generator()

    system_prompt = generator._get_system_prompt(is_individual=True)

    assert "persona writer" in system_prompt
    assert "valid JSON" in system_prompt
    assert FORCED_ENGLISH_LLM_INSTRUCTION in system_prompt
    assert "使用中文" not in system_prompt


def test_individual_persona_prompt_english_contract(monkeypatch):
    _force_english(monkeypatch)
    generator = _make_generator()

    prompt = generator._build_individual_persona_prompt(
        entity_name="Marcus R.",
        entity_type="Person",
        entity_summary="An order was drafted for Marcus R.",
        entity_attributes={},
        context="## facts\n- Starter pricing is $699/mo.",
    )

    # Grounding contract: source facts block + explicit no-invention rule
    assert "Graph context (source facts)" in prompt
    assert "Starter pricing is $699/mo." in prompt
    assert "STRICT GROUNDING RULES" in prompt
    assert "Do NOT invent" in prompt
    # English output contract for structured fields
    assert '"male" or "female"' in prompt
    assert 'country name in English' in prompt
    assert "Respond in English only" in prompt
    # The Chinese-only field instructions must not leak into English runs
    assert "使用中文" not in prompt
    assert "国家" not in prompt


def test_group_persona_prompt_english_contract(monkeypatch):
    _force_english(monkeypatch)
    generator = _make_generator()

    prompt = generator._build_group_persona_prompt(
        entity_name="neolife",
        entity_type="Organization",
        entity_summary="neolife is the fulfillment infrastructure.",
        entity_attributes={},
        context="## facts\n- neolife offers Starter at $699/mo.",
    )

    assert "Graph context (source facts)" in prompt
    assert "STRICT GROUNDING RULES" in prompt
    assert "Do NOT invent" in prompt
    assert '"other"' in prompt
    assert 'country name in English' in prompt
    assert "使用中文" not in prompt


def test_default_persona_prompts_keep_chinese_contract(monkeypatch):
    monkeypatch.delenv("MIROFISH_LLM_LANGUAGE", raising=False)
    generator = _make_generator()

    individual = generator._build_individual_persona_prompt(
        entity_name="张三", entity_type="Person", entity_summary="", entity_attributes={}, context="上下文"
    )
    group = generator._build_group_persona_prompt(
        entity_name="某组织", entity_type="Organization", entity_summary="", entity_attributes={}, context=""
    )
    system = generator._get_system_prompt(is_individual=False)

    # Chinese default behavior is preserved, now with an explicit
    # no-invention clause.
    for prompt in (individual, group, system):
        assert "请使用中文回答" in prompt
    assert "严禁编造" in individual
    assert "严禁编造" in group
    assert "国家（使用中文" in individual


def test_memory_updater_episode_text_is_english_when_forced(monkeypatch):
    _force_english(monkeypatch)
    from app.services.zep_graph_memory_updater import AgentActivity

    post = AgentActivity(
        platform="twitter",
        agent_id=1,
        agent_name="agent_1",
        action_type="CREATE_POST",
        action_args={"content": "Pricing feels fair"},
        round_num=2,
        timestamp="2026-09-20T10:00:00",
    )
    like = AgentActivity(
        platform="twitter",
        agent_id=2,
        agent_name="agent_2",
        action_type="LIKE_POST",
        action_args={"post_content": "Great product", "post_author_name": "agent_1"},
        round_num=2,
        timestamp="2026-09-20T10:05:00",
    )
    unknown = AgentActivity(
        platform="twitter",
        agent_id=3,
        agent_name="agent_3",
        action_type="SOMETHING_ELSE",
        action_args={},
        round_num=2,
        timestamp="2026-09-20T10:06:00",
    )

    for activity in (post, like, unknown):
        text = activity.to_episode_text()
        assert not re.search(r"[\u4e00-\u9fff]", text), text

    assert 'posted: "Pricing feels fair"' in post.to_episode_text()
    assert "liked agent_1's post" in like.to_episode_text()
    assert "performed the SOMETHING_ELSE action" in unknown.to_episode_text()


def test_memory_updater_episode_text_default_stays_chinese(monkeypatch):
    monkeypatch.delenv("MIROFISH_LLM_LANGUAGE", raising=False)
    from app.services.zep_graph_memory_updater import AgentActivity

    activity = AgentActivity(
        platform="twitter",
        agent_id=1,
        agent_name="agent_1",
        action_type="CREATE_POST",
        action_args={"content": "价格不错"},
        round_num=1,
        timestamp="2026-09-20T10:00:00",
    )

    text = activity.to_episode_text()
    assert "发布了一条帖子" in text
    assert "价格不错" in text


def test_ontology_system_prompt_carries_english_only_directive(monkeypatch):
    _force_english(monkeypatch)
    from app.services.ontology_generator import OntologyGenerator

    captured = {}

    class StubLLM:
        def chat_json(self, messages, **kwargs):
            captured["messages"] = messages
            return {"entity_types": [], "edge_types": [], "analysis_summary": ""}

    generator = OntologyGenerator(llm_client=StubLLM())
    generator.generate(document_texts=["doc"], simulation_requirement="req")

    system_prompt = captured["messages"][0]["content"]
    assert "PascalCase" in system_prompt
    assert "Write ALL natural-language output" in system_prompt
    assert "English only" in system_prompt


def test_simulation_config_prompts_carry_forced_instruction(monkeypatch):
    _force_english(monkeypatch)
    from app.services.simulation_config_generator import SimulationConfigGenerator

    captured = {}

    def fake_llm_call(self, prompt, system_prompt):
        captured["system_prompt"] = system_prompt
        return {}

    monkeypatch.setattr(
        SimulationConfigGenerator, "_call_llm_with_retry", fake_llm_call
    )
    generator = SimulationConfigGenerator(api_key="test-key")
    generator._generate_time_config(context="ctx", num_entities=10)

    assert FORCED_ENGLISH_LLM_INSTRUCTION in captured["system_prompt"]


def test_report_plan_outline_prompt_carries_forced_instruction(monkeypatch):
    _force_english(monkeypatch)
    from app.services.report_agent import ReportAgent

    captured = {}

    class StubLLM:
        def chat_json(self, messages, **kwargs):
            captured["messages"] = messages
            return {
                "title": "Report",
                "summary": "Summary",
                "sections": [{"title": "Section"}],
            }

    class StubTools:
        def get_simulation_context(self, graph_id, simulation_requirement):
            return {
                "graph_statistics": {"total_nodes": 1, "total_edges": 1, "entity_types": {}},
                "total_entities": 1,
                "related_facts": [],
            }

    agent = ReportAgent(
        graph_id="graph",
        simulation_id="sim",
        simulation_requirement="requirement",
        llm_client=StubLLM(),
        zep_tools=StubTools(),
    )
    agent.plan_outline()

    system_prompt = captured["messages"][0]["content"]
    assert FORCED_ENGLISH_LLM_INSTRUCTION in system_prompt
