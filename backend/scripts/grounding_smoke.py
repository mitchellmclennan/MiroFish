"""Source-to-prompt grounding smoke audit.

Bounded, read-only audit that proves the grounding pipeline end to end
against a live graph WITHOUT restarting the MiroFish stack or running an
OASIS simulation:

1. Reads entities from the graph (read-only Zep calls).
2. Applies the entity quality filter and reports what would be dropped.
3. For sampled entities, builds the real grounding context and verifies:
   - OpenZep search facts are actually retrieved (non-zero facts);
   - the English persona prompt contains those facts verbatim;
   - the prompt contains the STRICT GROUNDING RULES no-invention block.
4. Optionally calls the configured cheap-tier LLM (gpt-4o-mini) to
   generate a few real personas, records provenance, and writes
   prompts/personas/provenance artifacts for manual inspection.

Usage:
    python scripts/grounding_smoke.py --graph-id mirofish_xxx \
        [--samples "Name One,Name Two"] [--llm-count 3] \
        [--output-dir /tmp/opencode/grounding-smoke]

The script refuses to run if LLM_MODEL_NAME is not the expected cheap
tier (no silent model upgrades).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, ".."))
_project_root = os.path.abspath(os.path.join(_backend_dir, ".."))
sys.path.insert(0, _backend_dir)

from dotenv import load_dotenv  # noqa: E402

_env_file = os.path.join(_project_root, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)

# 本审计面向英文运行：在任何app导入前固定语言策略
os.environ.setdefault("MIROFISH_LLM_LANGUAGE", "en")

from app.config import Config  # noqa: E402
from app.services.entity_quality_filter import (  # noqa: E402
    EntityQualityFilter,
    filter_entities_for_profiles,
)
from app.services.oasis_profile_generator import (  # noqa: E402
    ENGLISH_SEARCH_QUERY_TEMPLATE,
    PersonaProvenance,
    OasisProfileGenerator,
)
from app.services.zep_entity_reader import ZepEntityReader  # noqa: E402
from app.utils.locale import is_english_forced  # noqa: E402

EXPECTED_MODEL = "gpt-4o-mini"
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def _guard_model_tier():
    model = os.environ.get("LLM_MODEL_NAME", Config.LLM_MODEL_NAME)
    if model != EXPECTED_MODEL:
        raise SystemExit(
            f"Model guard failed: LLM_MODEL_NAME={model!r} but this audit is "
            f"restricted to the cheap tier {EXPECTED_MODEL!r}. Refusing to run."
        )
    return model


def main():
    parser = argparse.ArgumentParser(description="Source-to-prompt grounding smoke")
    parser.add_argument("--graph-id", required=True, help="Zep graph id to audit")
    parser.add_argument(
        "--samples",
        default="",
        help="Comma-separated entity names to audit (default: auto-pick)",
    )
    parser.add_argument(
        "--llm-count",
        type=int,
        default=3,
        help="How many sampled entities get a real LLM persona (0 disables LLM calls)",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/opencode/grounding-smoke",
        help="Directory for audit artifacts",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip the entity quality filter stage",
    )
    args = parser.parse_args()

    started = time.time()
    model = _guard_model_tier()
    assert is_english_forced(), "MIROFISH_LLM_LANGUAGE=en must be in effect"

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "graph_id": args.graph_id,
        "started_at": datetime.now().isoformat(),
        "model": model,
        "llm_base_url": os.environ.get("LLM_BASE_URL", Config.LLM_BASE_URL),
        "english_forced": True,
        "entity_filter_enabled": not args.skip_filter,
    }

    # ---------- Stage 1: read entities ----------
    reader = ZepEntityReader()
    filtered = reader.filter_defined_entities(
        graph_id=args.graph_id,
        defined_entity_types=None,
        enrich_with_edges=True,
    )
    report["graph_entities_total_nodes"] = filtered.total_count
    report["graph_entities_defined"] = filtered.filtered_count
    report["graph_entity_types"] = sorted(filtered.entity_types)

    # ---------- Stage 2: entity quality filter ----------
    if not args.skip_filter:
        quality = filter_entities_for_profiles(filtered.entities)
        report["entity_quality"] = quality.to_dict()
        pool = quality.kept
    else:
        pool = filtered.entities

    # ---------- Stage 3: pick samples ----------
    wanted = [name.strip() for name in args.samples.split(",") if name.strip()]
    if wanted:
        wanted_set = set(wanted)
        seen_names = set()
        samples = []
        for entity in pool:
            if entity.name in wanted_set and entity.name not in seen_names:
                samples.append(entity)
                seen_names.add(entity.name)
        missing = [name for name in wanted if name not in seen_names]
        if missing:
            print(f"WARNING: sample names not found in graph: {missing}")
    else:
        # Auto-pick: prefer a spread across entity types
        by_type = {}
        for entity in pool:
            by_type.setdefault(entity.get_entity_type() or "Unknown", []).append(entity)
        samples = []
        for entity_type in sorted(by_type):
            candidates = sorted(by_type[entity_type], key=lambda e: len(e.related_edges or []), reverse=True)
            samples.append(candidates[0])
        samples = samples[:5]

    if not samples:
        raise SystemExit("No sampled entities to audit")

    # ---------- Stage 4: grounding context + prompt verification ----------
    generator = OasisProfileGenerator(graph_id=args.graph_id)
    audit_entries = []
    failures = []

    for entity in samples:
        grounding = generator._build_grounding_context(entity)
        prompt = (
            generator._build_individual_persona_prompt(
                entity.name, entity.get_entity_type() or "Entity", entity.summary,
                entity.attributes, grounding.context_text,
            )
            if generator._is_individual_entity(entity.get_entity_type() or "")
            else generator._build_group_persona_prompt(
                entity.name, entity.get_entity_type() or "Entity", entity.summary,
                entity.attributes, grounding.context_text,
            )
        )
        entry = {
            "entity": entity.name,
            "entity_type": entity.get_entity_type(),
            "related_edge_count": len(entity.related_edges or []),
            "search_attempted": grounding.search_attempted,
            "search_query": grounding.search_query,
            "search_facts_returned": grounding.search_facts_returned,
            "facts_ledgered": len(grounding.facts),
            "context_chars": len(grounding.context_text),
            "prompt_chars": len(prompt),
            "facts": grounding.facts,
            "prompt": prompt,
        }

        # Source-to-prompt checks
        checks = {}
        if grounding.search_attempted and grounding.search_facts_returned == 0 and not (entity.related_edges or []):
            checks["retrieved_graph_facts_present"] = False
            failures.append(f"{entity.name}: no graph facts retrieved or attached")
        else:
            checks["retrieved_graph_facts_present"] = True
        injected = [
            f["text"] for f in grounding.facts if f["source"] in ("zep_search", "related_edge")
            and f["text"] in prompt
        ]
        checks["facts_injected_into_prompt"] = len(injected) >= min(1, len(grounding.facts))
        checks["no_invention_rule_present"] = "STRICT GROUNDING RULES" in prompt
        checks["english_prompt"] = "Graph context (source facts)" in prompt and "Respond in English only" in prompt
        entry["checks"] = checks
        if not all(checks.values()):
            failures.append(f"{entity.name}: failed checks {checks}")
        audit_entries.append(entry)

    report["samples"] = audit_entries

    # ---------- Stage 5: real LLM personas (bounded) ----------
    llm_entries = []
    if args.llm_count > 0:
        llm_entities = samples[: args.llm_count]
        for entity in llm_entities:
            t0 = time.time()
            profile = generator.generate_profile_from_entity(
                entity=entity, user_id=0, use_llm=True
            )
            elapsed = time.time() - t0
            llm_entries.append({
                "entity": entity.name,
                "entity_type": profile.source_entity_type,
                "elapsed_seconds": round(elapsed, 1),
                "model": model,
                "bio": profile.bio,
                "persona": profile.persona,
                "country": profile.country,
                "gender": profile.gender,
                "profession": profile.profession,
                "provenance": profile.provenance,
                "persona_has_cjk": bool(_CJK_PATTERN.search(profile.persona or "")),
                "persona_has_english": bool(re.search(r"[A-Za-z]{3,}", profile.persona or "")),
            })
            print(f"generated persona for {entity.name!r} in {elapsed:.1f}s")

    report["llm_personas"] = llm_entries
    report["failures"] = failures
    report["passed"] = not failures
    report["elapsed_seconds"] = round(time.time() - started, 1)
    report["finished_at"] = datetime.now().isoformat()

    output_path = os.path.join(
        args.output_dir,
        f"grounding_smoke_{args.graph_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "passed": report["passed"],
        "failures": failures,
        "samples": len(samples),
        "llm_personas": len(llm_entries),
        "output": output_path,
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
