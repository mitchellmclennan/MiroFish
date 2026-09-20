# Grounded English runs (NeoLife pricing/positioning lane)

This document describes the three grounding fixes introduced for bounded
NeoLife pricing/positioning runs, their configuration switches, and the
audit artifacts they produce. These changes apply to all runs; the English
run described here is opt-in via one environment variable.

## 1. Persona generation is grounded in OpenZep graph facts

`backend/app/services/oasis_profile_generator.py`:

- **Retrieval actually works against the local OpenZep server.** The local
  compatibility server returns search payloads under a `results` key of
  plain dicts, while the `zep-cloud` SDK exposes `edges`/`nodes`. Both
  shapes are now consumed; previously the local payloads were silently
  dropped and persona prompts received zero retrieved facts.
- **Retrieval queries no longer depend on UI locale.** When English is
  forced, the Zep search query uses a fixed English template
  (`ENGLISH_SEARCH_QUERY_TEMPLATE`). A Chinese-locale template query does
  not match an English-language graph.
- **Source facts are injected into every persona prompt** in a clearly
  labeled block (`Graph context (source facts)`; `图谱事实` in Chinese
  runs) together with an explicit **no-invention rule**
  (`STRICT GROUNDING RULES` in English runs, `严禁编造` in Chinese runs):
  identity/history/relationship claims must come from the injected facts;
  missing details must be described generically, not fabricated.
- **Auditable provenance** is recorded per persona:
  - `PERSONA_PROVENANCE <json>` log line per persona (facts ledger with
    `related_edge` / `zep_search` / `zep_search_node_summary` /
    `entity_attribute` sources, retrieval stats, model, fallback flags);
  - `persona_provenance.json` sidecar in the simulation directory
    (written live during `prepare_simulation`);
  - the `provenance` field on `OasisAgentProfile.to_dict()` (excluded from
    the strict OASIS reddit/twitter output formats).

## 2. Run-level English forcing

Set the environment variable before starting the backend (or the run
scripts):

```
MIROFISH_LLM_LANGUAGE=en
```

Effects (implemented in `backend/app/utils/locale.py` and the prompt sites
that append `get_language_instruction()`):

| Pipeline stage | English behavior |
| --- | --- |
| Ontology generation | explicit "write all natural-language output in English" directive; type/relation/attribute names were already English by contract |
| Persona generation | full English prompt templates (system + individual + group), English field contract (`gender` male/female/other, `country` in English) |
| Action/post episode text pushed back to Zep | English templates in `zep_graph_memory_updater.py` (`posted: "..."`, `liked X's post: ...`) so mid-run extraction stays English |
| Simulation config (initial posts, narratives) | forced English instruction appended to every config prompt |
| Report generation | forced English instruction appended to outline/section/chat prompts |

Notes: OASIS/camel's internal action prompts are English by default;
English personas (`user_char`) and English initial posts keep generated
posts English. Default (unset) behavior is unchanged: locale-driven
Chinese, matching pre-existing behavior.

## 3. Junk-entity filtering before profile generation

`backend/app/services/entity_quality_filter.py` (rule-based, deterministic,
no LLM cost) drops or fixes, before any profile is generated:

- **statutes / legal documents** (by label, or by name: Anti-Kickback,
  EKRA, AKS, HIPAA, U.S.C., C.F.R., "X Act", section/title citations);
- **products mislabeled as people** (dropped), and product-shaped names
  (dosage forms, concentrations, "compounded", ODT) for non-organization
  labels — products are not social-media speakers;
- **companies mislabeled as people** — relabeled to `Organization` and
  kept (legitimate companies remain legitimate speakers);
- **boilerplate fragments** — empty names, placeholders (`N/A`, `Unknown`),
  honorifics (`Dr.`, `CEO`), filenames, URLs/emails, number-led fragments
  (`300 orders/mo`), lowercase generic fragments for untyped
  (`ExtractedEntity`) nodes;
- **low-information entities** — no summary, no attributes, and no
  related edges: nothing to ground a persona on.

Legitimate organizations (including lowercase ones like `neolife`) are
kept untouched. The filter is **on by default**; disable with
`MIROFISH_ENTITY_QUALITY_FILTER=0`. Audit artifacts:

- `entity_quality_report.json` in the simulation directory (kept/dropped/
  relabeled with reasons);
- the `POST /api/simulation/generate-profiles` response now includes an
  `entity_quality` block.

## Smoke audit tooling

`backend/scripts/grounding_smoke.py` performs a bounded, read-only
source-to-prompt audit against a live graph (no stack restart, no OASIS
run): entity read → quality filter → grounding context → prompt checks
(facts injected, no-invention block present, English contract) → optional
real `gpt-4o-mini` personas with provenance. It refuses to run unless
`LLM_MODEL_NAME=gpt-4o-mini` (no silent model upgrades).

```
.venv/bin/python scripts/grounding_smoke.py --graph-id <graph_id> \
    --samples "Name One,Name Two" --llm-count 3 \
    --output-dir /tmp/opencode/grounding-smoke
```

## Full NeoLife run recipe (when the stack is next started)

```
# .env additions for the run process (backend + simulation scripts):
MIROFISH_LLM_LANGUAGE=en
# LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME=gpt-4o-mini already configured
```

The simulation directory will then contain, per run:
`reddit_profiles.json` / `twitter_profiles.csv`, `persona_provenance.json`,
`entity_quality_report.json`, `simulation_config.json`, plus
`PERSONA_PROVENANCE` lines in backend logs.
