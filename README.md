# Clinical Trials Query-to-Visualization Agent

A backend service that converts natural-language clinical-trial questions into structured
visualization specifications, backed by live data from the [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api).

Ask "How has pembrolizumab trial activity changed since 2015?" and get back a fully-specified
time-series chart — real bucket counts, a frontend-ready encoding, deep citations back to the
individual studies that produced each number, and a two-sentence summary of what the data shows.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add your OPENAI_API_KEY (optional -- see below)
uvicorn app.main:app --reload
```

- **Demo UI:** [http://localhost:8000](http://localhost:8000)
- **OpenAPI docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health check:** `GET /health`

Requires Python 3.11+. No database, no external services beyond ClinicalTrials.gov and
(optionally) OpenAI.

**No `OPENAI_API_KEY`? The service still works.** Every query falls back to a deterministic
keyword classifier and still returns a valid, structured 200 response — see
[Design Decisions](#design-decisions).

### Running tests

```bash
pytest tests/unit              # 294 tests, zero network, zero API key, <1s
pytest tests/unit -m live       # a handful of tests that hit the real API (opt-in)
ruff check .
```

### Capturing example outputs

```bash
python examples/run_examples.py
```

Regenerates the five files in `examples/outputs/` against the live API (and the live LLM, if
`OPENAI_API_KEY` is set).

---

## How It Works

A **two-touch agent**: an LLM touches the request once at the start (planning + classification)
and once at the end (interpreting the results), with a fully deterministic pipeline in between.
The LLM never computes a count, never sees a raw study record, and never constructs an API URL.

```
request
  → [1] validate                    (pydantic — QueryRequest)
  → [2] PLAN + CLASSIFY      (LLM)  ← Touch 1: intent + entities + query_plan
        → cross-field sanity check → retry once → heuristic fallback
  → [3] build CTGov params          (deterministic, allow-listed)
  → [4] fetch studies                (deterministic, paginated, rate-limited)
  → [5] extract records              (deterministic, safe_get everywhere)
  → [6] aggregate                    (deterministic, composable extractors)
  → [7] attach citations             (deterministic, verbatim excerpts)
  → [8] INTERPRET RESULTS    (LLM)  ← Touch 2: 2-3 sentence summary + quality flags
  → [9] assemble response
  → response
```

Steps 2 and 8 are the only two points an LLM is involved. Steps 3–7 are pure functions —
unit-tested with zero network and zero API key.

### Query breadth from composition, not one-off handlers

Six analysis types (`trend`, `distribution`, `comparison`, `geographic`, `network`, `count`) are
all built from the same small set of primitives:

- **7 extractors** (`app/analysis/extractors.py`) — `by_year`, `by_phase`, `by_status`,
  `by_sponsor_class`, `by_sponsor_name`, `by_country`, `by_intervention_type`. Each turns one
  `TrialRecord` into `(bucket_key, nct_id, field_path, excerpt)` tuples.
- **2 aggregation functions** (`app/analysis/aggregate.py`) — `group_and_count()` (used by
  trend, distribution, and geographic — same function, different extractor) and
  `compare_groups()` (zero-fills across both sides for a grouped bar chart).
- **1 network builder** (`app/analysis/network.py`) — sponsor↔drug bipartite by default,
  drug↔drug co-occurrence when the query is specifically about combinations.

A new distribution dimension, or a new phrasing of an existing query type, is a new extractor or
a new keyword pattern — never a new branch through the whole pipeline.

---

## API Reference

### `POST /query`

**Request** (`QueryRequest`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `query` | string | **yes** | Natural-language question, min 8 characters. |
| `drug_name` | string | no | Intervention name filter. |
| `condition` | string | no | Condition/disease filter. |
| `trial_phase` | string | no | e.g. `"PHASE1"` or `"PHASE1,PHASE2"`. |
| `sponsor` | string | no | Sponsor name filter. |
| `country` | string | no | Location filter. |
| `status` | string | no | e.g. `"RECRUITING"`. |
| `start_year` / `end_year` | int | no | Bounds `[1990, 2100]`; `start_year <= end_year` enforced. |
| `compare_a` / `compare_b` | string | no | What to compare, for a comparison query. |
| `compare_type` | string | no | `drug` / `condition` / `sponsor` — which field `compare_a`/`compare_b` fill in. |
| `dimension` | string | no | Breakdown axis for distribution/comparison, or `"drug_cooccurrence"` for a drug↔drug network. |
| `max_studies` | int | no | Default 500, max 5000 (pagination hard cap). |
| `include_citations` | bool | no | Default `true`. |
| `include_summary` | bool | no | Default `false` — Touch 2 costs an extra LLM call. |

Every structured field above is **ground truth**: if you supply it, it overrides whatever the
LLM (or heuristic) guessed from the query text — see
[Ground-truth overrides](#ground-truth-overrides).

```json
{
  "query": "How has the number of trials for pembrolizumab changed per year since 2015?",
  "drug_name": "pembrolizumab",
  "start_year": 2015
}
```

**Response** (`QueryResponse`) — bucket-based charts (`bar_chart`, `grouped_bar_chart`,
`time_series`, `histogram`, `scatter_plot`):

```json
{
  "visualization": {
    "type": "time_series",
    "title": "pembrolizumab: trials started per year",
    "encoding": {
      "x": { "field": "year", "type": "temporal" },
      "y": { "field": "count", "type": "quantitative" }
    },
    "data": [
      {
        "x": "2015", "y": 6, "series": null,
        "citations": [
          {
            "nct_id": "NCT02325557",
            "field_path": "protocolSection.statusModule.startDateStruct.date",
            "excerpt": "2015-06-04",
            "url": "https://clinicaltrials.gov/study/NCT02325557"
          }
        ]
      }
    ],
    "network_data": null
  },
  "summary": "Since 2015, the number of trials for pembrolizumab has shown a peak in 2017 with 27 trials, followed by a general decline in subsequent years, except for an increase to 24 trials in both 2018 and 2022...",
  "meta": {
    "query_interpretation": "This query is asking for the year-by-year change in clinical trials associated with the drug pembrolizumab from 2015 onwards.",
    "query_plan": "Retrieve the number of clinical trials involving pembrolizumab for each year starting from 2015 to the present. The data will be visualized as a line graph to show trends over time in the number of trials.",
    "analysis_type": "trend",
    "filters_applied": { "drug_name": "pembrolizumab", "start_year": 2015 },
    "assumptions": [],
    "total_studies_matched": 2845,
    "total_studies_fetched": 200,
    "unique_study_count": 200,
    "source": "https://clinicaltrials.gov/api/v2/studies",
    "generated_at": "2026-07-26T21:34:00Z",
    "intent_source": "llm"
  }
}
```

`data[].series` is non-null only for a `grouped_bar_chart` (comparison), naming which side
(`compare_a`/`compare_b`) that bar belongs to.

**Response — `network_graph`, a genuinely different shape:**

```json
{
  "visualization": {
    "type": "network_graph",
    "title": "breast cancer: trial network",
    "encoding": {
      "nodes": { "id": "id", "label": "label", "group": "type", "size": "weight" },
      "edges": { "source": "source", "target": "target", "width": "weight" }
    },
    "data": [],
    "network_data": {
      "nodes": [
        { "id": "s_hoffmann_la_roche", "label": "Hoffmann-La Roche", "type": "sponsor", "weight": 4, "nct_ids": ["NCT01026142", "NCT01301729", "NCT01777945", "NCT03101280"] }
      ],
      "edges": [
        {
          "source": "s_hoffmann_la_roche", "target": "d_capecitabine", "weight": 2,
          "citations": [
            { "nct_id": "NCT01026142", "field_path": "protocolSection.sponsorCollaboratorsModule.leadSponsor.name", "excerpt": "Hoffmann-La Roche sponsoring capecitabine", "url": "https://clinicaltrials.gov/study/NCT01026142" }
          ]
        }
      ]
    }
  }
}
```

**Error response** (any 4xx/5xx from `/query`, and a 200 for `no_results`):

```json
{ "error_type": "no_results", "message": "No trials found matching this query.", "suggestion": "Try a broader search or different filters." }
```

`error_type` is one of `validation_error` (422), `parsing_error` (502), `no_results` (**200** —
zero matches is a valid answer, not a failure), `api_error` (502), `unsupported_query` (422),
`internal_error` (500).

Full field-level schemas: `app/schemas/request.py`, `app/schemas/intent.py`, `app/schemas/viz.py`,
`app/schemas/response.py`.

---

## Supported Query Types

| Type | Example | Viz |
|---|---|---|
| **trend** | "How has pembrolizumab trial activity changed per year since 2015?" | `time_series` |
| **distribution** | "How are lung cancer trials distributed across phases?" | `bar_chart` |
| **comparison** | "Compare phases for trials involving Keytruda vs Opdivo." | `grouped_bar_chart` |
| **geographic** | "Which countries have the most recruiting trials for diabetes?" | `bar_chart` |
| **network** | "Show a network of sponsors and drugs for breast cancer trials." | `network_graph` |
| **count** | "How many trials are there for pembrolizumab in total?" | `stat_card` |

Real, live-captured outputs for all six are in `examples/outputs/`.

---

## Design Decisions

**Two-touch agent, not ReAct.** A ReAct-style loop (LLM decides each tool call, observes the
result, decides the next step) would let the model construct CTGov queries and read raw study
data directly — exactly the surface area most prone to hallucination (invented field names,
miscounted studies, fabricated NCT IDs). Instead, the LLM touches the request twice — plan once,
interpret once — and every count, filter, and citation in between is produced by plain,
unit-tested Python. The LLM can misclassify a query; it cannot corrupt a count.

**Composable extractors, not one-off handlers per query phrasing.** See
[Query breadth from composition](#query-breadth-from-composition-not-one-off-handlers) above.

**Count-vs-citation decoupling.** If a query matches 20,000 studies and only 500 are fetched, a
naive implementation's bar chart shows the *sample's* distribution, not the *population's*.
Every fetch passes `countTotal=true` and reports the server's authoritative `total_studies_matched`
separately from `total_studies_fetched` — the chart's own counts come from whatever was actually
fetched, and `meta` is explicit about the gap when the pagination cap (5000 studies) is hit.

**OpenAI Structured Outputs, not free-text JSON parsing.** `response_format: json_schema` with
`strict: true` guarantees a well-formed `Intent` on every successful call — no `json.loads`
failures from a model wrapping its answer in prose or markdown fences.

**Heuristic fallback, not a hard dependency on an LLM.** `app/intent/heuristics.py` is a small
ordered regex table (network > comparison > geographic > trend > count > default distribution)
that classifies from keywords alone when no API key is configured, or when the LLM path fails
twice. It's structurally incapable of extracting `compare_a`/`compare_b` from free text (no NER)
— see [Limitations](#limitations) and [ground-truth overrides](#ground-truth-overrides) below for
how a caller reaches `comparison` anyway.

**Deterministic citations.** `citations/attach.py` dedupes by `nct_id`, sorts by `nct_id`, and
takes the first 3 — the same bucket, built from the same records, always cites the same studies
in the same order.

### Ground-truth overrides

Touch 1 (LLM or heuristic) only ever sees the query string plus whatever entities it can extract.
`pipeline.py`'s `_apply_request_overrides()` re-applies every structured field the *caller*
supplied directly onto the result afterward — for both paths uniformly — so a caller-supplied
`drug_name` always wins over the model's guess, and supplying both `compare_a` and `compare_b`
forces `analysis_type=comparison` outright. This is also the **only** way to reach a comparison
query without an LLM key, since the heuristic path has no NER of its own.

### Live-API corrections found during development

The build plan's assumed CTGov API surface didn't fully match the real API. Found by probing
`https://clinicaltrials.gov/api/v2` directly during Phase 1, before writing the query builder:

- **`/stats/field/values` cannot be scoped.** Any `query.*`/`filter.*` param is rejected outright
  (`400: Invalid prefix in parameter name`) — it can only report unscoped, database-wide stats.
  This confirmed that paginated fetch + `countTotal=true` is the only viable path for scoped
  counting, not an optimization to reach for later.
- **`filter.phase` doesn't exist.** Phase filtering requires `filter.advanced` with Essie query
  syntax: `AREA[Phase](PHASE1 OR PHASE2)`. Verified this ANDs correctly with a date-range clause
  in the same `filter.advanced` string (`... AND AREA[StartDate]RANGE[2015,MAX]`).
  `app/ctgov/query_builder.py`.
- A few `fields` projection names differ from initial assumptions (`Condition` not `Conditions`,
  `CollaboratorName`, `StudyFirstPostDate`) — confirmed against real responses and captured in
  `tests/fixtures/sample_studies.json`.

### Bugs found by actually using the system (not just unit-testing it)

Each of these was invisible to the offline unit suite and only surfaced by driving the real
demo UI or the real LLM against real data — recorded here because *how* they were found is the
useful part:

- **`"N/A"` (no phase recorded) contains a literal `/` character**, so the multi-phase-bucket
  assumption check (`"/" in bucket.key`) was folding "no phase data" studies in with genuinely
  multi-phase studies — a live query reported 31 studies with multiple phases when the real
  number was 7. Found via a live smoke test through the running server.
- **Structured fields weren't actually ground truth.** The LLM path only ever sent the raw query
  string to the model — `request.drug_name`/`condition`/etc. were never merged into the resulting
  `Intent`, contradicting the system prompt's own rule. Also, `QueryRequest` had nowhere to put
  `compare_a`/`compare_b` at all, so a comparison query was unreachable without an LLM key. Found
  while wiring the demo UI's "Keytruda vs Opdivo" example button to real request fields.
- **`query_plan` could describe a stale, pre-downgrade guess.** A heuristic "Compare X vs Y"
  query correctly showed `analysis_type=distribution` (downgraded — see below) and `notes`
  explaining why, but `query_plan` still said `analysis_type='comparison'` — an internally
  inconsistent response. Fixed at the source in `Intent`'s own downgrade validator.
- **Once a real `OPENAI_API_KEY` was configured**, three LLM prompt-quality issues surfaced
  immediately: `notes` came back empty on every query; "since 2015" (no stated end) got an
  invented `end_year`; and "Compare Keytruda vs Opdivo **by phase**" returned
  `compare_type="phase"` instead of `compare_type="drug"` + `dimension="phase"` — the JSON schema
  had bare types with no field-level `description`, so the model had no signal beyond the field
  *name* for what distinguished "what's being compared" from "the breakdown axis." Fixed by
  adding a `description` to every entity field and to `analysis_type` itself
  (`app/intent/prompt.py`).
- **The `.env` file wasn't reaching the LLM clients at all.** `IntentLLMClient`/
  `SummaryLLMClient` read `os.environ.get("OPENAI_API_KEY")` directly, but nothing loads `.env`
  into the process environment — only `pydantic-settings` reads the file itself, into a separate
  `Settings` object. Fixed by routing both clients through `get_settings()`. This in turn exposed
  a **second** issue: `pydantic-settings` reads the literal `.env` file regardless of
  `os.environ`, so `monkeypatch.delenv("OPENAI_API_KEY")` silently stopped simulating "no key"
  the moment a real key existed on disk — several tests would have started making real network
  calls. Fixed with a `no_openai_key` fixture (`tests/conftest.py`) that forces
  `Settings(_env_file=None, openai_api_key=None)` regardless of what's actually on disk.
- **A geographic query was classified as `count`.** "Which countries have the most recruiting
  trials for diabetes?" (the assignment appendix's own geographic example) came back as
  `analysis_type=count`, dropping the country breakdown to a single total — the model latched
  onto "how many"/"most" with no schema signal to tell `geographic` (breakdown by country) apart
  from `count` (a single number, no breakdown). Fixed with a description on the `analysis_type`
  field explicitly distinguishing the two, naming this exact query as the example.

---

## AI Tools & Integrity

- **Claude Code** (Sonnet 5) implemented the system end-to-end, phase by phase, against a
  pre-written build plan: schemas and extractors first, then aggregation, then the LLM touches,
  then the pipeline wiring, then the UI.
- **OpenAI** (`gpt-4o-mini`) powers both agent touches: intent classification (structured
  outputs) and result interpretation (free text).
- **Validation:** every module has an offline unit test suite (294 tests) exercised before each
  commit; live behavior (the real CTGov API, and — once configured — the real OpenAI API) was
  spot-checked manually at multiple points, which is how most of the bugs listed above were
  actually found. `tests/fixtures/sample_studies.json` is a real captured API response, not
  synthetic data.
- **Designed deliberately:** the pipeline architecture (two-touch agent, composable extractors,
  count-vs-citation decoupling, the dispatch table's deliberate non-uniformity for `comparison`/
  `count`), the schema shapes, and every fix described above were reasoned through explicitly,
  not generated wholesale and accepted as-is.
- Git history is unsquashed on purpose — one PR per build-plan phase, atomic commits within each,
  bug-fix commits kept separate from feature commits — as a record of how the system was actually
  built and where it went wrong along the way.

---

## Limitations

- **Light name normalization, not real entity resolution.** Drug/sponsor names are lowercased and
  stripped of dosage suffixes (`"Drug X 200mg"` → `"drug x"`) before becoming a network node —
  enough to merge obvious variants, not enough to resolve genuine aliases or misspellings.
- **The heuristic fallback has no NER.** Without an LLM, entities come only from whatever
  structured request fields the caller supplies — a "compare A vs B" query typed as free text
  alone can never populate `compare_a`/`compare_b` and downgrades to `distribution` (transparently
  — see `notes`/`query_plan` in the response).
- **Pagination caps at 5000 studies** (`max_studies`, itself capped at 5000). Very broad queries
  report `total_studies_matched` (server-authoritative) separately from `total_studies_fetched`,
  but a chart's own bucket counts only reflect what was actually fetched.
- **No conversational follow-up.** Every request is independent; there's no session or
  turn-taking to refine a previous chart.
- **No caching.** Identical queries re-fetch from ClinicalTrials.gov every time.
- **Network layout is a plain circle, not a force-directed graph.** Plotly has no native network
  chart type; nodes are placed on a ring and sized by weight, which is readable but not as clear
  as a real force-directed layout for large graphs (labels are limited to the top 20 nodes by
  weight to stay legible).

## Future Improvements

- Redis (or even in-process) caching for repeated queries.
- A real force-directed network layout (client-side, e.g. d3-force) instead of the circular
  placeholder.
- Vega-Lite output as an alternative to the current bespoke `VisualizationSpec`, for direct
  embedding in tools that already speak Vega-Lite.
- Conversational follow-up ("now break that down by phase") via a lightweight session store.
- Streaming responses for very large fetches.
- OpenTelemetry tracing across the two LLM touches and the CTGov fetch.
- CSV export of the underlying bucket data alongside the chart.
