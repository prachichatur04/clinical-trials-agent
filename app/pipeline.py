from datetime import UTC, datetime

from app.analysis.dispatch import ANALYSIS_DISPATCH, run_comparison, run_count
from app.ctgov.client import CTGovClient
from app.ctgov.record_extractor import extract_record
from app.exceptions import NoResultsError, UnsupportedQueryError
from app.intent.llm_client import IntentLLMClient
from app.intent.parser import parse_intent
from app.schemas.intent import AnalysisType, Confidence, Entities, Intent
from app.schemas.internal import DispatchResult, TrialRecord
from app.schemas.request import QueryRequest
from app.schemas.response import Meta, QueryResponse
from app.services.summary_generator import SummaryLLMClient, generate_summary
from app.viz.builder import build_visualization

CTGOV_STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
_COMPARE_TYPE_TO_FIELD = {"drug": "drug_name", "condition": "condition", "sponsor": "sponsor"}


async def run_pipeline(
    request: QueryRequest,
    *,
    ctgov_client: CTGovClient | None = None,
    llm_client: IntentLLMClient | None = None,
    summary_llm_client: SummaryLLMClient | None = None,
) -> QueryResponse:
    """The thin orchestrator: Touch 1 -> fetch -> extract -> aggregate ->
    build viz -> Touch 2 (if requested) -> assemble response.
    """
    generated_at = datetime.now(UTC)
    parsed = await parse_intent(request, llm_client=llm_client)
    intent = _apply_request_overrides(parsed.intent, request)

    if not _has_any_scoping_entity(intent.entities):
        raise UnsupportedQueryError(
            "Could not identify a drug, condition, sponsor, or other clinical-trial filter in this query.",
            suggestion="Try including a specific drug name, condition, sponsor, or date range.",
        )

    client = ctgov_client or CTGovClient()
    try:
        if intent.analysis_type == AnalysisType.COMPARISON:
            total_matched, total_fetched, unique_study_count, dispatch_result = await _fetch_comparison(
                client, request, intent
            )
        elif intent.analysis_type == AnalysisType.COUNT:
            total_matched, total_fetched, unique_study_count, dispatch_result = await _fetch_count(client, intent)
        else:
            total_matched, total_fetched, unique_study_count, dispatch_result = await _fetch_single(
                client, request, intent
            )
    finally:
        if ctgov_client is None:
            await client.aclose()

    # A count of zero is itself a valid answer to "how many trials..." --
    # only the other analysis types treat zero matches as a no-results case.
    if intent.analysis_type != AnalysisType.COUNT and total_matched == 0:
        raise NoResultsError("No trials found matching this query.", suggestion="Try a broader search or different filters.")

    visualization = build_visualization(
        dispatch_result.viz_type,
        dispatch_result.title,
        dispatch_result.aggregated,
        include_citations=request.include_citations,
    )

    summary = None
    if request.include_summary:
        summary = await generate_summary(
            request.query,
            intent.analysis_type,
            dispatch_result.aggregated,
            total_matched,
            total_fetched,
            llm_client=summary_llm_client,
        )

    meta = Meta(
        query_interpretation=intent.notes,
        query_plan=intent.query_plan,
        analysis_type=intent.analysis_type,
        filters_applied=_filters_applied(intent),
        assumptions=dispatch_result.aggregated.assumptions,
        total_studies_matched=total_matched,
        total_studies_fetched=total_fetched,
        unique_study_count=unique_study_count,
        source=CTGOV_STUDIES_URL,
        generated_at=generated_at,
        intent_source=parsed.source,
    )
    return QueryResponse(visualization=visualization, summary=summary, meta=meta)


_OVERRIDABLE_ENTITY_FIELDS = (
    "drug_name",
    "condition",
    "trial_phase",
    "sponsor",
    "country",
    "status",
    "start_year",
    "end_year",
    "compare_a",
    "compare_b",
    "compare_type",
    "dimension",
)


def _apply_request_overrides(intent: Intent, request: QueryRequest) -> Intent:
    """Structured fields the caller supplied directly are ground truth --
    they override whatever Touch 1 (LLM or heuristic) extracted from the
    query text, per the system prompt's own rule. This matters most for the
    LLM path, which only ever sees the raw query string, never the
    request's other fields.

    If the caller supplies both compare_a and compare_b, that's stronger
    evidence than any keyword match, so it forces analysis_type=comparison
    outright -- this is also the only way to reach a comparison analysis
    without an LLM, since the heuristic path has no NER to extract
    compare_a/compare_b from free text on its own.
    """
    for field in _OVERRIDABLE_ENTITY_FIELDS:
        value = getattr(request, field, None)
        if value is not None:
            setattr(intent.entities, field, value)

    if intent.entities.compare_a and intent.entities.compare_b and intent.analysis_type != AnalysisType.COMPARISON:
        override_note = (
            f"analysis_type overridden to comparison: caller supplied both "
            f"compare_a ({intent.entities.compare_a}) and compare_b ({intent.entities.compare_b})"
        )
        intent.analysis_type = AnalysisType.COMPARISON
        intent.confidence = Confidence.HIGH
        intent.notes = f"{intent.notes} ({override_note})".strip()
        intent.query_plan = f"{intent.query_plan} ({override_note})".strip()

    return intent


_SCOPING_ENTITY_FIELDS = (
    "drug_name",
    "condition",
    "sponsor",
    "country",
    "status",
    "trial_phase",
    "start_year",
    "end_year",
    "compare_a",
    "compare_b",
)


def _has_any_scoping_entity(entities: Entities) -> bool:
    """True if the query (or the caller's own structured fields) identified
    at least one thing to actually filter by. Without this check, a query
    with no recognizable entity at all -- a name, a typo, gibberish --
    still produces a well-formed-looking response: no query.*/filter.*
    params get built, so the fetch is completely unscoped and silently
    returns stats for the entire ClinicalTrials.gov database instead of
    failing loudly. `dimension` deliberately doesn't count here: it only
    controls how already-fetched records get bucketed, not what gets
    fetched, so setting it alone doesn't scope anything.
    """
    return any(getattr(entities, field) is not None for field in _SCOPING_ENTITY_FIELDS)


def _entity_fetch_kwargs(entities: Entities) -> dict:
    return {
        "drug_name": entities.drug_name,
        "condition": entities.condition,
        "sponsor": entities.sponsor,
        "country": entities.country,
        "status": entities.status,
        "phases": entities.trial_phase,
        "start_year": entities.start_year,
        "end_year": entities.end_year,
    }


async def _fetch_single(
    client: CTGovClient, request: QueryRequest, intent: Intent
) -> tuple[int, int, int, DispatchResult]:
    raw_studies, total_matched = await client.paginate(
        max_studies=request.max_studies, **_entity_fetch_kwargs(intent.entities)
    )
    records = [extract_record(study) for study in raw_studies]
    dispatch_fn = ANALYSIS_DISPATCH[intent.analysis_type.value]
    dispatch_result = dispatch_fn(records, intent)
    return total_matched, len(records), _unique_count(records), dispatch_result


async def _fetch_count(client: CTGovClient, intent: Intent) -> tuple[int, int, int, DispatchResult]:
    # No extraction needed for a bare count -- one lightweight page is enough
    # to read the server's authoritative totalCount.
    page = await client.search(page_size=1, **_entity_fetch_kwargs(intent.entities))
    total_matched = page.get("totalCount", 0)
    return total_matched, 0, 0, run_count(intent, total_matched)


async def _fetch_comparison(
    client: CTGovClient, request: QueryRequest, intent: Intent
) -> tuple[int, int, int, DispatchResult]:
    field = _COMPARE_TYPE_TO_FIELD.get(intent.entities.compare_type or "drug", "drug_name")
    half_budget = max(request.max_studies // 2, 1)

    kwargs_a = _entity_fetch_kwargs(intent.entities) | {field: intent.entities.compare_a}
    kwargs_b = _entity_fetch_kwargs(intent.entities) | {field: intent.entities.compare_b}

    raw_a, total_a = await client.paginate(max_studies=half_budget, **kwargs_a)
    raw_b, total_b = await client.paginate(max_studies=half_budget, **kwargs_b)

    records_a = [extract_record(study) for study in raw_a]
    records_b = [extract_record(study) for study in raw_b]

    dispatch_result = run_comparison(records_a, records_b, intent)
    total_fetched = len(records_a) + len(records_b)
    unique_study_count = len({r.nct_id for r in records_a} | {r.nct_id for r in records_b})
    return total_a + total_b, total_fetched, unique_study_count, dispatch_result


def _unique_count(records: list[TrialRecord]) -> int:
    return len({r.nct_id for r in records})


def _filters_applied(intent: Intent) -> dict:
    return {key: value for key, value in intent.entities.model_dump().items() if value is not None}
