from datetime import UTC, datetime

from app.analysis.dispatch import ANALYSIS_DISPATCH, run_comparison, run_count
from app.ctgov.client import CTGovClient
from app.ctgov.record_extractor import extract_record
from app.exceptions import NoResultsError
from app.intent.llm_client import IntentLLMClient
from app.intent.parser import parse_intent
from app.schemas.intent import AnalysisType, Entities, Intent
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
    intent = parsed.intent

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
