import httpx
import pytest

from app.ctgov.client import CTGovClient
from app.exceptions import NoResultsError
from app.pipeline import run_pipeline
from app.schemas.intent import AnalysisType, Confidence, Entities, Intent, VizType
from app.schemas.request import QueryRequest
from app.schemas.response import IntentSource
from app.utils.rate_limiter import RateLimiter


def _study(nct_id: str, **overrides) -> dict:
    protocol_section = {
        "identificationModule": {"nctId": nct_id, "briefTitle": f"Study {nct_id}"},
        "statusModule": {"overallStatus": "RECRUITING", "startDateStruct": {"date": "2020-01-01"}},
        "designModule": {"phases": ["PHASE1"]},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Pfizer", "class": "INDUSTRY"}},
    }
    protocol_section.update(overrides)
    return {"protocolSection": protocol_section}


def _page_response(studies, total_count):
    return httpx.Response(200, json={"studies": studies, "totalCount": total_count})


def _mock_client(handler) -> CTGovClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return CTGovClient(http_client=http_client, rate_limiter=RateLimiter(min_interval=0))


def _stub_llm(intent: Intent):
    class _Stub:
        async def classify(self, query, extra_context=None):
            return intent

    return _Stub()


def _intent(analysis_type=AnalysisType.DISTRIBUTION, entities=None, **overrides) -> Intent:
    defaults = {
        "analysis_type": analysis_type,
        "entities": entities or Entities(),
        "suggested_viz": VizType.BAR_CHART,
        "query_plan": "plan",
        "notes": "interpretation",
        "confidence": Confidence.HIGH,
    }
    defaults.update(overrides)
    return Intent(**defaults)


# --- single-fetch path (trend/distribution/geographic/network) -------------


async def test_distribution_end_to_end():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1"), _study("NCT2")], total_count=2)

    client = _mock_client(handler)
    request = QueryRequest(query="How are trials distributed across phases?")
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION))

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert response.visualization.type == VizType.BAR_CHART
    assert response.visualization.data[0].x == "Phase 1"
    assert response.visualization.data[0].y == 2
    assert response.meta.total_studies_matched == 2
    assert response.meta.total_studies_fetched == 2
    assert response.meta.unique_study_count == 2
    assert response.meta.intent_source == IntentSource.LLM
    assert response.meta.query_interpretation == "interpretation"
    assert response.summary is None  # Touch 2 lands in Phase 6


async def test_filters_applied_omits_unset_entities():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    request = QueryRequest(query="How are trials distributed?")
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION, entities=Entities(drug_name="Pembrolizumab")))

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert response.meta.filters_applied == {"drug_name": "Pembrolizumab"}


async def test_zero_matches_raises_no_results_for_non_count_types():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([], total_count=0)

    client = _mock_client(handler)
    request = QueryRequest(query="How are trials distributed?")
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION))

    with pytest.raises(NoResultsError):
        await run_pipeline(request, ctgov_client=client, llm_client=llm)


async def test_trend_reports_date_assumptions_in_meta():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response(
            [_study("NCT1", statusModule={"overallStatus": "RECRUITING"})],  # no startDateStruct
            total_count=1,
        )

    client = _mock_client(handler)
    request = QueryRequest(query="How has this changed since 2015?")
    llm = _stub_llm(_intent(AnalysisType.TREND))

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert response.visualization.data[0].x == "unknown"
    assert "unparseable start date" in response.meta.assumptions[0]


# --- count path -----------------------------------------------------------


async def test_count_uses_single_lightweight_page_not_full_pagination():
    captured_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_calls.append(request.url.params.get("pageSize"))
        return _page_response([_study("NCT1")], total_count=8123)

    client = _mock_client(handler)
    request = QueryRequest(query="How many trials are there in total?")
    llm = _stub_llm(_intent(AnalysisType.COUNT))

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert len(captured_calls) == 1
    assert captured_calls[0] == "1"
    assert response.visualization.type == VizType.STAT_CARD
    assert response.visualization.data[0].y == 8123
    assert response.meta.total_studies_matched == 8123
    assert response.meta.total_studies_fetched == 0


async def test_count_of_zero_is_a_valid_answer_not_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([], total_count=0)

    client = _mock_client(handler)
    request = QueryRequest(query="How many trials exist for this obscure drug?")
    llm = _stub_llm(_intent(AnalysisType.COUNT))

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert response.visualization.data[0].y == 0


# --- comparison path -------------------------------------------------------


async def test_comparison_fetches_both_sides_with_compare_type_field():
    captured_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_calls.append(dict(request.url.params))
        drug = request.url.params.get("query.intr")
        if drug == "Keytruda":
            return _page_response([_study("NCT1", designModule={"phases": ["PHASE1"]})], total_count=1)
        return _page_response([_study("NCT2", designModule={"phases": ["PHASE2"]})], total_count=1)

    client = _mock_client(handler)
    request = QueryRequest(query="Compare Keytruda vs Opdivo by phase.", max_studies=100)
    llm = _stub_llm(
        _intent(
            AnalysisType.COMPARISON,
            entities=Entities(compare_a="Keytruda", compare_b="Opdivo", compare_type="drug", dimension="phase"),
        )
    )

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert len(captured_calls) == 2
    assert {c.get("query.intr") for c in captured_calls} == {"Keytruda", "Opdivo"}
    assert response.visualization.type == VizType.GROUPED_BAR_CHART
    assert response.meta.total_studies_matched == 2
    assert response.meta.total_studies_fetched == 2


async def test_comparison_uses_half_max_studies_budget_per_side():
    captured_page_sizes = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_page_sizes.append(int(request.url.params["pageSize"]))
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    request = QueryRequest(query="Compare A vs B.", max_studies=100)
    llm = _stub_llm(
        _intent(AnalysisType.COMPARISON, entities=Entities(compare_a="A", compare_b="B", compare_type="drug"))
    )

    await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert all(size == 50 for size in captured_page_sizes)


async def test_comparison_both_sides_empty_raises_no_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([], total_count=0)

    client = _mock_client(handler)
    request = QueryRequest(query="Compare A vs B.")
    llm = _stub_llm(
        _intent(AnalysisType.COMPARISON, entities=Entities(compare_a="A", compare_b="B", compare_type="drug"))
    )

    with pytest.raises(NoResultsError):
        await run_pipeline(request, ctgov_client=client, llm_client=llm)


async def test_comparison_unique_study_count_dedupes_overlap_across_sides():
    def handler(request: httpx.Request) -> httpx.Response:
        # Same NCT ID shows up on both sides (e.g. a combo study).
        return _page_response([_study("NCT-SHARED")], total_count=1)

    client = _mock_client(handler)
    request = QueryRequest(query="Compare A vs B.")
    llm = _stub_llm(
        _intent(AnalysisType.COMPARISON, entities=Entities(compare_a="A", compare_b="B", compare_type="drug"))
    )

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert response.meta.total_studies_fetched == 2  # counted once per side
    assert response.meta.unique_study_count == 1  # but only one distinct study


# --- heuristic fallback path (no LLM) ---------------------------------------


async def test_no_llm_falls_back_to_heuristic_and_still_returns_valid_response(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    request = QueryRequest(query="How are trials distributed across phases?")

    response = await run_pipeline(request, ctgov_client=client, llm_client=None)

    assert response.meta.intent_source == IntentSource.HEURISTIC_FALLBACK
    assert response.visualization.type == VizType.BAR_CHART
