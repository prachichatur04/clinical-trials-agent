import httpx
import pytest

from app.ctgov.client import CTGovClient
from app.exceptions import NoResultsError, UnsupportedQueryError
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


def _stub_summary(text: str | None = "Summary text.", raises: Exception | None = None):
    class _Stub:
        def __init__(self):
            self.calls = 0

        async def summarize(self, prompt):
            self.calls += 1
            if raises is not None:
                raise raises
            return text

    return _Stub()


def _intent(analysis_type=AnalysisType.DISTRIBUTION, entities=None, **overrides) -> Intent:
    # Default entities carry a scoping field (condition) so run_pipeline's
    # _has_any_scoping_entity check passes -- tests that care about specific
    # entities pass their own `entities=` and override this.
    defaults = {
        "analysis_type": analysis_type,
        "entities": entities or Entities(condition="lung cancer"),
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


async def test_no_llm_falls_back_to_heuristic_and_still_returns_valid_response(no_openai_key):
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    request = QueryRequest(query="How are trials distributed across phases?", condition="lung cancer")

    response = await run_pipeline(request, ctgov_client=client, llm_client=None)

    assert response.meta.intent_source == IntentSource.HEURISTIC_FALLBACK
    assert response.visualization.type == VizType.BAR_CHART


# --- Touch 2: include_summary wiring ----------------------------------------


async def test_include_summary_true_calls_generator_and_sets_summary():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    summary_stub = _stub_summary("Phase 1 dominates.")
    request = QueryRequest(query="How are trials distributed across phases?", include_summary=True)
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION))

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm, summary_llm_client=summary_stub)

    assert response.summary == "Phase 1 dominates."
    assert summary_stub.calls == 1


async def test_include_summary_false_never_calls_generator():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    summary_stub = _stub_summary("should never be used")
    request = QueryRequest(query="How are trials distributed across phases?", include_summary=False)
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION))

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm, summary_llm_client=summary_stub)

    assert response.summary is None
    assert summary_stub.calls == 0


async def test_summary_generation_failure_does_not_fail_the_request():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    summary_stub = _stub_summary(raises=RuntimeError("LLM unavailable"))
    request = QueryRequest(query="How are trials distributed across phases?", include_summary=True)
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION))

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm, summary_llm_client=summary_stub)

    assert response.summary is None
    assert response.visualization.type == VizType.BAR_CHART


# --- request field overrides (ground truth beats Touch 1's guess) ----------


async def test_request_compare_a_and_b_force_comparison_without_an_llm(no_openai_key):
    # The heuristic path alone can never produce analysis_type=comparison
    # (no NER), so this is the only way to reach it without an LLM key --
    # explicit structured fields are stronger evidence than keyword regex.
    captured_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_calls.append(request.url.params.get("query.intr"))
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    request = QueryRequest(
        query="Compare Keytruda vs Opdivo by phase.",
        compare_a="Keytruda",
        compare_b="Opdivo",
        compare_type="drug",
        dimension="phase",
    )

    response = await run_pipeline(request, ctgov_client=client, llm_client=None)

    assert response.visualization.type == VizType.GROUPED_BAR_CHART
    assert response.meta.analysis_type == AnalysisType.COMPARISON
    assert set(captured_calls) == {"Keytruda", "Opdivo"}
    # query_plan/notes should reflect the forced override, not silently
    # keep describing whatever Touch 1 originally guessed (the same class
    # of inconsistency fixed in schemas/intent.py's downgrade path).
    assert "overridden to comparison" in response.meta.query_plan


async def test_request_drug_name_overrides_llm_guessed_entity():
    captured_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_calls.append(request.url.params.get("query.intr"))
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    # LLM guessed a different (or no) drug_name from the text alone.
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION, entities=Entities(drug_name="wrong-guess")))
    request = QueryRequest(query="How are trials distributed?", drug_name="Pembrolizumab")

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert captured_calls == ["Pembrolizumab"]
    assert response.meta.filters_applied["drug_name"] == "Pembrolizumab"


async def test_request_fields_left_unset_do_not_clobber_llm_extracted_entities():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION, entities=Entities(drug_name="from-llm")))
    request = QueryRequest(query="How are trials distributed?")  # no drug_name supplied

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert response.meta.filters_applied["drug_name"] == "from-llm"


# --- unsupported (unscoped) queries -----------------------------------------


async def test_query_with_no_recognizable_entity_raises_unsupported_query():
    # Reported behavior: a query like a person's name extracts no drug/
    # condition/sponsor/etc, so the fetch was completely unscoped and
    # silently returned stats for the entire ClinicalTrials.gov database
    # (596k+ studies) instead of failing. Real name from the bug report:
    # a random person's name gave back "total trials: 595,630".
    llm = _stub_llm(_intent(AnalysisType.COUNT, entities=Entities()))
    request = QueryRequest(query="sarveshsawant is my friend I think")

    with pytest.raises(UnsupportedQueryError):
        await run_pipeline(request, llm_client=llm)


async def test_dimension_alone_does_not_count_as_a_scoping_entity():
    # dimension only controls how already-fetched records get bucketed --
    # it never becomes a query.*/filter.* param, so it doesn't scope the
    # fetch at all. A dimension with nothing else set is exactly as
    # unscoped as no entities at all.
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION, entities=Entities(dimension="phase")))
    request = QueryRequest(query="How are trials distributed across phases?")

    with pytest.raises(UnsupportedQueryError):
        await run_pipeline(request, llm_client=llm)


async def test_compare_a_and_b_alone_count_as_scoping_even_with_no_other_entity():
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    llm = _stub_llm(
        _intent(AnalysisType.COMPARISON, entities=Entities(compare_a="Keytruda", compare_b="Opdivo"))
    )
    request = QueryRequest(query="Compare Keytruda vs Opdivo.")

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert response.visualization.type == VizType.GROUPED_BAR_CHART


async def test_caller_supplied_structured_field_alone_is_enough_even_with_empty_intent():
    # A caller who supplies e.g. drug_name directly shouldn't be rejected
    # just because Touch 1 itself extracted nothing from the query text --
    # the ground-truth override (_apply_request_overrides) runs before this
    # check, so the request's own fields count too.
    def handler(request: httpx.Request) -> httpx.Response:
        return _page_response([_study("NCT1")], total_count=1)

    client = _mock_client(handler)
    llm = _stub_llm(_intent(AnalysisType.DISTRIBUTION, entities=Entities()))
    request = QueryRequest(query="How are trials distributed?", drug_name="Pembrolizumab")

    response = await run_pipeline(request, ctgov_client=client, llm_client=llm)

    assert response.meta.filters_applied["drug_name"] == "Pembrolizumab"
