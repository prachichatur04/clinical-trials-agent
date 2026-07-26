"""Tests that hit the real ClinicalTrials.gov API (and, for the intent/
pipeline tests below, the real OpenAI API when OPENAI_API_KEY is set).
Excluded by default (see pyproject.toml's `addopts = "-m 'not live'"`);
run explicitly with:

    pytest tests/integration -m live
"""

import pytest

from app.analysis.dispatch import run_distribution
from app.ctgov.client import CTGovClient
from app.ctgov.record_extractor import extract_record
from app.exceptions import NoResultsError
from app.intent.llm_client import IntentLLMClient
from app.pipeline import run_pipeline
from app.schemas.intent import AnalysisType, Confidence, Entities, Intent, VizType
from app.schemas.request import QueryRequest

pytestmark = pytest.mark.live


# --- CTGovClient against the real API ---------------------------------


async def test_paginate_returns_plausible_data_for_a_known_drug():
    client = CTGovClient()
    try:
        studies, total_matched = await client.paginate(max_studies=20, drug_name="Pembrolizumab")
    finally:
        await client.aclose()

    assert total_matched > 1000  # Pembrolizumab has thousands of registered trials
    assert len(studies) == 20
    assert all("protocolSection" in s for s in studies)


async def test_phase_filter_narrows_results():
    client = CTGovClient()
    try:
        _, total_unfiltered = await client.paginate(max_studies=1, drug_name="Pembrolizumab")
        _, total_phase1 = await client.paginate(max_studies=1, drug_name="Pembrolizumab", phases="PHASE1")
    finally:
        await client.aclose()

    assert 0 < total_phase1 < total_unfiltered


async def test_paginate_follows_cursor_across_multiple_real_pages():
    # page_size=500 forces 3 real page fetches (following nextPageToken)
    # to reach 1500 studies -- the mock-transport unit tests exercise the
    # cursor-following logic itself, but never against the real API's
    # actual pageToken values/shape.
    client = CTGovClient()
    try:
        studies, total_matched = await client.paginate(max_studies=1500, drug_name="Pembrolizumab", page_size=500)
    finally:
        await client.aclose()

    assert total_matched > 1500
    assert len(studies) == 1500
    nct_ids = [s["protocolSection"]["identificationModule"]["nctId"] for s in studies]
    assert len(nct_ids) == len(set(nct_ids))  # no duplicate studies fetched across pages


# --- citations against real data ---------------------------------------


async def test_citations_never_dangle_outside_the_fetched_record_set():
    # Every nct_id a bucket cites must be one of the studies actually
    # fetched -- a citation pointing to a study we never retrieved would
    # be a serious correctness bug (deep citations are supposed to be
    # traceable to real, in-hand data, not invented).
    client = CTGovClient()
    try:
        raw_studies, _ = await client.paginate(max_studies=50, condition="lung cancer")
    finally:
        await client.aclose()

    records = [extract_record(s) for s in raw_studies]
    fetched_nct_ids = {r.nct_id for r in records}

    intent = Intent(
        analysis_type=AnalysisType.DISTRIBUTION,
        entities=Entities(dimension="phase"),
        suggested_viz=VizType.BAR_CHART,
        query_plan="plan",
        notes="notes",
        confidence=Confidence.HIGH,
    )
    dispatch_result = run_distribution(records, intent)

    cited_nct_ids = {
        sample.nct_id for bucket in dispatch_result.aggregated.buckets for sample in bucket.samples
    }
    assert cited_nct_ids <= fetched_nct_ids


# --- Touch 1 (real LLM) golden-query classification ------------------------
#
# test_heuristics.py already covers these same golden queries against the
# heuristic fallback. These mirror them against the real IntentLLMClient --
# added after two real prompt bugs (geographic misclassified as count;
# compare_type/dimension conflated) shipped to production and were only
# caught by manual spot-checking, not by any test. These exist so a future
# prompt edit can't silently reintroduce either failure mode.


async def test_real_llm_trend_classification():
    client = IntentLLMClient()
    intent = await client.classify("How has the number of trials for pembrolizumab changed per year since 2015?")

    assert intent.analysis_type == AnalysisType.TREND
    assert intent.entities.start_year == 2015
    assert intent.entities.end_year is None  # must not invent an upper bound for an open range
    assert intent.notes  # must never be empty


async def test_real_llm_distribution_classification():
    client = IntentLLMClient()
    intent = await client.classify("How are lung cancer trials distributed across phases?")

    assert intent.analysis_type == AnalysisType.DISTRIBUTION
    assert intent.entities.dimension == "phase"


async def test_real_llm_comparison_classification():
    client = IntentLLMClient()
    intent = await client.classify("Compare phases for trials involving Keytruda vs Opdivo.")

    assert intent.analysis_type == AnalysisType.COMPARISON
    assert intent.entities.compare_a and intent.entities.compare_b
    # Regression: compare_type is WHAT is compared (drug), not the
    # breakdown axis (phase) -- these two fields were previously conflated.
    assert intent.entities.compare_type == "drug"
    assert intent.entities.dimension == "phase"


async def test_real_llm_geographic_classification_not_count():
    client = IntentLLMClient()
    intent = await client.classify("Which countries have the most recruiting trials for diabetes?")

    # Regression: this exact query was classified as COUNT before
    # analysis_type got a field description distinguishing it from geographic.
    assert intent.analysis_type == AnalysisType.GEOGRAPHIC
    assert intent.entities.dimension == "country"


async def test_real_llm_network_classification():
    client = IntentLLMClient()
    intent = await client.classify("Show a network of sponsors and drugs for breast cancer trials.")

    assert intent.analysis_type == AnalysisType.NETWORK


async def test_real_llm_drug_cooccurrence_classification():
    client = IntentLLMClient()
    intent = await client.classify("Which drugs frequently co-occur in combination studies?")

    assert intent.analysis_type == AnalysisType.NETWORK
    assert intent.entities.dimension == "drug_cooccurrence"


async def test_real_llm_count_classification():
    client = IntentLLMClient()
    intent = await client.classify("How many trials are there for pembrolizumab in total?")

    assert intent.analysis_type == AnalysisType.COUNT


# --- full pipeline, every analysis type, against the real API ------------


async def test_full_pipeline_end_to_end_against_real_api():
    request = QueryRequest(
        query="How are lung cancer trials distributed across phases?",
        condition="lung cancer",
        max_studies=30,
    )

    response = await run_pipeline(request)

    assert response.meta.total_studies_matched > 0
    assert response.visualization.data
    all_cited_ids = {c.nct_id for dp in response.visualization.data for c in dp.citations}
    assert all(nct_id.startswith("NCT") for nct_id in all_cited_ids)


async def test_full_pipeline_with_summary_produces_real_touch_2_output():
    # The one other live pipeline test never sets include_summary=True, so
    # Touch 2 (generate_summary) was never exercised by the permanent
    # test suite -- only by ad-hoc scripts during development.
    request = QueryRequest(
        query="How are lung cancer trials distributed across phases?",
        condition="lung cancer",
        max_studies=30,
        include_summary=True,
    )

    response = await run_pipeline(request)

    assert response.summary is not None
    assert len(response.summary) > 20


async def test_comparison_end_to_end_against_real_api():
    request = QueryRequest(
        query="Compare phases for trials involving Keytruda vs Opdivo.",
        compare_a="Keytruda",
        compare_b="Opdivo",
        compare_type="drug",
        dimension="phase",
        max_studies=100,
    )

    response = await run_pipeline(request)

    assert response.visualization.type == VizType.GROUPED_BAR_CHART
    assert response.meta.analysis_type == AnalysisType.COMPARISON
    series_labels = {dp.series for dp in response.visualization.data}
    assert series_labels == {"Keytruda", "Opdivo"}


async def test_count_end_to_end_against_real_api():
    request = QueryRequest(query="How many trials are there for pembrolizumab in total?", drug_name="pembrolizumab")

    response = await run_pipeline(request)

    assert response.visualization.type == VizType.STAT_CARD
    assert response.visualization.data[0].y > 1000


async def test_drug_cooccurrence_network_end_to_end_against_real_api():
    request = QueryRequest(
        query="Which drugs frequently co-occur in combination studies for breast cancer?",
        condition="breast cancer",
        dimension="drug_cooccurrence",  # ground-truth override, decoupled from Touch 1's own guess
        max_studies=150,
    )

    response = await run_pipeline(request)

    assert response.visualization.type == VizType.NETWORK_GRAPH
    network_data = response.visualization.network_data
    # A drug<->drug co-occurrence network only ever has "drug" nodes,
    # never "sponsor" -- unlike the default bipartite mode.
    node_types = {n.type for n in network_data.nodes}
    assert node_types <= {"drug"}


async def test_pipeline_raises_no_results_for_a_genuinely_nonexistent_drug():
    # A count query treats zero matches as a valid answer (see
    # test_count_of_zero_is_a_valid_answer_not_an_error in test_pipeline.py),
    # so this must be phrased as a non-count query to actually exercise
    # NoResultsError against a real "nothing matched" response.
    request = QueryRequest(
        query="How are trials for this fictional drug distributed across phases?",
        drug_name="Zzyzxaquinolinumab12345NonexistentDrug",
    )

    with pytest.raises(NoResultsError):
        await run_pipeline(request)
