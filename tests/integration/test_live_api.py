"""Tests that hit the real ClinicalTrials.gov API. Excluded by default
(see pyproject.toml's `addopts = "-m 'not live'"`); run explicitly with:

    pytest tests/integration -m live
"""

import pytest

from app.analysis.dispatch import run_distribution
from app.ctgov.client import CTGovClient
from app.ctgov.record_extractor import extract_record
from app.pipeline import run_pipeline
from app.schemas.intent import AnalysisType, Confidence, Entities, Intent, VizType
from app.schemas.request import QueryRequest

pytestmark = pytest.mark.live


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
