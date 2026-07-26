import pytest

from app.intent.heuristics import classify_heuristically
from app.schemas.intent import AnalysisType, Confidence, VizType
from app.schemas.request import QueryRequest

# Golden queries lifted from the assignment appendix, one per stated example.
#
# The two "compare" queries expect DISTRIBUTION, not COMPARISON: the
# heuristic path has no NER, so it can never populate compare_a/compare_b
# from free text. Intent's own model validator (schemas/intent.py) correctly
# downgrades any comparison intent that's missing both entities -- so every
# heuristic "this looks comparative" match self-corrects to distribution.
# That's the schema's safety net doing its job, not a bug here.
GOLDEN_QUERIES = [
    ("How has the number of trials for pembrolizumab changed per year since 2015?", AnalysisType.TREND),
    ("How many trials started each year for lung cancer?", AnalysisType.TREND),
    ("How are lung cancer trials distributed across phases?", AnalysisType.DISTRIBUTION),
    ("What are the most common intervention types for pembrolizumab trials?", AnalysisType.DISTRIBUTION),
    ("Compare phases for trials involving Drug A vs Drug B.", AnalysisType.DISTRIBUTION),
    ("Compare sponsor categories across two conditions.", AnalysisType.DISTRIBUTION),
    ("Which countries have the most recruiting trials for diabetes?", AnalysisType.GEOGRAPHIC),
    ("Show a network of sponsors and drugs for breast cancer trials.", AnalysisType.NETWORK),
    ("Which drugs frequently co-occur in combination studies (drug-drug network)?", AnalysisType.NETWORK),
]


@pytest.mark.parametrize("query,expected_type", GOLDEN_QUERIES)
def test_golden_appendix_queries_classify_correctly(query, expected_type):
    intent = classify_heuristically(QueryRequest(query=query))
    assert intent.analysis_type == expected_type


def test_unmatched_query_defaults_to_distribution():
    intent = classify_heuristically(QueryRequest(query="Tell me about trial X in general terms."))
    assert intent.analysis_type == AnalysisType.DISTRIBUTION


def test_heuristic_always_reports_low_confidence():
    intent = classify_heuristically(QueryRequest(query="How are trials distributed across phases?"))
    assert intent.confidence == Confidence.LOW


def test_heuristic_notes_mention_no_llm():
    intent = classify_heuristically(QueryRequest(query="How are trials distributed across phases?"))
    assert "no LLM available" in intent.notes


def test_comparison_shaped_query_downgrades_and_notes_why():
    intent = classify_heuristically(QueryRequest(query="Compare drug A vs drug B."))
    assert intent.analysis_type == AnalysisType.DISTRIBUTION
    assert intent.confidence == Confidence.LOW
    assert "downgraded from comparison" in intent.notes


@pytest.mark.parametrize(
    "analysis_type,expected_viz",
    [
        (AnalysisType.TREND, VizType.TIME_SERIES),
        (AnalysisType.DISTRIBUTION, VizType.BAR_CHART),
        (AnalysisType.GEOGRAPHIC, VizType.BAR_CHART),
        (AnalysisType.NETWORK, VizType.NETWORK_GRAPH),
        (AnalysisType.COUNT, VizType.STAT_CARD),
    ],
)
def test_suggested_viz_matches_analysis_type(analysis_type, expected_viz):
    # COMPARISON is intentionally excluded: see
    # test_comparison_shaped_query_downgrades_and_notes_why for why the
    # heuristic path can never actually surface it.
    query_by_type = {
        AnalysisType.TREND: "How has this changed since 2015?",
        AnalysisType.DISTRIBUTION: "How are trials distributed?",
        AnalysisType.GEOGRAPHIC: "Which countries have the most trials?",
        AnalysisType.NETWORK: "Show a network of sponsors and drugs.",
        AnalysisType.COUNT: "How many trials are there in total?",
    }
    intent = classify_heuristically(QueryRequest(query=query_by_type[analysis_type]))
    assert intent.analysis_type == analysis_type
    assert intent.suggested_viz == expected_viz


def test_count_pattern_matches_when_no_trend_keywords_present():
    intent = classify_heuristically(QueryRequest(query="How many trials are there for pembrolizumab?"))
    assert intent.analysis_type == AnalysisType.COUNT


def test_entities_pulled_from_structured_request_fields():
    request = QueryRequest(
        query="How are trials distributed across phases?",
        drug_name="Pembrolizumab",
        condition="lung cancer",
        trial_phase="PHASE1,PHASE2",
        sponsor="Merck",
        country="Germany",
        status="RECRUITING",
        start_year=2015,
        end_year=2020,
    )
    intent = classify_heuristically(request)
    assert intent.entities.drug_name == "Pembrolizumab"
    assert intent.entities.condition == "lung cancer"
    assert intent.entities.trial_phase == "PHASE1,PHASE2"
    assert intent.entities.sponsor == "Merck"
    assert intent.entities.country == "Germany"
    assert intent.entities.status == "RECRUITING"
    assert intent.entities.start_year == 2015
    assert intent.entities.end_year == 2020


def test_heuristic_never_fabricates_compare_entities():
    # No NER without an LLM -- compare_a/compare_b stay unset even for a
    # clearly comparative query, and the schema's own downgrade rule handles
    # the resulting incomplete comparison.
    intent = classify_heuristically(QueryRequest(query="Compare drug A vs drug B."))
    assert intent.entities.compare_a is None
    assert intent.entities.compare_b is None
    # The Intent model's own validator downgrades this for us.
    assert intent.analysis_type == AnalysisType.DISTRIBUTION
