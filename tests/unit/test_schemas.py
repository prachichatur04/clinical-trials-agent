from datetime import UTC, datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.exceptions import (
    ApiError,
    AppError,
    InternalError,
    NoResultsError,
    ParsingError,
    UnsupportedQueryError,
    ValidationError,
)
from app.schemas.intent import AnalysisType, Confidence, Entities, Intent, VizType
from app.schemas.request import QueryRequest
from app.schemas.response import ErrorResponse, ErrorType, IntentSource, Meta, QueryResponse
from app.schemas.viz import (
    Citation,
    DataPoint,
    Edge,
    Encoding,
    EncodingChannel,
    NetworkData,
    Node,
    VisualizationSpec,
)

# --- QueryRequest -----------------------------------------------------------


def test_valid_request_passes():
    req = QueryRequest(query="How has pembrolizumab trended since 2015?", drug_name="pembrolizumab")
    assert req.max_studies == 500
    assert req.include_citations is True
    assert req.include_summary is False


def test_missing_query_raises():
    with pytest.raises(PydanticValidationError):
        QueryRequest()


def test_too_short_query_raises():
    with pytest.raises(PydanticValidationError):
        QueryRequest(query="short")


def test_bad_year_range_raises():
    with pytest.raises(PydanticValidationError):
        QueryRequest(query="How many trials started each year?", start_year=2020, end_year=2015)


def test_equal_start_and_end_year_is_valid():
    req = QueryRequest(query="How many trials started each year?", start_year=2020, end_year=2020)
    assert req.start_year == req.end_year == 2020


def test_max_studies_too_low_raises():
    with pytest.raises(PydanticValidationError):
        QueryRequest(query="How many trials started each year?", max_studies=0)


def test_max_studies_too_high_raises():
    with pytest.raises(PydanticValidationError):
        QueryRequest(query="How many trials started each year?", max_studies=5001)


def test_max_studies_at_hard_cap_is_valid():
    req = QueryRequest(query="How many trials started each year?", max_studies=5000)
    assert req.max_studies == 5000


def test_compare_fields_and_dimension_are_optional_and_none_by_default():
    req = QueryRequest(query="How are trials distributed across phases?")
    assert req.compare_a is None
    assert req.compare_b is None
    assert req.compare_type is None
    assert req.dimension is None


def test_compare_fields_and_dimension_accept_explicit_values():
    req = QueryRequest(
        query="Compare Keytruda vs Opdivo by phase.",
        compare_a="Keytruda",
        compare_b="Opdivo",
        compare_type="drug",
        dimension="phase",
    )
    assert req.compare_a == "Keytruda"
    assert req.compare_b == "Opdivo"
    assert req.compare_type == "drug"
    assert req.dimension == "phase"


# --- Intent -------------------------------------------------------------


def _intent(analysis_type, entities=None, **overrides):
    defaults = {
        "analysis_type": analysis_type,
        "entities": entities or Entities(),
        "suggested_viz": VizType.BAR_CHART,
        "query_plan": "plan",
        "notes": "notes",
        "confidence": Confidence.HIGH,
    }
    defaults.update(overrides)
    return Intent(**defaults)


def test_comparison_with_both_entities_stays_comparison():
    intent = _intent(
        AnalysisType.COMPARISON, entities=Entities(compare_a="Keytruda", compare_b="Opdivo")
    )
    assert intent.analysis_type == AnalysisType.COMPARISON
    assert intent.confidence == Confidence.HIGH


def test_comparison_missing_compare_b_downgrades_to_distribution():
    intent = _intent(AnalysisType.COMPARISON, entities=Entities(compare_a="Keytruda"))
    assert intent.analysis_type == AnalysisType.DISTRIBUTION
    assert intent.confidence == Confidence.LOW
    assert "downgraded from comparison" in intent.notes


def test_comparison_downgrade_also_annotates_query_plan_not_just_notes():
    # Regression: query_plan previously kept describing the pre-downgrade
    # analysis_type ('comparison') even after analysis_type and notes both
    # correctly reflected the downgrade to 'distribution' -- an internally
    # inconsistent response. Found via manual UI testing.
    intent = _intent(
        AnalysisType.COMPARISON,
        entities=Entities(compare_a="Keytruda"),
        query_plan="Search for Keytruda vs [missing], compare by phase.",
    )
    assert intent.analysis_type == AnalysisType.DISTRIBUTION
    assert "downgraded from comparison" in intent.query_plan


def test_comparison_missing_both_entities_downgrades_to_distribution():
    intent = _intent(AnalysisType.COMPARISON, entities=Entities())
    assert intent.analysis_type == AnalysisType.DISTRIBUTION
    assert intent.confidence == Confidence.LOW


def test_non_comparison_analysis_type_unaffected_by_downgrade_rule():
    intent = _intent(AnalysisType.TREND, entities=Entities())
    assert intent.analysis_type == AnalysisType.TREND
    assert intent.confidence == Confidence.HIGH


# --- Viz schemas ----------------------------------------------------------


def test_bar_chart_style_visualization_spec_round_trips():
    spec = VisualizationSpec(
        type=VizType.TIME_SERIES,
        title="Pembrolizumab trials started per year, 2015-2026",
        encoding=Encoding(
            x=EncodingChannel(field="year", type="temporal"),
            y=EncodingChannel(field="count", type="quantitative"),
        ),
        data=[
            DataPoint(
                x="2015",
                y=12,
                citations=[
                    Citation(
                        nct_id="NCT02335411",
                        field_path="protocolSection.statusModule.startDateStruct.date",
                        excerpt="2015-01",
                        url="https://clinicaltrials.gov/study/NCT02335411",
                    )
                ],
            )
        ],
    )
    dumped = spec.model_dump(mode="json")
    assert dumped["type"] == "time_series"
    assert dumped["encoding"]["x"]["field"] == "year"
    assert dumped["data"][0]["citations"][0]["nct_id"] == "NCT02335411"
    assert dumped["network_data"] is None


def test_network_graph_visualization_spec_uses_network_data_not_data():
    spec = VisualizationSpec(
        type=VizType.NETWORK_GRAPH,
        title="Sponsor-drug network",
        encoding=Encoding(
            nodes={"id": "id", "label": "label", "group": "type", "size": "weight"},
            edges={"source": "source", "target": "target", "width": "weight"},
        ),
        data=[],
        network_data=NetworkData(
            nodes=[Node(id="s_pfizer", label="Pfizer", type="sponsor", weight=34)],
            edges=[Edge(source="s_pfizer", target="d_pembrolizumab", weight=12)],
        ),
    )
    assert spec.data == []
    assert spec.network_data.nodes[0].id == "s_pfizer"
    assert spec.network_data.edges[0].weight == 12


# --- Response / Meta -------------------------------------------------------


def test_full_query_response_matches_plan_shape():
    response = QueryResponse(
        visualization=VisualizationSpec(
            type=VizType.TIME_SERIES,
            title="Pembrolizumab trials started per year, 2015-2026",
            encoding=Encoding(
                x=EncodingChannel(field="year", type="temporal"),
                y=EncodingChannel(field="count", type="quantitative"),
            ),
            data=[DataPoint(x="2015", y=12)],
        ),
        summary="Trial activity for pembrolizumab grew steadily since 2015.",
        meta=Meta(
            query_interpretation="Yearly trend of trial counts for pembrolizumab from 2015.",
            query_plan="Search for pembrolizumab, group by year.",
            analysis_type=AnalysisType.TREND,
            filters_applied={"drug_name": "pembrolizumab", "start_year": 2015},
            assumptions=["Year from startDateStruct.date; 4 missing -> 'unknown' bucket."],
            total_studies_matched=812,
            total_studies_fetched=500,
            unique_study_count=500,
            source="https://clinicaltrials.gov/api/v2/studies",
            generated_at=datetime(2026, 7, 26, 14, 0, tzinfo=UTC),
            intent_source=IntentSource.LLM,
        ),
    )
    assert response.meta.analysis_type == AnalysisType.TREND
    assert response.meta.intent_source == IntentSource.LLM


def test_error_response_enumerates_all_error_types():
    for error_type in ErrorType:
        err = ErrorResponse(error_type=error_type, message="x")
        assert err.error_type == error_type


# --- Exceptions -------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_cls,expected_type,expected_status",
    [
        (ValidationError, ErrorType.VALIDATION_ERROR, 422),
        (ParsingError, ErrorType.PARSING_ERROR, 502),
        (NoResultsError, ErrorType.NO_RESULTS, 200),
        (ApiError, ErrorType.API_ERROR, 502),
        (UnsupportedQueryError, ErrorType.UNSUPPORTED_QUERY, 422),
        (InternalError, ErrorType.INTERNAL_ERROR, 500),
    ],
)
def test_each_app_error_subclass_maps_to_its_error_type_and_status(exc_cls, expected_type, expected_status):
    err = exc_cls("something went wrong", suggestion="try again")
    assert err.error_type == expected_type
    assert err.status_code == expected_status

    response = err.to_response()
    assert response.error_type == expected_type
    assert response.message == "something went wrong"
    assert response.suggestion == "try again"


def test_app_error_suggestion_defaults_to_none():
    err = InternalError("boom")
    assert err.suggestion is None
    assert err.to_response().suggestion is None


def test_app_error_is_a_real_exception_and_can_be_raised():
    with pytest.raises(AppError):
        raise NoResultsError("No trials found.", suggestion="Try a broader search.")
