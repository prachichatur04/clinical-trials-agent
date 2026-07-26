from app.analysis.dispatch import (
    ANALYSIS_DISPATCH,
    run_comparison,
    run_count,
    run_distribution,
    run_geographic,
    run_network,
    run_trend,
)
from app.schemas.intent import AnalysisType, Confidence, Entities, Intent, VizType
from app.schemas.internal import TrialRecord


def _record(nct_id: str, **overrides) -> TrialRecord:
    defaults = {"nct_id": nct_id}
    defaults.update(overrides)
    return TrialRecord(**defaults)


def _intent(entities: Entities | None = None) -> Intent:
    return Intent(
        analysis_type=AnalysisType.DISTRIBUTION,
        entities=entities or Entities(),
        suggested_viz=VizType.BAR_CHART,
        query_plan="plan",
        notes="notes",
        confidence=Confidence.HIGH,
    )


# --- run_trend ---------------------------------------------------------


def test_run_trend_sorts_chronologically_not_by_count():
    records = [
        _record("NCT1", start_date="2020-01-01"),
        _record("NCT2", start_date="2020-01-01"),
        _record("NCT3", start_date="2015-01-01"),
    ]
    result = run_trend(records, _intent())
    assert [b.key for b in result.aggregated.buckets] == ["2015", "2020"]
    assert result.viz_type == VizType.TIME_SERIES


def test_run_trend_puts_unknown_bucket_last():
    records = [_record("NCT1", start_date="2020-01-01"), _record("NCT2")]
    result = run_trend(records, _intent())
    assert [b.key for b in result.aggregated.buckets] == ["2020", "unknown"]


def test_run_trend_reports_unknown_date_assumption():
    records = [_record("NCT1"), _record("NCT2")]
    result = run_trend(records, _intent())
    assert "2 studies" in result.aggregated.assumptions[0]
    assert "unparseable start date" in result.aggregated.assumptions[0]


def test_run_trend_no_assumption_when_all_dates_present():
    records = [_record("NCT1", start_date="2020-01-01")]
    result = run_trend(records, _intent())
    assert result.aggregated.assumptions == []


def test_run_trend_title_includes_drug_name():
    result = run_trend([], _intent(Entities(drug_name="Pembrolizumab")))
    assert "Pembrolizumab" in result.title


# --- run_distribution ----------------------------------------------------


def test_run_distribution_defaults_to_phase_dimension():
    records = [_record("NCT1", phases=["PHASE1"])]
    result = run_distribution(records, _intent())
    assert result.aggregated.dimension_used == "phase"
    assert result.aggregated.buckets[0].key == "Phase 1"


def test_run_distribution_uses_requested_dimension():
    records = [_record("NCT1", overall_status="RECRUITING")]
    result = run_distribution(records, _intent(Entities(dimension="status")))
    assert result.aggregated.dimension_used == "status"
    assert result.aggregated.buckets[0].key == "RECRUITING"


def test_run_distribution_unrecognized_dimension_falls_back_to_phase():
    records = [_record("NCT1", phases=["PHASE1"])]
    result = run_distribution(records, _intent(Entities(dimension="not_a_real_dimension")))
    assert result.aggregated.buckets[0].key == "Phase 1"


def test_run_distribution_truncates_to_top_n():
    records = [_record(f"NCT{i}", lead_sponsor_name=f"Sponsor{i}") for i in range(30)]
    result = run_distribution(records, _intent(Entities(dimension="sponsor_name")))
    assert len(result.aggregated.buckets) == 20


def test_run_distribution_reports_multi_phase_assumption():
    records = [_record("NCT1", phases=["PHASE1", "PHASE2"])]
    result = run_distribution(records, _intent(Entities(dimension="phase")))
    assert "multiple phases" in result.aggregated.assumptions[0]
    assert "Phase 1/Phase 2" in result.aggregated.assumptions[0]


def test_run_distribution_na_bucket_not_mistaken_for_multi_phase():
    # Regression: "N/A" contains a literal "/" too -- must not be counted
    # as a combined multi-phase bucket, and must not appear in the assumption.
    records = [
        _record("NCT1", phases=[]),  # -> "N/A"
        _record("NCT2", phases=["PHASE1", "PHASE2"]),  # -> "Phase 1/Phase 2"
    ]
    result = run_distribution(records, _intent(Entities(dimension="phase")))
    assumption = result.aggregated.assumptions[0]
    assert "1 studies" in assumption
    assert "N/A" not in assumption


def test_run_distribution_no_multi_phase_assumption_for_other_dimensions():
    records = [_record("NCT1", overall_status="RECRUITING")]
    result = run_distribution(records, _intent(Entities(dimension="status")))
    assert result.aggregated.assumptions == []


# --- run_geographic ----------------------------------------------------


def test_run_geographic_groups_by_country():
    records = [_record("NCT1", countries=["Germany"]), _record("NCT2", countries=["Germany"])]
    result = run_geographic(records, _intent())
    assert result.aggregated.buckets[0].key == "Germany"
    assert result.aggregated.buckets[0].count == 2
    assert result.viz_type == VizType.BAR_CHART


# --- run_network ---------------------------------------------------------


def test_run_network_delegates_to_cooccurrence_graph():
    records = [
        _record("NCT1", lead_sponsor_name="Pfizer", intervention_types=["DRUG"], intervention_names=["X"])
    ]
    result = run_network(records, _intent())
    assert result.viz_type == VizType.NETWORK_GRAPH
    assert result.aggregated.network is not None
    assert len(result.aggregated.network.nodes) == 2


def test_run_network_surfaces_notes_as_assumptions():
    records = [_record("NCT1", intervention_types=["DRUG"], intervention_names=["A"])]
    result = run_network(records, _intent(Entities(dimension="drug_cooccurrence")))
    assert result.aggregated.assumptions == result.aggregated.network.notes
    assert len(result.aggregated.assumptions) == 1


# --- run_comparison ----------------------------------------------------


def test_run_comparison_zero_fills_and_labels_series():
    records_a = [_record("NCT1", phases=["PHASE1"])]
    records_b = [_record("NCT2", phases=["PHASE2"])]
    intent = _intent(Entities(compare_a="Keytruda", compare_b="Opdivo", dimension="phase"))

    result = run_comparison(records_a, records_b, intent)

    assert result.viz_type == VizType.GROUPED_BAR_CHART
    assert "Keytruda" in result.title
    assert "Opdivo" in result.title
    series_labels = {b.series for b in result.aggregated.buckets}
    assert series_labels == {"Keytruda", "Opdivo"}


def test_run_comparison_defaults_labels_when_entities_missing():
    result = run_comparison([], [], _intent())
    assert "A" in result.title
    assert "B" in result.title


# --- run_count -----------------------------------------------------------


def test_run_count_uses_server_total_not_fetched_length():
    result = run_count(_intent(), total_matched=8123)
    assert result.viz_type == VizType.STAT_CARD
    assert result.aggregated.stat_value == 8123


# --- ANALYSIS_DISPATCH ---------------------------------------------------


def test_analysis_dispatch_covers_the_four_uniform_signature_types():
    assert set(ANALYSIS_DISPATCH.keys()) == {"trend", "distribution", "geographic", "network"}


def test_analysis_dispatch_functions_are_callable_with_records_and_intent():
    for analysis_type in AnalysisType:
        if analysis_type.value not in ANALYSIS_DISPATCH:
            continue
        fn = ANALYSIS_DISPATCH[analysis_type.value]
        result = fn([], _intent())
        assert result.aggregated is not None
