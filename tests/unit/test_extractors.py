from app.analysis.extractors import (
    by_country,
    by_intervention_type,
    by_phase,
    by_sponsor_class,
    by_sponsor_name,
    by_status,
    by_year,
)
from app.schemas.internal import TrialRecord


def _record(**overrides) -> TrialRecord:
    defaults = {"nct_id": "NCT00000001"}
    defaults.update(overrides)
    return TrialRecord(**defaults)


# --- by_year -----------------------------------------------------------


def test_by_year_uses_start_date():
    record = _record(start_date="2015-03-14", study_first_post_date="2016-01-01")
    (key, nct_id, field_path, excerpt) = next(by_year(record))
    assert key == "2015"
    assert nct_id == "NCT00000001"
    assert field_path == "protocolSection.statusModule.startDateStruct.date"
    assert excerpt == "2015-03-14"


def test_by_year_falls_back_to_study_first_post_date_when_start_date_missing():
    record = _record(study_first_post_date="2016-01-01")
    (key, _, field_path, excerpt) = next(by_year(record))
    assert key == "2016"
    assert field_path == "protocolSection.statusModule.studyFirstPostDateStruct.date"
    assert excerpt == "2016-01-01"


def test_by_year_yields_unknown_when_both_dates_missing():
    record = _record()
    (key, _, _, excerpt) = next(by_year(record))
    assert key == "unknown"
    assert excerpt == "no start date recorded"


def test_by_year_yields_exactly_one_tuple():
    record = _record(start_date="2015-03-14")
    assert len(list(by_year(record))) == 1


# --- by_phase ------------------------------------------------------------


def test_by_phase_single_phase():
    record = _record(phases=["PHASE2"])
    (key, _, _, excerpt) = next(by_phase(record))
    assert key == "Phase 2"
    assert excerpt == "PHASE2"


def test_by_phase_multi_phase_gets_combined_category():
    record = _record(phases=["PHASE1", "PHASE2"])
    (key, _, _, excerpt) = next(by_phase(record))
    assert key == "Phase 1/Phase 2"
    assert excerpt == "PHASE1, PHASE2"


def test_by_phase_missing_yields_na():
    record = _record(phases=[])
    (key, _, _, excerpt) = next(by_phase(record))
    assert key == "N/A"
    assert excerpt == "no phase recorded"


def test_by_phase_unrecognized_code_passed_through():
    record = _record(phases=["NA"])
    (key, _, _, _) = next(by_phase(record))
    assert key == "N/A"


def test_by_phase_yields_exactly_one_tuple_even_for_multi_phase():
    record = _record(phases=["PHASE1", "PHASE2", "PHASE3"])
    assert len(list(by_phase(record))) == 1


# --- by_status -------------------------------------------------------------


def test_by_status_present():
    record = _record(overall_status="RECRUITING")
    (key, _, _, excerpt) = next(by_status(record))
    assert key == "RECRUITING"
    assert excerpt == "RECRUITING"


def test_by_status_missing_yields_unknown():
    record = _record()
    (key, _, _, excerpt) = next(by_status(record))
    assert key == "UNKNOWN"
    assert excerpt == "no status recorded"


# --- by_sponsor_class / by_sponsor_name ------------------------------------


def test_by_sponsor_class_present():
    record = _record(lead_sponsor_class="INDUSTRY")
    (key, _, _, _) = next(by_sponsor_class(record))
    assert key == "INDUSTRY"


def test_by_sponsor_class_missing_yields_unknown():
    record = _record()
    (key, _, _, _) = next(by_sponsor_class(record))
    assert key == "UNKNOWN"


def test_by_sponsor_name_present():
    record = _record(lead_sponsor_name="Pfizer")
    (key, _, _, _) = next(by_sponsor_name(record))
    assert key == "Pfizer"


def test_by_sponsor_name_missing_yields_unknown_sponsor():
    record = _record()
    (key, _, _, _) = next(by_sponsor_name(record))
    assert key == "Unknown Sponsor"


# --- by_country (multi-count, already deduped per study) --------------------


def test_by_country_single_country():
    record = _record(countries=["Germany"])
    results = list(by_country(record))
    assert [r[0] for r in results] == ["Germany"]


def test_by_country_multiple_countries_each_yields_own_tuple():
    record = _record(countries=["United States", "Germany"])
    results = list(by_country(record))
    assert [r[0] for r in results] == ["United States", "Germany"]
    assert all(r[1] == "NCT00000001" for r in results)


def test_by_country_missing_yields_unknown():
    record = _record()
    (key, _, _, excerpt) = next(by_country(record))
    assert key == "Unknown"
    assert excerpt == "no location recorded"


# --- by_intervention_type (multi-count) -------------------------------------


def test_by_intervention_type_multiple_types_each_yields_own_tuple():
    record = _record(intervention_types=["DRUG", "BIOLOGICAL"])
    results = list(by_intervention_type(record))
    assert [r[0] for r in results] == ["DRUG", "BIOLOGICAL"]


def test_by_intervention_type_missing_yields_unknown():
    record = _record()
    (key, _, _, _) = next(by_intervention_type(record))
    assert key == "UNKNOWN"


def test_by_intervention_type_duplicate_types_counted_twice():
    # Two DRUG interventions on one study -> two tuples, so group_and_count
    # will count this study twice under "DRUG" -- intentional multi-count.
    record = _record(intervention_types=["DRUG", "DRUG"])
    results = list(by_intervention_type(record))
    assert [r[0] for r in results] == ["DRUG", "DRUG"]
