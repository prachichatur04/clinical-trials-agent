from app.analysis.aggregate import compare_groups, group_and_count
from app.analysis.extractors import by_phase, by_status
from app.schemas.internal import TrialRecord


def _record(nct_id: str, **overrides) -> TrialRecord:
    defaults = {"nct_id": nct_id}
    defaults.update(overrides)
    return TrialRecord(**defaults)


# --- group_and_count ---------------------------------------------------


def test_empty_input_yields_no_buckets():
    assert group_and_count([], by_phase) == []


def test_counts_correctly_across_multiple_records():
    records = [
        _record("NCT1", phases=["PHASE1"]),
        _record("NCT2", phases=["PHASE1"]),
        _record("NCT3", phases=["PHASE2"]),
    ]
    buckets = group_and_count(records, by_phase)
    counts = {b.key: b.count for b in buckets}
    assert counts == {"Phase 1": 2, "Phase 2": 1}


def test_sorted_by_count_descending():
    records = [
        _record("NCT1", phases=["PHASE2"]),
        _record("NCT2", phases=["PHASE1"]),
        _record("NCT3", phases=["PHASE1"]),
        _record("NCT4", phases=["PHASE1"]),
    ]
    buckets = group_and_count(records, by_phase)
    assert [b.key for b in buckets] == ["Phase 1", "Phase 2"]


def test_ties_broken_alphabetically_for_determinism():
    records = [_record("NCT1", overall_status="RECRUITING"), _record("NCT2", overall_status="COMPLETED")]
    buckets = group_and_count(records, by_status)
    assert [b.key for b in buckets] == ["COMPLETED", "RECRUITING"]


def test_top_n_truncates_to_largest_buckets():
    records = [
        _record("NCT1", overall_status="A"),
        _record("NCT2", overall_status="A"),
        _record("NCT3", overall_status="B"),
        _record("NCT4", overall_status="C"),
    ]
    buckets = group_and_count(records, by_status, top_n=1)
    assert len(buckets) == 1
    assert buckets[0].key == "A"


def test_top_n_none_keeps_all_buckets():
    records = [_record("NCT1", overall_status="A"), _record("NCT2", overall_status="B")]
    buckets = group_and_count(records, by_status, top_n=None)
    assert len(buckets) == 2


def test_samples_collected_per_bucket():
    records = [_record("NCT1", overall_status="RECRUITING"), _record("NCT2", overall_status="RECRUITING")]
    buckets = group_and_count(records, by_status)
    assert len(buckets[0].samples) == 2
    assert {s.nct_id for s in buckets[0].samples} == {"NCT1", "NCT2"}


def test_multi_count_extractor_inflates_bucket_count_correctly():
    # by_country yields one tuple per country -- a study with 2 countries
    # contributes to 2 buckets, each getting +1 (not the whole study to one).
    from app.analysis.extractors import by_country

    records = [_record("NCT1", countries=["United States", "Germany"])]
    buckets = group_and_count(records, by_country)
    counts = {b.key: b.count for b in buckets}
    assert counts == {"United States": 1, "Germany": 1}


# --- compare_groups ----------------------------------------------------


def test_compare_groups_zero_fills_keys_missing_on_one_side():
    records_a = [_record("NCT1", phases=["PHASE1"])]
    records_b = [_record("NCT2", phases=["PHASE2"])]

    buckets = compare_groups(records_a, records_b, by_phase, "Drug A", "Drug B")

    by_key_series = {(b.key, b.series): b.count for b in buckets}
    assert by_key_series[("Phase 1", "Drug A")] == 1
    assert by_key_series[("Phase 1", "Drug B")] == 0
    assert by_key_series[("Phase 2", "Drug A")] == 0
    assert by_key_series[("Phase 2", "Drug B")] == 1


def test_compare_groups_produces_two_buckets_per_key():
    records_a = [_record("NCT1", phases=["PHASE1"])]
    records_b = [_record("NCT2", phases=["PHASE1"])]

    buckets = compare_groups(records_a, records_b, by_phase, "A", "B")

    assert len(buckets) == 2
    assert {b.series for b in buckets} == {"A", "B"}


def test_compare_groups_empty_both_sides():
    assert compare_groups([], [], by_phase, "A", "B") == []


def test_compare_groups_all_keys_have_both_series_labels_present():
    records_a = [_record("NCT1", overall_status="RECRUITING"), _record("NCT2", overall_status="COMPLETED")]
    records_b = [_record("NCT3", overall_status="RECRUITING")]

    buckets = compare_groups(records_a, records_b, by_status, "A", "B")

    keys = {b.key for b in buckets}
    assert keys == {"RECRUITING", "COMPLETED"}
    for key in keys:
        series_for_key = {b.series for b in buckets if b.key == key}
        assert series_for_key == {"A", "B"}


def test_compare_groups_preserves_samples_per_series():
    records_a = [_record("NCT1", overall_status="RECRUITING")]
    records_b = [_record("NCT2", overall_status="RECRUITING")]

    buckets = compare_groups(records_a, records_b, by_status, "A", "B")

    bucket_a = next(b for b in buckets if b.series == "A")
    bucket_b = next(b for b in buckets if b.series == "B")
    assert [s.nct_id for s in bucket_a.samples] == ["NCT1"]
    assert [s.nct_id for s in bucket_b.samples] == ["NCT2"]
