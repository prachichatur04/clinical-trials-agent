from app.analysis.aggregate import compare_groups, group_and_count
from app.analysis.extractors import (
    by_country,
    by_intervention_type,
    by_phase,
    by_sponsor_class,
    by_sponsor_name,
    by_status,
    by_year,
)
from app.analysis.network import build_cooccurrence_graph
from app.schemas.intent import Intent, VizType
from app.schemas.internal import AggregatedResult, Bucket, DispatchResult, TrialRecord

DEFAULT_DIMENSION = "phase"
DISTRIBUTION_TOP_N = 20

DIMENSION_MAP = {
    "phase": by_phase,
    "status": by_status,
    "sponsor_class": by_sponsor_class,
    "sponsor_name": by_sponsor_name,
    "intervention_type": by_intervention_type,
    "country": by_country,
}


def run_trend(records: list[TrialRecord], intent: Intent) -> DispatchResult:
    buckets = group_and_count(records, by_year)
    buckets.sort(key=lambda b: (b.key == "unknown", b.key))  # chronological, not count-descending

    assumptions = []
    if any(b.key == "unknown" for b in buckets):
        unknown_count = next(b.count for b in buckets if b.key == "unknown")
        assumptions.append(f"{unknown_count} studies have a missing/unparseable start date -> 'unknown' bucket.")

    return DispatchResult(
        viz_type=VizType.TIME_SERIES,
        title=_title_for(intent, "trials started per year"),
        aggregated=AggregatedResult(buckets=buckets, dimension_used="year", assumptions=assumptions),
    )


def run_distribution(records: list[TrialRecord], intent: Intent) -> DispatchResult:
    dimension = intent.entities.dimension or DEFAULT_DIMENSION
    extractor = DIMENSION_MAP.get(dimension, by_phase)
    buckets = group_and_count(records, extractor, top_n=DISTRIBUTION_TOP_N)

    return DispatchResult(
        viz_type=VizType.BAR_CHART,
        title=_title_for(intent, f"trials by {dimension.replace('_', ' ')}"),
        aggregated=AggregatedResult(
            buckets=buckets,
            dimension_used=dimension,
            assumptions=_phase_multivalue_assumption(buckets) if dimension == "phase" else [],
        ),
    )


def run_geographic(records: list[TrialRecord], intent: Intent) -> DispatchResult:
    buckets = group_and_count(records, by_country, top_n=DISTRIBUTION_TOP_N)
    return DispatchResult(
        viz_type=VizType.BAR_CHART,
        title=_title_for(intent, "trials by country"),
        aggregated=AggregatedResult(buckets=buckets, dimension_used="country"),
    )


def run_network(records: list[TrialRecord], intent: Intent) -> DispatchResult:
    network = build_cooccurrence_graph(records, intent)
    return DispatchResult(
        viz_type=VizType.NETWORK_GRAPH,
        title=_title_for(intent, "trial network"),
        aggregated=AggregatedResult(network=network, assumptions=list(network.notes)),
    )


# run_comparison and run_count are intentionally NOT in a single uniform
# dispatch dict with the four functions above: comparison needs two record
# sets (fetched separately for compare_a/compare_b) and count needs the
# server's authoritative total_matched, which isn't available from records
# alone if the fetch was capped. Forcing them into the same (records, intent)
# signature as the others would just move that special-casing into an
# awkward wrapper instead of removing it -- the pipeline (Phase 5) branches
# on analysis_type once, here, rather than pretending otherwise.


def run_comparison(records_a: list[TrialRecord], records_b: list[TrialRecord], intent: Intent) -> DispatchResult:
    dimension = intent.entities.dimension or DEFAULT_DIMENSION
    extractor = DIMENSION_MAP.get(dimension, by_phase)
    label_a = intent.entities.compare_a or "A"
    label_b = intent.entities.compare_b or "B"
    buckets = compare_groups(records_a, records_b, extractor, label_a, label_b)

    return DispatchResult(
        viz_type=VizType.GROUPED_BAR_CHART,
        title=f"{label_a} vs {label_b}: trials by {dimension.replace('_', ' ')}",
        aggregated=AggregatedResult(buckets=buckets, dimension_used=dimension),
    )


def run_count(intent: Intent, total_matched: int) -> DispatchResult:
    return DispatchResult(
        viz_type=VizType.STAT_CARD,
        title=_title_for(intent, "total trials"),
        aggregated=AggregatedResult(stat_value=total_matched),
    )


ANALYSIS_DISPATCH = {
    "trend": run_trend,
    "distribution": run_distribution,
    "geographic": run_geographic,
    "network": run_network,
}


def _title_for(intent: Intent, suffix: str) -> str:
    entities = intent.entities
    subject = entities.drug_name or entities.condition or entities.sponsor
    if subject:
        return f"{subject}: {suffix}"
    return suffix.capitalize()


def _phase_multivalue_assumption(buckets: list[Bucket]) -> list[str]:
    combined = [b for b in buckets if "/" in b.key]
    if not combined:
        return []
    total = sum(b.count for b in combined)
    examples = ", ".join(b.key for b in combined[:2])
    return [f"{total} studies have multiple phases and are counted under a combined bucket (e.g. '{examples}')."]
