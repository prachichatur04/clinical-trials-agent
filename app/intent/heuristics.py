import re

from app.schemas.intent import AnalysisType, Confidence, Entities, Intent, VizType
from app.schemas.request import QueryRequest

# Checked in order -- network/comparison/geographic are the most specific
# patterns and must be tried before the broader trend/count ones. Trend is
# checked before count so e.g. "How many trials started each year" (a time
# trend in the assignment appendix, despite containing "how many") lands as
# trend, not count.
_PATTERNS: list[tuple[re.Pattern, AnalysisType]] = [
    (re.compile(r"\bnetwork\b|\bco-occur|\bcombination\b", re.IGNORECASE), AnalysisType.NETWORK),
    (re.compile(r"\bcompare\b|\bvs\.?\b|\bversus\b", re.IGNORECASE), AnalysisType.COMPARISON),
    (re.compile(r"\bcountr|\bwhere\b|\blocation\b", re.IGNORECASE), AnalysisType.GEOGRAPHIC),
    (
        re.compile(
            r"\bper year\b|\beach year\b|\bby year\b|\bsince\b|\btrend\b|\bover time\b|\bchanged\b",
            re.IGNORECASE,
        ),
        AnalysisType.TREND,
    ),
    (re.compile(r"\bhow many\b|\bcount\b|\btotal\b", re.IGNORECASE), AnalysisType.COUNT),
]

_SUGGESTED_VIZ: dict[AnalysisType, VizType] = {
    AnalysisType.TREND: VizType.TIME_SERIES,
    AnalysisType.DISTRIBUTION: VizType.BAR_CHART,
    AnalysisType.COMPARISON: VizType.GROUPED_BAR_CHART,
    AnalysisType.GEOGRAPHIC: VizType.BAR_CHART,
    AnalysisType.NETWORK: VizType.NETWORK_GRAPH,
    AnalysisType.COUNT: VizType.STAT_CARD,
}


def classify_heuristically(request: QueryRequest) -> Intent:
    """Regex/keyword classifier used when no LLM is available, or when the
    LLM path fails twice. Limited by design: no real entity extraction from
    free text, only whatever structured fields the caller already supplied.

    Notably, a "comparison"-shaped query (e.g. "Drug A vs Drug B") can never
    surface as analysis_type=comparison here, since compare_a/compare_b can
    only come from NER this path doesn't have. Intent's own model validator
    downgrades it to distribution -- working as intended, not a bug.
    """
    analysis_type = _match_analysis_type(request.query)
    return Intent(
        analysis_type=analysis_type,
        entities=_entities_from_request(request),
        suggested_viz=_SUGGESTED_VIZ[analysis_type],
        query_plan=(
            f"Heuristic fallback: keyword-matched query text to analysis_type="
            f"'{analysis_type.value}'; filters taken only from structured request fields."
        ),
        notes=f"Classified as '{analysis_type.value}' via keyword match (no LLM available).",
        confidence=Confidence.LOW,
    )


def _match_analysis_type(query: str) -> AnalysisType:
    for pattern, analysis_type in _PATTERNS:
        if pattern.search(query):
            return analysis_type
    return AnalysisType.DISTRIBUTION


def _entities_from_request(request: QueryRequest) -> Entities:
    return Entities(
        drug_name=request.drug_name,
        condition=request.condition,
        trial_phase=request.trial_phase,
        sponsor=request.sponsor,
        country=request.country,
        status=request.status,
        start_year=request.start_year,
        end_year=request.end_year,
    )
