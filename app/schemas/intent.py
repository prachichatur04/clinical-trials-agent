from enum import Enum

from pydantic import BaseModel, Field, model_validator


class AnalysisType(str, Enum):
    TREND = "trend"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    GEOGRAPHIC = "geographic"
    NETWORK = "network"
    COUNT = "count"


class VizType(str, Enum):
    BAR_CHART = "bar_chart"
    TIME_SERIES = "time_series"
    GROUPED_BAR_CHART = "grouped_bar_chart"
    HISTOGRAM = "histogram"
    SCATTER_PLOT = "scatter_plot"
    NETWORK_GRAPH = "network_graph"
    STAT_CARD = "stat_card"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Entities(BaseModel):
    drug_name: str | None = None
    condition: str | None = None
    trial_phase: str | None = None
    sponsor: str | None = None
    country: str | None = None
    status: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    dimension: str | None = None
    compare_a: str | None = None
    compare_b: str | None = None
    compare_type: str | None = None


class Intent(BaseModel):
    """Output of Touch 1 (planning + classification). Produced by the LLM
    (structured outputs) or the heuristic fallback -- both paths converge on
    this same shape, so everything downstream only ever handles one type.
    """

    analysis_type: AnalysisType
    entities: Entities = Field(default_factory=Entities)
    suggested_viz: VizType
    query_plan: str
    notes: str
    confidence: Confidence

    @model_validator(mode="after")
    def downgrade_incomplete_comparison(self) -> "Intent":
        """Schema-level safety net for the "comparison needs both entities"
        rule: enforced here so it holds regardless of whether the LLM or the
        heuristic fallback produced this Intent, not just as a prompt rule.
        """
        is_comparison = self.analysis_type == AnalysisType.COMPARISON
        has_both_entities = bool(self.entities.compare_a and self.entities.compare_b)
        if is_comparison and not has_both_entities:
            self.analysis_type = AnalysisType.DISTRIBUTION
            self.confidence = Confidence.LOW
            self.notes = (
                f"{self.notes} (downgraded from comparison: missing compare_a/compare_b)".strip()
            )
        return self
