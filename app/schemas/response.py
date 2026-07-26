from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.intent import AnalysisType
from app.schemas.viz import VisualizationSpec


class IntentSource(str, Enum):
    LLM = "llm"
    LLM_RETRY = "llm_retry"
    HEURISTIC_FALLBACK = "heuristic_fallback"


class Meta(BaseModel):
    query_interpretation: str
    query_plan: str
    analysis_type: AnalysisType
    filters_applied: dict = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    total_studies_matched: int
    total_studies_fetched: int
    unique_study_count: int
    source: str
    generated_at: datetime
    intent_source: IntentSource


class QueryResponse(BaseModel):
    visualization: VisualizationSpec
    summary: str | None = None
    meta: Meta


class ErrorType(str, Enum):
    VALIDATION_ERROR = "validation_error"
    PARSING_ERROR = "parsing_error"
    NO_RESULTS = "no_results"
    API_ERROR = "api_error"
    UNSUPPORTED_QUERY = "unsupported_query"
    INTERNAL_ERROR = "internal_error"


class ErrorResponse(BaseModel):
    error_type: ErrorType
    message: str
    suggestion: str | None = None
