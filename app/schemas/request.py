from pydantic import BaseModel, Field, model_validator

MIN_YEAR = 1990
MAX_YEAR = 2100
MAX_STUDIES_HARD_CAP = 5000


class QueryRequest(BaseModel):
    """POST /query body. `query` is the only required field -- everything
    else is an optional structured hint the caller may already know, taken
    as ground truth by the intent parser rather than re-derived from text.
    """

    query: str = Field(..., min_length=8, description="Natural-language question about clinical trials.")

    drug_name: str | None = None
    condition: str | None = None
    trial_phase: str | None = None
    sponsor: str | None = None
    country: str | None = None
    status: str | None = None
    start_year: int | None = Field(default=None, ge=MIN_YEAR, le=MAX_YEAR)
    end_year: int | None = Field(default=None, ge=MIN_YEAR, le=MAX_YEAR)

    # Comparison/dimension hints. Usually the LLM (or, for comparison,
    # nothing -- see heuristics.py's documented limitation) extracts these
    # from free text, but a caller who already knows what they want to
    # compare can supply them directly; pipeline.py treats these as ground
    # truth that overrides whatever Touch 1 guessed.
    compare_a: str | None = None
    compare_b: str | None = None
    compare_type: str | None = None
    dimension: str | None = None

    max_studies: int = Field(default=500, ge=1, le=MAX_STUDIES_HARD_CAP)
    include_citations: bool = True
    include_summary: bool = False

    @model_validator(mode="after")
    def validate_year_range(self) -> "QueryRequest":
        if self.start_year is not None and self.end_year is not None and self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        return self
