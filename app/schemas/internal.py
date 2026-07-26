from pydantic import BaseModel, Field


class TrialRecord(BaseModel):
    """A single CTGov study, flattened to the fields the analysis layer needs.

    `raw` retains the original study JSON so citations/attach.py can pull an
    exact excerpt for a given field path later; it's excluded from
    serialization so it never leaks into an API response.
    """

    nct_id: str
    brief_title: str | None = None
    overall_status: str | None = None
    start_date: str | None = None
    study_first_post_date: str | None = None
    phases: list[str] = Field(default_factory=list)
    lead_sponsor_name: str | None = None
    lead_sponsor_class: str | None = None
    collaborator_names: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    intervention_types: list[str] = Field(default_factory=list)
    intervention_names: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    raw: dict = Field(default_factory=dict, exclude=True, repr=False)

    @property
    def study_url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"
