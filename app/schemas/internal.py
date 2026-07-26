from pydantic import BaseModel, Field

from app.schemas.intent import VizType
from app.schemas.viz import Node


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


class Sample(BaseModel):
    """Raw material for a deep citation, before citations/attach.py picks a
    deterministic subset and turns it into a viz.Citation with a study URL.
    """

    nct_id: str
    field_path: str
    excerpt: str


class Bucket(BaseModel):
    """One group_and_count()/compare_groups() output bucket. `series` is
    only set for comparison output (which side -- compare_a/compare_b --
    this bucket belongs to); plain distribution/trend/geographic buckets
    leave it None.
    """

    key: str
    count: int
    series: str | None = None
    samples: list[Sample] = Field(default_factory=list)


class EdgeSamples(BaseModel):
    """A network edge before citation attachment -- same idea as Bucket,
    carrying raw Samples instead of resolved Citations."""

    source: str
    target: str
    weight: int
    samples: list[Sample] = Field(default_factory=list)


class NetworkResult(BaseModel):
    """Output of build_cooccurrence_graph(). Nodes reuse viz.Node directly
    (id/label/type/weight/nct_ids) since nct_ids alone is already the node's
    citation trail -- only edges need a separate pre-citation-attachment
    shape (EdgeSamples) to carry excerpt material for a full deep Citation.
    """

    nodes: list[Node] = Field(default_factory=list)
    edges: list[EdgeSamples] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AggregatedResult(BaseModel):
    """What a dispatch.py run_*() function hands to viz/builder.py. Exactly
    one of `buckets`, `network`, or `stat_value` is meaningful for any given
    analysis_type -- which one depends on the paired viz_type in
    DispatchResult, not on anything in this model itself.
    """

    buckets: list[Bucket] = Field(default_factory=list)
    network: NetworkResult | None = None
    stat_value: int | None = None
    dimension_used: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class DispatchResult(BaseModel):
    viz_type: VizType
    title: str
    aggregated: AggregatedResult
