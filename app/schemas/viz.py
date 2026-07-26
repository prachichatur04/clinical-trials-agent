from pydantic import BaseModel, Field

from app.schemas.intent import VizType


class Citation(BaseModel):
    """Deep citation for a single visualized datum -- a bar, a time bucket,
    a node/edge weight -- back to the exact study record that produced it.
    """

    nct_id: str
    field_path: str
    excerpt: str
    url: str


class EncodingChannel(BaseModel):
    field: str
    type: str | None = None  # e.g. "temporal", "quantitative", "nominal"


class Encoding(BaseModel):
    """Field -> visual channel mapping. Bar/time-series/scatter/histogram
    charts use x/y(/series); network graphs use nodes/edges instead -- a
    VisualizationSpec only ever populates the subset its type needs.
    """

    x: EncodingChannel | None = None
    y: EncodingChannel | None = None
    series: EncodingChannel | None = None
    nodes: dict[str, str] | None = None
    edges: dict[str, str] | None = None


class DataPoint(BaseModel):
    """One point on a bar/time-series/scatter/histogram chart.

    The same x/y/series shape covers every non-network viz type instead of
    a bespoke shape per chart -- e.g. a histogram's bin label is `x`, its
    frequency is `y`, exactly like a bar chart's category and count.
    """

    x: str | int | float | None = None
    y: str | int | float | None = None
    series: str | None = None
    citations: list[Citation] = Field(default_factory=list)


class Node(BaseModel):
    id: str
    label: str
    type: str
    weight: int
    nct_ids: list[str] = Field(default_factory=list)


class Edge(BaseModel):
    source: str
    target: str
    weight: int
    citations: list[Citation] = Field(default_factory=list)


class NetworkData(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class VisualizationSpec(BaseModel):
    type: VizType
    title: str
    encoding: Encoding
    data: list[DataPoint] = Field(default_factory=list)
    network_data: NetworkData | None = None
