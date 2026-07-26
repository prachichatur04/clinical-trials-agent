from app.citations.attach import attach_citations
from app.schemas.intent import VizType
from app.schemas.internal import AggregatedResult
from app.schemas.viz import (
    DataPoint,
    Edge,
    Encoding,
    EncodingChannel,
    NetworkData,
    VisualizationSpec,
)


def build_visualization(
    viz_type: VizType,
    title: str,
    result: AggregatedResult,
    *,
    include_citations: bool = True,
) -> VisualizationSpec:
    """AggregatedResult -> the public VisualizationSpec. Which of
    result.buckets/network/stat_value is meaningful is determined entirely
    by viz_type, matching how dispatch.py populated the AggregatedResult.
    """
    if viz_type == VizType.NETWORK_GRAPH:
        return _build_network_spec(title, result, include_citations=include_citations)
    if viz_type == VizType.STAT_CARD:
        return _build_stat_card_spec(title, result)
    return _build_chart_spec(viz_type, title, result, include_citations=include_citations)


def _build_chart_spec(
    viz_type: VizType, title: str, result: AggregatedResult, *, include_citations: bool
) -> VisualizationSpec:
    # Every non-network, non-stat-card type -- bar/grouped-bar/time-series/
    # histogram/scatter -- shares this one x/y/series DataPoint shape.
    x_type = "temporal" if viz_type == VizType.TIME_SERIES else "nominal"
    encoding = Encoding(
        x=EncodingChannel(field=result.dimension_used or "category", type=x_type),
        y=EncodingChannel(field="count", type="quantitative"),
    )
    data = [
        DataPoint(
            x=bucket.key,
            y=bucket.count,
            series=bucket.series,
            citations=attach_citations(bucket.samples) if include_citations else [],
        )
        for bucket in result.buckets
    ]
    return VisualizationSpec(type=viz_type, title=title, encoding=encoding, data=data)


def _build_network_spec(title: str, result: AggregatedResult, *, include_citations: bool) -> VisualizationSpec:
    network = result.network
    edges = [
        Edge(
            source=edge.source,
            target=edge.target,
            weight=edge.weight,
            citations=attach_citations(edge.samples) if include_citations else [],
        )
        for edge in (network.edges if network else [])
    ]
    encoding = Encoding(
        nodes={"id": "id", "label": "label", "group": "type", "size": "weight"},
        edges={"source": "source", "target": "target", "width": "weight"},
    )
    network_data = NetworkData(nodes=(network.nodes if network else []), edges=edges)
    return VisualizationSpec(
        type=VizType.NETWORK_GRAPH, title=title, encoding=encoding, data=[], network_data=network_data
    )


def _build_stat_card_spec(title: str, result: AggregatedResult) -> VisualizationSpec:
    encoding = Encoding(y=EncodingChannel(field="count", type="quantitative"))
    data = [DataPoint(x="total", y=result.stat_value)]
    return VisualizationSpec(type=VizType.STAT_CARD, title=title, encoding=encoding, data=data)
