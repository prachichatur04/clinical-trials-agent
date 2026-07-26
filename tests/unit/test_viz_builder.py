from app.schemas.intent import VizType
from app.schemas.internal import AggregatedResult, Bucket, EdgeSamples, NetworkResult, Sample
from app.schemas.viz import Node
from app.viz.builder import build_visualization


def _sample(nct_id: str) -> Sample:
    return Sample(nct_id=nct_id, field_path="protocolSection.statusModule.overallStatus", excerpt="RECRUITING")


# --- bar / time-series charts -------------------------------------------


def test_bar_chart_builds_data_points_from_buckets():
    result = AggregatedResult(
        buckets=[Bucket(key="Phase 1", count=5, samples=[_sample("NCT1")])],
        dimension_used="phase",
    )
    spec = build_visualization(VizType.BAR_CHART, "Trials by phase", result)

    assert spec.type == VizType.BAR_CHART
    assert spec.title == "Trials by phase"
    assert spec.encoding.x.field == "phase"
    assert spec.encoding.x.type == "nominal"
    assert spec.data[0].x == "Phase 1"
    assert spec.data[0].y == 5
    assert spec.data[0].citations[0].nct_id == "NCT1"


def test_time_series_uses_temporal_encoding_type():
    result = AggregatedResult(buckets=[Bucket(key="2015", count=3)], dimension_used="year")
    spec = build_visualization(VizType.TIME_SERIES, "Trend", result)
    assert spec.encoding.x.type == "temporal"


def test_grouped_bar_chart_preserves_series_label():
    result = AggregatedResult(
        buckets=[
            Bucket(key="Phase 1", count=2, series="Keytruda"),
            Bucket(key="Phase 1", count=0, series="Opdivo"),
        ],
        dimension_used="phase",
    )
    spec = build_visualization(VizType.GROUPED_BAR_CHART, "Compare", result)
    series_values = {dp.series for dp in spec.data}
    assert series_values == {"Keytruda", "Opdivo"}


def test_include_citations_false_yields_no_citations():
    result = AggregatedResult(buckets=[Bucket(key="Phase 1", count=1, samples=[_sample("NCT1")])])
    spec = build_visualization(VizType.BAR_CHART, "Trials", result, include_citations=False)
    assert spec.data[0].citations == []


def test_missing_dimension_used_falls_back_to_category():
    result = AggregatedResult(buckets=[Bucket(key="X", count=1)])
    spec = build_visualization(VizType.BAR_CHART, "Trials", result)
    assert spec.encoding.x.field == "category"


def test_empty_buckets_yields_empty_data_not_error():
    result = AggregatedResult(buckets=[])
    spec = build_visualization(VizType.BAR_CHART, "Trials", result)
    assert spec.data == []


# --- network graph -------------------------------------------------------


def test_network_graph_uses_network_data_not_data():
    network = NetworkResult(
        nodes=[Node(id="s_pfizer", label="Pfizer", type="sponsor", weight=3, nct_ids=["NCT1"])],
        edges=[EdgeSamples(source="s_pfizer", target="d_x", weight=2, samples=[_sample("NCT1")])],
    )
    result = AggregatedResult(network=network)

    spec = build_visualization(VizType.NETWORK_GRAPH, "Network", result)

    assert spec.data == []
    assert spec.network_data.nodes[0].id == "s_pfizer"
    assert spec.network_data.edges[0].weight == 2
    assert spec.network_data.edges[0].citations[0].nct_id == "NCT1"


def test_network_graph_encoding_matches_plan_shape():
    result = AggregatedResult(network=NetworkResult())
    spec = build_visualization(VizType.NETWORK_GRAPH, "Network", result)
    assert spec.encoding.nodes == {"id": "id", "label": "label", "group": "type", "size": "weight"}
    assert spec.encoding.edges == {"source": "source", "target": "target", "width": "weight"}


def test_network_graph_handles_none_network_gracefully():
    result = AggregatedResult(network=None)
    spec = build_visualization(VizType.NETWORK_GRAPH, "Network", result)
    assert spec.network_data.nodes == []
    assert spec.network_data.edges == []


def test_network_graph_include_citations_false():
    network = NetworkResult(edges=[EdgeSamples(source="a", target="b", weight=1, samples=[_sample("NCT1")])])
    result = AggregatedResult(network=network)
    spec = build_visualization(VizType.NETWORK_GRAPH, "Network", result, include_citations=False)
    assert spec.network_data.edges[0].citations == []


# --- stat card -----------------------------------------------------------


def test_stat_card_builds_single_data_point_from_stat_value():
    result = AggregatedResult(stat_value=8123)
    spec = build_visualization(VizType.STAT_CARD, "Total trials", result)
    assert spec.type == VizType.STAT_CARD
    assert len(spec.data) == 1
    assert spec.data[0].y == 8123
    assert spec.network_data is None
