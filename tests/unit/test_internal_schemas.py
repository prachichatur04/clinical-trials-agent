from app.schemas.intent import VizType
from app.schemas.internal import (
    AggregatedResult,
    Bucket,
    DispatchResult,
    EdgeSamples,
    NetworkResult,
    Sample,
)
from app.schemas.viz import Node


def test_bucket_defaults():
    bucket = Bucket(key="Phase 1", count=3)
    assert bucket.series is None
    assert bucket.samples == []


def test_bucket_with_series_and_samples():
    bucket = Bucket(
        key="Phase 1",
        count=2,
        series="Keytruda",
        samples=[Sample(nct_id="NCT1", field_path="x.y", excerpt="Phase 1 study")],
    )
    assert bucket.series == "Keytruda"
    assert bucket.samples[0].nct_id == "NCT1"


def test_network_result_nodes_reuse_viz_node_directly():
    result = NetworkResult(
        nodes=[Node(id="s_pfizer", label="Pfizer", type="sponsor", weight=3, nct_ids=["NCT1"])],
        edges=[EdgeSamples(source="s_pfizer", target="d_x", weight=1)],
    )
    assert result.nodes[0].label == "Pfizer"
    assert result.edges[0].weight == 1
    assert result.notes == []


def test_aggregated_result_defaults_empty():
    result = AggregatedResult()
    assert result.buckets == []
    assert result.network is None
    assert result.stat_value is None
    assert result.assumptions == []


def test_dispatch_result_bundles_viz_type_title_and_aggregated():
    dispatch_result = DispatchResult(
        viz_type=VizType.BAR_CHART,
        title="Trials by phase",
        aggregated=AggregatedResult(buckets=[Bucket(key="Phase 1", count=5)]),
    )
    assert dispatch_result.viz_type == VizType.BAR_CHART
    assert dispatch_result.aggregated.buckets[0].count == 5
