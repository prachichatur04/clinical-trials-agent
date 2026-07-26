from app.analysis.network import MAX_NETWORK_NODES, build_cooccurrence_graph
from app.schemas.intent import AnalysisType, Confidence, Entities, Intent, VizType
from app.schemas.internal import TrialRecord


def _record(nct_id: str, **overrides) -> TrialRecord:
    defaults = {"nct_id": nct_id}
    defaults.update(overrides)
    return TrialRecord(**defaults)


def _intent(dimension: str | None = None) -> Intent:
    return Intent(
        analysis_type=AnalysisType.NETWORK,
        entities=Entities(dimension=dimension),
        suggested_viz=VizType.NETWORK_GRAPH,
        query_plan="plan",
        notes="notes",
        confidence=Confidence.HIGH,
    )


# --- sponsor<->drug bipartite (default) -------------------------------------


def test_sponsor_drug_network_builds_nodes_and_edge():
    records = [
        _record("NCT1", lead_sponsor_name="Pfizer", intervention_types=["DRUG"], intervention_names=["Pembrolizumab"])
    ]
    result = build_cooccurrence_graph(records, _intent())

    node_ids = {n.id for n in result.nodes}
    assert node_ids == {"s_pfizer", "d_pembrolizumab"}
    assert result.edges[0].source == "s_pfizer"
    assert result.edges[0].target == "d_pembrolizumab"
    assert result.edges[0].weight == 1


def test_sponsor_drug_edge_weight_accumulates_across_studies():
    records = [
        _record("NCT1", lead_sponsor_name="Pfizer", intervention_types=["DRUG"], intervention_names=["Pembrolizumab"]),
        _record("NCT2", lead_sponsor_name="Pfizer", intervention_types=["DRUG"], intervention_names=["Pembrolizumab"]),
    ]
    result = build_cooccurrence_graph(records, _intent())
    assert result.edges[0].weight == 2
    assert len(result.edges[0].samples) == 2


def test_missing_sponsor_or_drug_excludes_record():
    records = [
        _record("NCT1", intervention_types=["DRUG"], intervention_names=["X"]),  # no sponsor
        _record("NCT2", lead_sponsor_name="Pfizer"),  # no drug intervention
    ]
    result = build_cooccurrence_graph(records, _intent())
    assert result.nodes == []
    assert result.edges == []


def test_non_drug_interventions_excluded_from_sponsor_drug_network():
    records = [
        _record(
            "NCT1",
            lead_sponsor_name="Pfizer",
            intervention_types=["PROCEDURE"],
            intervention_names=["Surgery"],
        )
    ]
    result = build_cooccurrence_graph(records, _intent())
    assert result.nodes == []


def test_dosage_suffix_normalized_so_variants_merge():
    records = [
        _record("NCT1", lead_sponsor_name="Pfizer", intervention_types=["DRUG"], intervention_names=["Drug X 200mg"]),
        _record("NCT2", lead_sponsor_name="Pfizer", intervention_types=["DRUG"], intervention_names=["Drug X"]),
    ]
    result = build_cooccurrence_graph(records, _intent())
    drug_nodes = [n for n in result.nodes if n.type == "drug"]
    assert len(drug_nodes) == 1
    assert drug_nodes[0].weight == 2


def test_node_nct_ids_track_which_studies_contributed():
    records = [
        _record("NCT1", lead_sponsor_name="Pfizer", intervention_types=["DRUG"], intervention_names=["X"]),
        _record("NCT2", lead_sponsor_name="Pfizer", intervention_types=["DRUG"], intervention_names=["X"]),
    ]
    result = build_cooccurrence_graph(records, _intent())
    sponsor_node = next(n for n in result.nodes if n.type == "sponsor")
    assert sponsor_node.nct_ids == ["NCT1", "NCT2"]


# --- drug<->drug co-occurrence -----------------------------------------------


def test_drug_cooccurrence_mode_selected_by_dimension():
    records = [_record("NCT1", intervention_types=["DRUG", "DRUG"], intervention_names=["A", "B"])]
    result = build_cooccurrence_graph(records, _intent(dimension="drug_cooccurrence"))
    assert {n.id for n in result.nodes} == {"d_a", "d_b"}
    assert len(result.edges) == 1


def test_single_drug_study_excluded_from_cooccurrence_edges_with_note():
    records = [_record("NCT1", intervention_types=["DRUG"], intervention_names=["A"])]
    result = build_cooccurrence_graph(records, _intent(dimension="drug_cooccurrence"))
    assert result.edges == []
    assert "fewer than 2 drug interventions" in result.notes[0]


def test_zero_drug_study_also_excluded_from_cooccurrence():
    records = [_record("NCT1", intervention_types=["PROCEDURE"], intervention_names=["Surgery"])]
    result = build_cooccurrence_graph(records, _intent(dimension="drug_cooccurrence"))
    assert result.nodes == []
    assert result.edges == []


def test_three_drug_study_creates_all_pairwise_edges():
    records = [_record("NCT1", intervention_types=["DRUG"] * 3, intervention_names=["A", "B", "C"])]
    result = build_cooccurrence_graph(records, _intent(dimension="drug_cooccurrence"))
    edge_pairs = {(e.source, e.target) for e in result.edges}
    assert edge_pairs == {("d_a", "d_b"), ("d_a", "d_c"), ("d_b", "d_c")}


def test_cooccurrence_edge_weight_accumulates_across_studies():
    records = [
        _record("NCT1", intervention_types=["DRUG", "DRUG"], intervention_names=["A", "B"]),
        _record("NCT2", intervention_types=["DRUG", "DRUG"], intervention_names=["B", "A"]),  # order-independent
    ]
    result = build_cooccurrence_graph(records, _intent(dimension="drug_cooccurrence"))
    assert len(result.edges) == 1
    assert result.edges[0].weight == 2


# --- capping -----------------------------------------------------------


def test_node_count_capped_at_max_network_nodes():
    records = [
        _record(f"NCT{i}", lead_sponsor_name=f"Sponsor{i}", intervention_types=["DRUG"], intervention_names=[f"Drug{i}"])
        for i in range(60)
    ]
    result = build_cooccurrence_graph(records, _intent())
    assert len(result.nodes) <= MAX_NETWORK_NODES


def test_high_weight_nodes_survive_capping():
    # One sponsor/drug pair repeated many times should always survive the
    # cap over 60 one-off sponsor/drug pairs with weight 1.
    records = [
        _record(f"NCT-common-{i}", lead_sponsor_name="BigPharma", intervention_types=["DRUG"], intervention_names=["Big Drug"])
        for i in range(50)
    ] + [
        _record(f"NCT{i}", lead_sponsor_name=f"Sponsor{i}", intervention_types=["DRUG"], intervention_names=[f"Drug{i}"])
        for i in range(60)
    ]
    result = build_cooccurrence_graph(records, _intent())
    node_ids = {n.id for n in result.nodes}
    assert "s_bigpharma" in node_ids
    assert "d_big_drug" in node_ids


def test_edges_between_capped_out_nodes_are_dropped_not_dangling():
    records = [
        _record(f"NCT-common-{i}", lead_sponsor_name="BigPharma", intervention_types=["DRUG"], intervention_names=["Big Drug"])
        for i in range(50)
    ] + [
        _record(f"NCT{i}", lead_sponsor_name=f"Sponsor{i}", intervention_types=["DRUG"], intervention_names=[f"Drug{i}"])
        for i in range(60)
    ]
    result = build_cooccurrence_graph(records, _intent())
    node_ids = {n.id for n in result.nodes}
    for edge in result.edges:
        assert edge.source in node_ids
        assert edge.target in node_ids


def test_empty_records_yields_empty_network():
    result = build_cooccurrence_graph([], _intent())
    assert result.nodes == []
    assert result.edges == []
    assert result.notes == []
