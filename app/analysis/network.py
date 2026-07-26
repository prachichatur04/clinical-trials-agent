import re
from collections import defaultdict
from itertools import combinations

from app.schemas.intent import Intent
from app.schemas.internal import EdgeSamples, NetworkResult, Sample, TrialRecord
from app.schemas.viz import Node

MAX_NETWORK_NODES = 40

_DOSAGE_SUFFIX_RE = re.compile(r"\s*\(?\d+(\.\d+)?\s*(mg|mcg|g|ml|iu)\b.*$", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def build_cooccurrence_graph(records: list[TrialRecord], intent: Intent) -> NetworkResult:
    """Two network modes, per the plan:

    - drug<->drug co-occurrence: set intent.entities.dimension to
      "drug_cooccurrence" (the heuristic/LLM path does this for queries
      like "which drugs co-occur in combination studies").
    - sponsor<->drug bipartite (default): everything else.
    """
    if intent.entities.dimension == "drug_cooccurrence":
        return _build_drug_cooccurrence_network(records)
    return _build_sponsor_drug_network(records)


def _build_sponsor_drug_network(records: list[TrialRecord]) -> NetworkResult:
    builder = _GraphBuilder()

    for record in records:
        sponsor_name = record.lead_sponsor_name
        drug_names = _drug_names(record)
        if not sponsor_name or not drug_names:
            continue

        sponsor_id = builder.add_node(f"s_{_slug(sponsor_name)}", sponsor_name, "sponsor", record.nct_id)
        for drug_name in drug_names:
            drug_id = builder.add_node(f"d_{_slug(drug_name)}", drug_name, "drug", record.nct_id)
            builder.add_edge(
                sponsor_id,
                drug_id,
                Sample(
                    nct_id=record.nct_id,
                    field_path="protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
                    excerpt=f"{sponsor_name} sponsoring {drug_name}",
                ),
            )

    return builder.finalize()


def _build_drug_cooccurrence_network(records: list[TrialRecord]) -> NetworkResult:
    builder = _GraphBuilder()
    skipped = 0

    for record in records:
        drug_names = _drug_names(record)
        if len(drug_names) < 2:
            skipped += 1
            continue

        node_ids = [builder.add_node(f"d_{_slug(name)}", name, "drug", record.nct_id) for name in drug_names]
        for (name_a, id_a), (name_b, id_b) in combinations(sorted(zip(drug_names, node_ids), key=lambda p: p[0]), 2):
            builder.add_edge(
                id_a,
                id_b,
                Sample(
                    nct_id=record.nct_id,
                    field_path="protocolSection.armsInterventionsModule.interventions",
                    excerpt=f"{name_a} + {name_b} combination",
                ),
            )

    result = builder.finalize()
    if skipped:
        result.notes.append(
            f"{skipped} studies had fewer than 2 drug interventions and were excluded from co-occurrence edges."
        )
    return result


def _drug_names(record: TrialRecord) -> list[str]:
    names = [
        _normalize_drug_name(name)
        for intervention_type, name in zip(record.intervention_types, record.intervention_names, strict=True)
        if intervention_type == "DRUG"
    ]
    seen: set[str] = set()
    deduped = []
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _normalize_drug_name(name: str) -> str:
    normalized = _DOSAGE_SUFFIX_RE.sub("", name).strip().lower()
    return normalized or name.strip().lower()


def _slug(text: str) -> str:
    return _NON_ALNUM_RE.sub("_", text.lower()).strip("_") or "unknown"


class _GraphBuilder:
    """Accumulates node weights/labels and edge weights/samples, then caps
    to the top-40-by-weight nodes and drops any edge that would dangle."""

    def __init__(self) -> None:
        self._node_weight: dict[str, int] = defaultdict(int)
        self._node_label: dict[str, str] = {}
        self._node_type: dict[str, str] = {}
        self._node_nct_ids: dict[str, set[str]] = defaultdict(set)
        self._edge_weight: dict[tuple[str, str], int] = defaultdict(int)
        self._edge_samples: dict[tuple[str, str], list[Sample]] = defaultdict(list)

    def add_node(self, node_id: str, label: str, node_type: str, nct_id: str) -> str:
        self._node_label[node_id] = label
        self._node_type[node_id] = node_type
        self._node_weight[node_id] += 1
        self._node_nct_ids[node_id].add(nct_id)
        return node_id

    def add_edge(self, source: str, target: str, sample: Sample) -> None:
        key = (source, target)
        self._edge_weight[key] += 1
        self._edge_samples[key].append(sample)

    def finalize(self) -> NetworkResult:
        top_node_ids = {
            node_id
            for node_id, _ in sorted(self._node_weight.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_NETWORK_NODES]
        }
        nodes = [
            Node(
                id=node_id,
                label=self._node_label[node_id],
                type=self._node_type[node_id],
                weight=self._node_weight[node_id],
                nct_ids=sorted(self._node_nct_ids[node_id]),
            )
            for node_id in sorted(top_node_ids)
        ]
        edges = [
            EdgeSamples(source=source, target=target, weight=weight, samples=self._edge_samples[(source, target)])
            for (source, target), weight in sorted(self._edge_weight.items(), key=lambda kv: (-kv[1], kv[0]))
            if source in top_node_ids and target in top_node_ids
        ]
        return NetworkResult(nodes=nodes, edges=edges)
