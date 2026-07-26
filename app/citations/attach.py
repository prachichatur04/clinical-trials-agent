from app.schemas.internal import Sample
from app.schemas.viz import Citation

DEFAULT_MAX_CITATIONS = 3


def attach_citations(samples: list[Sample], max_citations: int = DEFAULT_MAX_CITATIONS) -> list[Citation]:
    """Deterministically pick up to `max_citations` samples and turn them
    into deep Citations with a verbatim excerpt and study URL.

    Deduped and sorted by nct_id first: a bucket built from the same
    underlying records always cites the same studies in the same order
    (reproducibility), and a study contributing multiple samples to one
    bucket (e.g. two DRUG interventions counted under the same key) is
    only cited once.
    """
    deduped = _dedupe_by_nct_id(samples)
    chosen = sorted(deduped, key=lambda sample: sample.nct_id)[:max_citations]
    return [
        Citation(
            nct_id=sample.nct_id,
            field_path=sample.field_path,
            excerpt=sample.excerpt,
            url=f"https://clinicaltrials.gov/study/{sample.nct_id}",
        )
        for sample in chosen
    ]


def _dedupe_by_nct_id(samples: list[Sample]) -> list[Sample]:
    seen: set[str] = set()
    deduped: list[Sample] = []
    for sample in samples:
        if sample.nct_id not in seen:
            seen.add(sample.nct_id)
            deduped.append(sample)
    return deduped
