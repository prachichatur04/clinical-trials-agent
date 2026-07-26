from collections.abc import Callable, Iterator

from app.analysis.extractors import ExtractedValue
from app.schemas.internal import Bucket, Sample, TrialRecord

Extractor = Callable[[TrialRecord], Iterator[ExtractedValue]]


def group_and_count(
    records: list[TrialRecord],
    extractor: Extractor,
    top_n: int | None = None,
) -> list[Bucket]:
    """Run every record through `extractor`, tally counts per key, and keep
    every sample so citations/attach.py can pick from them later. Sorted by
    count descending; `top_n` truncates to the largest buckets only.
    """
    counts: dict[str, int] = {}
    samples: dict[str, list[Sample]] = {}

    for record in records:
        for key, nct_id, field_path, excerpt in extractor(record):
            counts[key] = counts.get(key, 0) + 1
            samples.setdefault(key, []).append(Sample(nct_id=nct_id, field_path=field_path, excerpt=excerpt))

    buckets = [Bucket(key=key, count=count, samples=samples[key]) for key, count in counts.items()]
    buckets.sort(key=lambda b: (-b.count, b.key))

    if top_n is not None:
        buckets = buckets[:top_n]
    return buckets


def compare_groups(
    records_a: list[TrialRecord],
    records_b: list[TrialRecord],
    extractor: Extractor,
    label_a: str,
    label_b: str,
) -> list[Bucket]:
    """Group both record sets independently, then zero-fill across the
    union of keys so every key present on either side has a bucket for
    both series -- a grouped bar chart with a bar missing on one side
    would otherwise look like the data was never fetched.
    """
    buckets_a = {b.key: b for b in group_and_count(records_a, extractor)}
    buckets_b = {b.key: b for b in group_and_count(records_b, extractor)}

    result: list[Bucket] = []
    for key in sorted(set(buckets_a) | set(buckets_b)):
        bucket_a = buckets_a.get(key)
        bucket_b = buckets_b.get(key)
        result.append(
            Bucket(
                key=key,
                count=bucket_a.count if bucket_a else 0,
                series=label_a,
                samples=bucket_a.samples if bucket_a else [],
            )
        )
        result.append(
            Bucket(
                key=key,
                count=bucket_b.count if bucket_b else 0,
                series=label_b,
                samples=bucket_b.samples if bucket_b else [],
            )
        )
    return result
