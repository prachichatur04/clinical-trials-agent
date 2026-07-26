from app.citations.attach import DEFAULT_MAX_CITATIONS, attach_citations
from app.schemas.internal import Sample


def _sample(nct_id: str, excerpt: str = "excerpt text") -> Sample:
    return Sample(nct_id=nct_id, field_path="protocolSection.statusModule.overallStatus", excerpt=excerpt)


def test_empty_samples_yields_no_citations():
    assert attach_citations([]) == []


def test_builds_citation_with_url_and_verbatim_excerpt():
    citations = attach_citations([_sample("NCT123", "RECRUITING")])
    assert len(citations) == 1
    assert citations[0].nct_id == "NCT123"
    assert citations[0].excerpt == "RECRUITING"
    assert citations[0].url == "https://clinicaltrials.gov/study/NCT123"
    assert citations[0].field_path == "protocolSection.statusModule.overallStatus"


def test_caps_at_max_citations():
    samples = [_sample(f"NCT{i}") for i in range(10)]
    citations = attach_citations(samples, max_citations=3)
    assert len(citations) == 3


def test_default_max_citations_is_three():
    samples = [_sample(f"NCT{i}") for i in range(10)]
    citations = attach_citations(samples)
    assert len(citations) == DEFAULT_MAX_CITATIONS


def test_fewer_samples_than_max_returns_all():
    samples = [_sample("NCT1"), _sample("NCT2")]
    citations = attach_citations(samples, max_citations=5)
    assert len(citations) == 2


def test_deterministic_ordering_by_nct_id():
    samples = [_sample("NCT999"), _sample("NCT111"), _sample("NCT555")]
    citations = attach_citations(samples, max_citations=3)
    assert [c.nct_id for c in citations] == ["NCT111", "NCT555", "NCT999"]


def test_same_input_always_produces_same_output():
    samples = [_sample("NCT3"), _sample("NCT1"), _sample("NCT2")]
    first = attach_citations(samples)
    second = attach_citations(samples)
    assert first == second


def test_duplicate_nct_id_deduped_to_a_single_citation():
    samples = [_sample("NCT1", "first excerpt"), _sample("NCT1", "second excerpt")]
    citations = attach_citations(samples)
    assert len(citations) == 1
    assert citations[0].excerpt == "first excerpt"


def test_dedup_does_not_prevent_other_studies_from_being_cited():
    samples = [_sample("NCT1"), _sample("NCT1"), _sample("NCT2")]
    citations = attach_citations(samples, max_citations=5)
    assert {c.nct_id for c in citations} == {"NCT1", "NCT2"}
