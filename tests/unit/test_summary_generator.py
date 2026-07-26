from types import SimpleNamespace

import pytest

from app.schemas.intent import AnalysisType
from app.schemas.internal import AggregatedResult, Bucket, EdgeSamples, NetworkResult
from app.services.summary_generator import SummaryLLMClient, generate_summary


class _FakeCompletions:
    def __init__(self, content_or_error, captured_calls: list):
        self._content_or_error = content_or_error
        self._captured_calls = captured_calls

    async def create(self, **kwargs):
        self._captured_calls.append(kwargs)
        if isinstance(self._content_or_error, Exception):
            raise self._content_or_error
        message = SimpleNamespace(content=self._content_or_error)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_openai_client(content_or_error):
    captured_calls: list = []
    fake = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(content_or_error, captured_calls)))
    return fake, captured_calls


async def test_missing_api_key_returns_none(no_openai_key):
    result = await generate_summary(
        "query", AnalysisType.DISTRIBUTION, AggregatedResult(buckets=[Bucket(key="Phase 1", count=5)]), 5, 5
    )
    assert result is None


async def test_successful_summary_returned_stripped():
    fake_client, _ = _fake_openai_client("  Phase 1 dominates with 5 studies.  \n")
    llm_client = SummaryLLMClient(client=fake_client)

    result = await generate_summary(
        "query",
        AnalysisType.DISTRIBUTION,
        AggregatedResult(buckets=[Bucket(key="Phase 1", count=5)]),
        5,
        5,
        llm_client=llm_client,
    )

    assert result == "Phase 1 dominates with 5 studies."


async def test_llm_failure_returns_none_not_raises():
    fake_client, _ = _fake_openai_client(RuntimeError("connection reset"))
    llm_client = SummaryLLMClient(client=fake_client)

    result = await generate_summary(
        "query", AnalysisType.DISTRIBUTION, AggregatedResult(buckets=[Bucket(key="Phase 1", count=5)]), 5, 5, llm_client=llm_client
    )

    assert result is None


async def test_prompt_includes_query_and_analysis_type_and_counts():
    fake_client, captured_calls = _fake_openai_client("summary")
    llm_client = SummaryLLMClient(client=fake_client)

    await generate_summary(
        "How are trials distributed?",
        AnalysisType.DISTRIBUTION,
        AggregatedResult(buckets=[Bucket(key="Phase 1", count=5)]),
        total_matched=812,
        total_fetched=500,
        llm_client=llm_client,
    )

    user_message = captured_calls[0]["messages"][1]["content"]
    assert "How are trials distributed?" in user_message
    assert "distribution" in user_message
    assert "812" in user_message
    assert "500" in user_message


async def test_prompt_includes_top_bucket_results():
    fake_client, captured_calls = _fake_openai_client("summary")
    llm_client = SummaryLLMClient(client=fake_client)

    await generate_summary(
        "query",
        AnalysisType.DISTRIBUTION,
        AggregatedResult(buckets=[Bucket(key="Phase 1", count=12), Bucket(key="Phase 2", count=8)]),
        20,
        20,
        llm_client=llm_client,
    )

    user_message = captured_calls[0]["messages"][1]["content"]
    assert "Phase 1: 12" in user_message
    assert "Phase 2: 8" in user_message


async def test_bucket_prompt_limited_to_top_10():
    fake_client, captured_calls = _fake_openai_client("summary")
    llm_client = SummaryLLMClient(client=fake_client)
    buckets = [Bucket(key=f"Bucket{i}", count=20 - i) for i in range(15)]

    await generate_summary("query", AnalysisType.DISTRIBUTION, AggregatedResult(buckets=buckets), 100, 100, llm_client=llm_client)

    user_message = captured_calls[0]["messages"][1]["content"]
    assert "Bucket9" in user_message
    assert "Bucket10" not in user_message


async def test_comparison_series_label_shown_in_prompt():
    fake_client, captured_calls = _fake_openai_client("summary")
    llm_client = SummaryLLMClient(client=fake_client)
    buckets = [Bucket(key="Phase 1", count=5, series="Keytruda"), Bucket(key="Phase 1", count=3, series="Opdivo")]

    await generate_summary("query", AnalysisType.COMPARISON, AggregatedResult(buckets=buckets), 8, 8, llm_client=llm_client)

    user_message = captured_calls[0]["messages"][1]["content"]
    assert "Phase 1 [Keytruda]: 5" in user_message
    assert "Phase 1 [Opdivo]: 3" in user_message


async def test_network_prompt_includes_top_edges_not_buckets():
    fake_client, captured_calls = _fake_openai_client("summary")
    llm_client = SummaryLLMClient(client=fake_client)
    network = NetworkResult(edges=[EdgeSamples(source="s_pfizer", target="d_x", weight=7)])

    await generate_summary(
        "query", AnalysisType.NETWORK, AggregatedResult(network=network), 50, 50, llm_client=llm_client
    )

    user_message = captured_calls[0]["messages"][1]["content"]
    assert "s_pfizer -> d_x: 7" in user_message
    assert "Results (top 10 buckets)" not in user_message


async def test_count_prompt_includes_stat_value_not_buckets():
    fake_client, captured_calls = _fake_openai_client("summary")
    llm_client = SummaryLLMClient(client=fake_client)

    await generate_summary(
        "How many trials?", AnalysisType.COUNT, AggregatedResult(stat_value=8123), 8123, 0, llm_client=llm_client
    )

    user_message = captured_calls[0]["messages"][1]["content"]
    assert "Total count: 8123" in user_message


async def test_prompt_includes_assumptions_when_present():
    fake_client, captured_calls = _fake_openai_client("summary")
    llm_client = SummaryLLMClient(client=fake_client)
    aggregated = AggregatedResult(
        buckets=[Bucket(key="Phase 1", count=5)],
        assumptions=["4 studies have a missing start date -> 'unknown' bucket."],
    )

    await generate_summary("query", AnalysisType.DISTRIBUTION, aggregated, 5, 5, llm_client=llm_client)

    user_message = captured_calls[0]["messages"][1]["content"]
    assert "Data quality notes:" in user_message
    assert "missing start date" in user_message


async def test_prompt_omits_quality_section_when_no_assumptions():
    fake_client, captured_calls = _fake_openai_client("summary")
    llm_client = SummaryLLMClient(client=fake_client)

    await generate_summary(
        "query", AnalysisType.DISTRIBUTION, AggregatedResult(buckets=[Bucket(key="Phase 1", count=5)]), 5, 5, llm_client=llm_client
    )

    user_message = captured_calls[0]["messages"][1]["content"]
    assert "Data quality notes:" not in user_message


def test_summary_client_missing_api_key_and_no_client_raises(no_openai_key):
    from app.intent.llm_client import LLMUnavailableError

    with pytest.raises(LLMUnavailableError):
        SummaryLLMClient()
