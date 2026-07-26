import pytest

from app.intent.parser import parse_intent
from app.schemas.intent import AnalysisType, Confidence, Entities, Intent, VizType
from app.schemas.request import QueryRequest
from app.schemas.response import IntentSource


class _StubLLMClient:
    """A minimal stand-in for IntentLLMClient: each call to classify() pops
    the next scripted response (an Intent to return, or an Exception to
    raise), so tests can script exact failure/retry sequences."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def classify(self, query, extra_context=None):
        self.calls.append({"query": query, "extra_context": extra_context})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _make_intent(analysis_type=AnalysisType.TREND, start_year=None, end_year=None, **overrides):
    defaults = {
        "analysis_type": analysis_type,
        "entities": Entities(start_year=start_year, end_year=end_year),
        "suggested_viz": VizType.TIME_SERIES,
        "query_plan": "plan",
        "notes": "notes",
        "confidence": Confidence.HIGH,
    }
    defaults.update(overrides)
    return Intent(**defaults)


async def test_missing_api_key_uses_heuristic_fallback(no_openai_key):
    request = QueryRequest(query="How are trials distributed across phases?")

    result = await parse_intent(request, llm_client=None)

    assert result.source == IntentSource.HEURISTIC_FALLBACK
    assert result.intent.analysis_type == AnalysisType.DISTRIBUTION


async def test_successful_first_attempt_uses_llm_source():
    stub = _StubLLMClient([_make_intent(start_year=2015)])
    request = QueryRequest(query="How has this trended since 2015?")

    result = await parse_intent(request, llm_client=stub)

    assert result.source == IntentSource.LLM
    assert result.intent.entities.start_year == 2015
    assert len(stub.calls) == 1
    assert stub.calls[0]["extra_context"] is None


async def test_malformed_first_response_then_valid_retry_uses_llm_retry_source():
    stub = _StubLLMClient([ValueError("bad json"), _make_intent(start_year=2015)])
    request = QueryRequest(query="How has this trended since 2015?")

    result = await parse_intent(request, llm_client=stub)

    assert result.source == IntentSource.LLM_RETRY
    assert len(stub.calls) == 2
    assert stub.calls[0]["extra_context"] is None
    assert "bad json" in stub.calls[1]["extra_context"]


async def test_insane_year_range_triggers_retry_then_succeeds():
    insane = _make_intent(start_year=2025, end_year=2015)  # start > end
    sane = _make_intent(start_year=2015, end_year=2020)
    stub = _StubLLMClient([insane, sane])
    request = QueryRequest(query="Trend from 2025 to 2015?")

    result = await parse_intent(request, llm_client=stub)

    assert result.source == IntentSource.LLM_RETRY
    assert result.intent.entities.start_year == 2015
    assert "start_year" in stub.calls[1]["extra_context"]


async def test_implausible_year_triggers_retry():
    implausible = _make_intent(start_year=1500)
    sane = _make_intent(start_year=2015)
    stub = _StubLLMClient([implausible, sane])
    request = QueryRequest(query="Trend since forever?")

    result = await parse_intent(request, llm_client=stub)

    assert result.source == IntentSource.LLM_RETRY
    assert result.intent.entities.start_year == 2015


async def test_malformed_twice_falls_back_to_heuristic():
    stub = _StubLLMClient([ValueError("bad json"), ValueError("still bad")])
    request = QueryRequest(query="How are trials distributed across phases?")

    result = await parse_intent(request, llm_client=stub)

    assert result.source == IntentSource.HEURISTIC_FALLBACK
    assert result.intent.analysis_type == AnalysisType.DISTRIBUTION
    assert len(stub.calls) == 2


async def test_non_value_error_exception_also_triggers_fallback():
    # Any LLM-path failure degrades to fallback, not just validation errors --
    # e.g. a transient OpenAI API/network error.
    stub = _StubLLMClient([RuntimeError("connection reset"), RuntimeError("connection reset")])
    request = QueryRequest(query="How are trials distributed across phases?")

    result = await parse_intent(request, llm_client=stub)

    assert result.source == IntentSource.HEURISTIC_FALLBACK


async def test_query_forwarded_unchanged_to_llm_client():
    stub = _StubLLMClient([_make_intent()])
    request = QueryRequest(query="A specific question about trial phases?")

    await parse_intent(request, llm_client=stub)

    assert stub.calls[0]["query"] == "A specific question about trial phases?"


@pytest.mark.parametrize("year", [1500, 2200])
def test_check_sane_rejects_implausible_years(year):
    from app.intent.parser import IntentSanityError, _check_sane

    intent = _make_intent(start_year=year)
    with pytest.raises(IntentSanityError):
        _check_sane(intent)


def test_check_sane_accepts_none_years():
    from app.intent.parser import _check_sane

    _check_sane(_make_intent())  # must not raise
