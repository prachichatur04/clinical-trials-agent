import json
from types import SimpleNamespace

import pytest

from app.intent.llm_client import IntentLLMClient, LLMUnavailableError

VALID_INTENT_JSON = json.dumps(
    {
        "analysis_type": "trend",
        "entities": {
            "drug_name": "Pembrolizumab",
            "condition": None,
            "trial_phase": None,
            "sponsor": None,
            "country": None,
            "status": None,
            "start_year": 2015,
            "end_year": None,
            "dimension": None,
            "compare_a": None,
            "compare_b": None,
            "compare_type": None,
        },
        "suggested_viz": "time_series",
        "query_plan": "Fetch pembrolizumab trials, group by year.",
        "notes": "Yearly trend since 2015.",
        "confidence": "high",
    }
)


class _FakeCompletions:
    def __init__(self, content: str, captured_calls: list):
        self._content = content
        self._captured_calls = captured_calls

    async def create(self, **kwargs):
        self._captured_calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_openai_client(content: str):
    captured_calls: list = []
    fake = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(content, captured_calls)))
    return fake, captured_calls


def test_missing_api_key_and_no_client_raises(no_openai_key):
    with pytest.raises(LLMUnavailableError):
        IntentLLMClient()


def test_explicit_client_bypasses_api_key_requirement(no_openai_key):
    fake_client, _ = _fake_openai_client(VALID_INTENT_JSON)
    IntentLLMClient(client=fake_client)  # must not raise


async def test_classify_returns_intent_from_valid_json():
    fake_client, _ = _fake_openai_client(VALID_INTENT_JSON)
    client = IntentLLMClient(client=fake_client)

    intent = await client.classify("How has pembrolizumab trended since 2015?")

    assert intent.analysis_type.value == "trend"
    assert intent.entities.drug_name == "Pembrolizumab"
    assert intent.entities.start_year == 2015


async def test_classify_raises_on_malformed_json():
    fake_client, _ = _fake_openai_client("not valid json")
    client = IntentLLMClient(client=fake_client)

    with pytest.raises(ValueError):
        await client.classify("some query")


async def test_classify_raises_on_json_missing_required_fields():
    fake_client, _ = _fake_openai_client(json.dumps({"analysis_type": "trend"}))
    client = IntentLLMClient(client=fake_client)

    with pytest.raises(ValueError):  # pydantic.ValidationError is a ValueError subclass
        await client.classify("some query")


async def test_classify_sends_system_prompt_and_query():
    fake_client, captured_calls = _fake_openai_client(VALID_INTENT_JSON)
    client = IntentLLMClient(client=fake_client)

    await client.classify("How has pembrolizumab trended since 2015?")

    messages = captured_calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "planning component" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "How has pembrolizumab trended since 2015?"


async def test_classify_appends_extra_context_to_user_message():
    fake_client, captured_calls = _fake_openai_client(VALID_INTENT_JSON)
    client = IntentLLMClient(client=fake_client)

    await client.classify("query text", extra_context="Your previous response was invalid.")

    user_message = captured_calls[0]["messages"][1]["content"]
    assert "query text" in user_message
    assert "Your previous response was invalid." in user_message


async def test_classify_uses_strict_json_schema_response_format():
    fake_client, captured_calls = _fake_openai_client(VALID_INTENT_JSON)
    client = IntentLLMClient(client=fake_client)

    await client.classify("query text")

    response_format = captured_calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "intent"
    assert response_format["json_schema"]["strict"] is True
