import pytest

from app.config import Settings


@pytest.fixture
def no_openai_key(monkeypatch):
    """Simulate a fully unconfigured OpenAI key for both LLM touches.

    Deleting OPENAI_API_KEY from the process environment isn't enough on
    its own: IntentLLMClient/SummaryLLMClient resolve the key through
    app.config.get_settings(), and pydantic-settings reads a real .env
    file directly (independent of os.environ) whenever one exists on the
    machine running the tests. `_env_file=None` bypasses that file, and
    the explicit `openai_api_key=None` forces the field regardless of any
    real env var either. `from ... import get_settings` binds a local
    name in each of those two modules, so both need patching individually
    -- patching app.config.get_settings alone would not reach them.
    """
    fake_settings = Settings(_env_file=None, openai_api_key=None)
    monkeypatch.setattr("app.intent.llm_client.get_settings", lambda: fake_settings)
    monkeypatch.setattr("app.services.summary_generator.get_settings", lambda: fake_settings)
    return fake_settings
