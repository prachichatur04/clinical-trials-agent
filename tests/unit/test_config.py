from app.config import Settings, get_settings


def test_defaults_when_no_env_vars_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.openai_api_key is None
    assert settings.ctgov_base_url == "https://clinicaltrials.gov/api/v2"
    assert settings.cors_origins == ["*"]


def test_reads_openai_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    settings = Settings(_env_file=None)
    assert settings.openai_api_key == "sk-test-123"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
