from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from databricks_docs_vectordb.embedding.config import Settings
from databricks_docs_vectordb.scraping.config import DATA_DIR_ENV_VAR


@pytest.fixture
def no_dotenv(tmp_path: Path) -> Path:
    """A .env path that doesn't exist, so the repo's real .env is never read."""
    return tmp_path / "missing.env"


def load(no_dotenv: Path, environ: dict[str, str] | None = None, **overrides):
    return Settings.load(environ=environ or {}, dotenv_path=no_dotenv, **overrides)


def test_data_dir_defaults_to_data(no_dotenv):
    assert load(no_dotenv).data_dir == Path("data")


def test_data_dir_env_override_and_derived_paths(no_dotenv, tmp_path):
    root = tmp_path / "d"
    settings = load(no_dotenv, environ={DATA_DIR_ENV_VAR: str(root)})
    assert settings.data_dir == root
    assert settings.parsed_path == root / "parsed.sqlite"
    assert settings.embeddings_path == root / "embeddings.sqlite"


def test_data_dir_shares_the_env_var_with_parsing(no_dotenv, tmp_path):
    """Phase 2 and Phase 3 read/write under the same data/ directory."""
    from databricks_docs_vectordb.parsing.config import Settings as ParsingSettings

    root = tmp_path / "d"
    environ = {DATA_DIR_ENV_VAR: str(root)}
    embedding_settings = load(no_dotenv, environ=environ)
    parsing_settings = ParsingSettings.load(environ=environ, dotenv_path=no_dotenv)
    assert embedding_settings.data_dir == parsing_settings.data_dir
    assert embedding_settings.parsed_path == parsing_settings.parsed_path


def test_real_environment_overrides_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{DATA_DIR_ENV_VAR}=from-dotenv\n")
    settings = Settings.load(
        environ={DATA_DIR_ENV_VAR: "from-env"}, dotenv_path=env_file
    )
    assert settings.data_dir == Path("from-env")


def test_defaults_match_the_design_doc():
    settings = Settings()
    assert settings.model == "text-embedding-3-small"
    assert settings.model_token_limit == 8191
    assert settings.split_overlap_tokens == 200
    assert settings.batch_size == 100
    assert settings.batch_token_budget == 200_000
    assert settings.rate_rpm == 500.0
    assert settings.rate_tpm == 1_000_000.0
    assert settings.concurrency == 4
    assert settings.max_attempts == 5
    assert settings.backoff_base == 1.0
    assert settings.backoff_max == 60.0
    assert settings.retry_after_max == 300.0
    assert settings.breaker_threshold == 10
    assert settings.breaker_cooldown == 600.0
    assert settings.breaker_max_trips == 3


def test_loading_requires_no_provider_api_key(no_dotenv, monkeypatch):
    # Only actually calling the provider (Step 5) needs one.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = load(no_dotenv, environ={})
    assert settings.model == "text-embedding-3-small"


def test_overrides_are_applied(no_dotenv):
    settings = load(
        no_dotenv,
        model="text-embedding-3-large",
        model_token_limit=8191,
        batch_size=50,
        rate_rpm=100.0,
    )
    assert settings.model == "text-embedding-3-large"
    assert settings.batch_size == 50
    assert settings.rate_rpm == 100.0


def test_none_overrides_are_ignored(no_dotenv):
    settings = load(no_dotenv, model=None, batch_size=None)
    assert settings.model == "text-embedding-3-small"
    assert settings.batch_size == 100


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": ""},
        {"model_token_limit": 0},
        {"model_token_limit": -1},
        {"split_overlap_tokens": -1},
        {"split_overlap_tokens": 8191},  # equal to model_token_limit: rejected
        {"split_overlap_tokens": 9000},  # bigger than model_token_limit
        {"batch_size": 0},
        {"batch_token_budget": 0},
        {"rate_rpm": 0},
        {"rate_tpm": 0},
        {"concurrency": 0},
        {"max_attempts": 0},
        {"backoff_base": 0},
        {"backoff_max": 0},
        {"retry_after_max": 0},
        {"breaker_threshold": 0},
        {"breaker_cooldown": -1},
        {"breaker_max_trips": 0},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        Settings(**overrides)


def test_zero_overlap_and_zero_cooldown_are_allowed():
    settings = Settings(split_overlap_tokens=0, breaker_cooldown=0)
    assert settings.split_overlap_tokens == 0
    assert settings.breaker_cooldown == 0


def test_settings_are_frozen():
    settings = Settings()
    with pytest.raises(FrozenInstanceError):
        settings.model = "other"  # type: ignore[misc]
