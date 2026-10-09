import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from databricks_docs_vectordb.indexing.config import (
    DEFAULT_QDRANT_URL,
    QDRANT_API_KEY_ENV_VAR,
    QDRANT_URL_ENV_VAR,
    Settings,
)
from databricks_docs_vectordb.scraping.config import DATA_DIR_ENV_VAR

API_KEY = "test-qdrant-key-7f3a"


@pytest.fixture
def no_dotenv(tmp_path: Path) -> Path:
    """A .env path that doesn't exist, so the repo's real .env is never read."""
    return tmp_path / "missing.env"


def load(no_dotenv: Path, environ: dict[str, str] | None = None, **overrides):
    if environ is None:
        environ = {QDRANT_API_KEY_ENV_VAR: API_KEY}
    return Settings.load(environ=environ, dotenv_path=no_dotenv, **overrides)


def make(**overrides) -> Settings:
    return Settings(api_key=API_KEY, **overrides)


def test_missing_api_key_raises_naming_the_variable(no_dotenv):
    with pytest.raises(RuntimeError, match=QDRANT_API_KEY_ENV_VAR):
        load(no_dotenv, environ={})


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_api_key_raises(no_dotenv, value):
    with pytest.raises(RuntimeError, match=QDRANT_API_KEY_ENV_VAR):
        load(no_dotenv, environ={QDRANT_API_KEY_ENV_VAR: value})


def test_empty_api_key_in_dotenv_raises(tmp_path):
    # An .env copied from .env.example without filling in the key.
    env_file = tmp_path / ".env"
    env_file.write_text(f'{QDRANT_API_KEY_ENV_VAR}=""\n')
    with pytest.raises(RuntimeError, match=QDRANT_API_KEY_ENV_VAR):
        Settings.load(environ={}, dotenv_path=env_file)


def test_api_key_is_not_in_repr_or_str(no_dotenv):
    settings = load(no_dotenv)
    assert settings.api_key == API_KEY
    assert API_KEY not in repr(settings)
    assert API_KEY not in str(settings)


def test_api_key_is_read_from_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f'{QDRANT_API_KEY_ENV_VAR}="from-dotenv"\n')
    settings = Settings.load(environ={}, dotenv_path=env_file)
    assert settings.api_key == "from-dotenv"


def test_real_environment_overrides_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{QDRANT_API_KEY_ENV_VAR}=from-dotenv\n"
        f"{QDRANT_URL_ENV_VAR}=http://from-dotenv:6333\n"
    )
    settings = Settings.load(
        environ={
            QDRANT_API_KEY_ENV_VAR: "from-env",
            QDRANT_URL_ENV_VAR: "http://from-env:6333",
        },
        dotenv_path=env_file,
    )
    assert settings.api_key == "from-env"
    assert settings.url == "http://from-env:6333"


def test_default_environment_is_os_environ(monkeypatch, no_dotenv):
    monkeypatch.setenv(QDRANT_API_KEY_ENV_VAR, "from-os")
    settings = Settings.load(dotenv_path=no_dotenv)
    assert settings.api_key == "from-os"


def test_loading_does_not_modify_os_environ(monkeypatch, tmp_path):
    monkeypatch.delenv(QDRANT_API_KEY_ENV_VAR, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{QDRANT_API_KEY_ENV_VAR}=from-dotenv\n")
    Settings.load(environ={}, dotenv_path=env_file)
    assert QDRANT_API_KEY_ENV_VAR not in os.environ


def test_url_defaults_to_the_compose_service(no_dotenv):
    assert load(no_dotenv).url == DEFAULT_QDRANT_URL == "http://qdrant:6333"


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_url_falls_back_to_the_default(no_dotenv, value):
    settings = load(
        no_dotenv,
        environ={QDRANT_API_KEY_ENV_VAR: API_KEY, QDRANT_URL_ENV_VAR: value},
    )
    assert settings.url == DEFAULT_QDRANT_URL


def test_url_env_override(no_dotenv):
    settings = load(
        no_dotenv,
        environ={
            QDRANT_API_KEY_ENV_VAR: API_KEY,
            QDRANT_URL_ENV_VAR: "http://localhost:6333",
        },
    )
    assert settings.url == "http://localhost:6333"


def test_data_dir_defaults_to_data(no_dotenv):
    assert load(no_dotenv).data_dir == Path("data")


def test_data_dir_env_override_and_derived_paths(no_dotenv, tmp_path):
    root = tmp_path / "d"
    settings = load(
        no_dotenv,
        environ={QDRANT_API_KEY_ENV_VAR: API_KEY, DATA_DIR_ENV_VAR: str(root)},
    )
    assert settings.data_dir == root
    assert settings.manifest_path == root / "manifest.sqlite"
    assert settings.parsed_path == root / "parsed.sqlite"
    assert settings.embeddings_path == root / "embeddings.sqlite"
    assert settings.logs_dir == root / "logs"
    assert settings.snapshots_dir == root / "qdrant_snapshots"


def test_data_dir_shares_the_env_var_with_embedding(no_dotenv, tmp_path):
    """Phase 3 writes the embeddings store this phase reads."""
    from databricks_docs_vectordb.embedding.config import (
        Settings as EmbeddingSettings,
    )

    environ = {QDRANT_API_KEY_ENV_VAR: API_KEY, DATA_DIR_ENV_VAR: str(tmp_path)}
    indexing_settings = load(no_dotenv, environ=environ)
    embedding_settings = EmbeddingSettings.load(environ=environ, dotenv_path=no_dotenv)
    assert indexing_settings.embeddings_path == embedding_settings.embeddings_path
    assert indexing_settings.parsed_path == embedding_settings.parsed_path
    assert indexing_settings.logs_dir == embedding_settings.logs_dir


def test_defaults_match_the_design_doc():
    settings = make()
    assert settings.url == "http://qdrant:6333"
    assert settings.collection_prefix == "databricks_docs"
    assert settings.distance == "Cosine"
    assert settings.normalize is True
    assert settings.batch_size == 64
    assert settings.max_attempts == 10
    assert settings.backoff_base == 1.0
    assert settings.backoff_max == 60.0
    assert settings.recall_sample_size == 200
    assert settings.recall_warn_below == 0.95
    assert settings.progress_every == 1000


def test_overrides_are_applied(no_dotenv):
    settings = load(no_dotenv, distance="Dot", normalize=False, batch_size=256)
    assert settings.distance == "Dot"
    assert settings.normalize is False
    assert settings.batch_size == 256


def test_none_overrides_are_ignored(no_dotenv):
    settings = load(no_dotenv, distance=None, batch_size=None)
    assert settings.distance == "Cosine"
    assert settings.batch_size == 64


@pytest.mark.parametrize("distance", ["Euclid", "Manhattan", "cosine", "dot", ""])
def test_distances_other_than_cosine_or_dot_are_rejected(distance):
    with pytest.raises(ValueError, match="distance"):
        make(distance=distance)


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": ""},
        {"url": ""},
        {"collection_prefix": ""},
        {"batch_size": 0},
        {"request_timeout": 0},
        {"max_attempts": 0},
        {"backoff_base": 0},
        {"backoff_max": 0},
        {"recall_sample_size": 0},
        {"recall_warn_below": 0},
        {"recall_warn_below": 1.5},
        {"progress_every": 0},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_invalid_values_are_rejected(overrides):
    kwargs = {"api_key": API_KEY, **overrides}
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_settings_are_frozen():
    settings = make()
    with pytest.raises(FrozenInstanceError):
        settings.distance = "Dot"  # type: ignore[misc]
