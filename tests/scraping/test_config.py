import os
from dataclasses import FrozenInstanceError
from importlib.metadata import version
from pathlib import Path

import pytest

from databricks_docs_vectordb.scraping.config import (
    CONTACT_ENV_VAR,
    DATA_DIR_ENV_VAR,
    SOURCES,
    Settings,
)

CONTACT = "me@example.com"


@pytest.fixture
def no_dotenv(tmp_path: Path) -> Path:
    """A .env path that doesn't exist, so the repo's real .env is never read."""
    return tmp_path / "missing.env"


def load(no_dotenv: Path, environ: dict[str, str] | None = None, **overrides):
    if environ is None:
        environ = {CONTACT_ENV_VAR: CONTACT}
    return Settings.load(environ=environ, dotenv_path=no_dotenv, **overrides)


def make(**overrides) -> Settings:
    return Settings(contact=CONTACT, **overrides)


def test_user_agent_includes_version_and_contact(no_dotenv):
    settings = load(no_dotenv)
    assert settings.user_agent == (
        f"databricks-docs-vectordb/{version('databricks-docs-vectordb')} (+{CONTACT})"
    )


def test_missing_contact_raises_naming_the_variable(no_dotenv):
    with pytest.raises(RuntimeError, match=CONTACT_ENV_VAR):
        load(no_dotenv, environ={})


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_contact_raises(no_dotenv, value):
    with pytest.raises(RuntimeError, match=CONTACT_ENV_VAR):
        load(no_dotenv, environ={CONTACT_ENV_VAR: value})


def test_multiline_contact_raises(no_dotenv):
    with pytest.raises(RuntimeError, match="single line"):
        load(no_dotenv, environ={CONTACT_ENV_VAR: "a@b.com\r\nX-Evil: 1"})


def test_contact_is_read_from_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f'{CONTACT_ENV_VAR}="from-dotenv@example.com"\n')
    settings = Settings.load(environ={}, dotenv_path=env_file)
    assert settings.contact == "from-dotenv@example.com"


def test_real_environment_overrides_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{CONTACT_ENV_VAR}=from-dotenv@example.com\n")
    settings = Settings.load(
        environ={CONTACT_ENV_VAR: "from-env@example.com"}, dotenv_path=env_file
    )
    assert settings.contact == "from-env@example.com"


def test_default_environment_is_os_environ(monkeypatch, no_dotenv):
    monkeypatch.setenv(CONTACT_ENV_VAR, "from-os@example.com")
    settings = Settings.load(dotenv_path=no_dotenv)
    assert settings.contact == "from-os@example.com"


def test_loading_does_not_modify_os_environ(monkeypatch, tmp_path):
    monkeypatch.delenv(CONTACT_ENV_VAR, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"{CONTACT_ENV_VAR}=from-dotenv@example.com\n")
    Settings.load(environ={}, dotenv_path=env_file)
    assert CONTACT_ENV_VAR not in os.environ


def test_contact_is_not_in_repr_or_str(no_dotenv):
    settings = load(no_dotenv)
    assert CONTACT not in repr(settings)
    assert CONTACT not in str(settings)


def test_data_dir_defaults_to_data(no_dotenv):
    assert load(no_dotenv).data_dir == Path("data")


def test_data_dir_env_override_and_derived_paths(no_dotenv, tmp_path):
    settings = load(
        no_dotenv,
        environ={CONTACT_ENV_VAR: CONTACT, DATA_DIR_ENV_VAR: str(tmp_path / "d")},
    )
    root = tmp_path / "d"
    assert settings.data_dir == root
    assert settings.raw_dir == root / "raw"
    assert settings.manifest_path == root / "manifest.sqlite"
    assert settings.sitemaps_dir == root / "sitemaps"
    assert settings.logs_dir == root / "logs"
    assert settings.robots_path == root / "robots.txt"


def test_overrides_are_applied(no_dotenv):
    settings = load(no_dotenv, rate=0.5, concurrency=1, breaker_cooldown=1.0)
    assert (settings.rate, settings.concurrency, settings.breaker_cooldown) == (
        0.5,
        1,
        1.0,
    )


def test_none_overrides_are_ignored(no_dotenv):
    settings = load(no_dotenv, rate=None, concurrency=None)
    assert settings.rate == 2.0
    assert settings.concurrency == 4


def test_defaults_match_the_design_doc():
    settings = make()
    assert settings.rate == 2.0
    assert settings.concurrency == 4
    assert (settings.connect_timeout, settings.read_timeout) == (10.0, 30.0)
    assert settings.max_redirects == 5
    assert settings.max_attempts == 5
    assert (
        settings.breaker_threshold,
        settings.breaker_cooldown,
        settings.breaker_max_trips,
    ) == (10, 600.0, 3)
    assert settings.progress_every == 100


def test_sources_cover_aws_en_and_api():
    assert [s.name for s in SOURCES] == ["aws-en", "api"]
    assert make().sources == SOURCES


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.name)
def test_source_urls_are_consistent(source):
    assert source.url_prefix.startswith("https://docs.databricks.com/")
    assert source.url_prefix.endswith("/")
    assert source.sitemap_url.startswith(source.url_prefix)
    assert source.sitemap_url.endswith("/sitemap.xml")


@pytest.mark.parametrize(
    "overrides",
    [
        {"rate": 0},
        {"rate": -1.0},
        {"concurrency": 0},
        {"connect_timeout": 0},
        {"read_timeout": 0},
        {"max_attempts": 0},
        {"backoff_base": 0},
        {"backoff_max": 0},
        {"breaker_threshold": 0},
        {"breaker_max_trips": 0},
        {"progress_every": 0},
        {"max_redirects": -1},
        {"breaker_cooldown": -1},
        {"sources": ()},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        make(**overrides)


def test_zero_cooldown_and_redirects_are_allowed():
    settings = make(breaker_cooldown=0, max_redirects=0)
    assert settings.breaker_cooldown == 0
    assert settings.max_redirects == 0


def test_settings_are_frozen():
    settings = make()
    with pytest.raises(FrozenInstanceError):
        settings.rate = 10.0  # type: ignore[misc]
