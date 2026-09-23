from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from databricks_docs_vectordb.parsing.config import OVERLAP_MODES, Settings
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
    assert settings.raw_dir == root / "raw"
    assert settings.manifest_path == root / "manifest.sqlite"
    assert settings.parsed_path == root / "parsed.sqlite"


def test_data_dir_shares_the_env_var_with_scraping(no_dotenv, tmp_path):
    """Both phases read/write under the same data/ directory."""
    from databricks_docs_vectordb.scraping.config import Settings as ScrapingSettings

    root = tmp_path / "d"
    environ = {DATA_DIR_ENV_VAR: str(root)}
    parsing_settings = load(no_dotenv, environ=environ)
    scraping_settings = ScrapingSettings.load(
        environ=environ, dotenv_path=no_dotenv, require_contact=False
    )
    assert parsing_settings.data_dir == scraping_settings.data_dir
    assert parsing_settings.raw_dir == scraping_settings.raw_dir
    assert parsing_settings.manifest_path == scraping_settings.manifest_path


def test_real_environment_overrides_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{DATA_DIR_ENV_VAR}=from-dotenv\n")
    settings = Settings.load(
        environ={DATA_DIR_ENV_VAR: "from-env"}, dotenv_path=env_file
    )
    assert settings.data_dir == Path("from-env")


def test_defaults_match_the_design_doc():
    settings = Settings()
    assert settings.target_chunk_size == 3000
    assert settings.overlap_size == 300
    assert settings.overlap_mode == "chars"


def test_overrides_are_applied(no_dotenv):
    settings = load(
        no_dotenv, target_chunk_size=500, overlap_size=50, overlap_mode="heading"
    )
    assert settings.target_chunk_size == 500
    assert settings.overlap_size == 50
    assert settings.overlap_mode == "heading"


def test_none_overrides_are_ignored(no_dotenv):
    settings = load(no_dotenv, target_chunk_size=None, overlap_size=None)
    assert settings.target_chunk_size == 3000
    assert settings.overlap_size == 300


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_chunk_size": 0},
        {"target_chunk_size": -1},
        {"overlap_size": -1},
        {"overlap_size": 3000},  # equal to target_chunk_size: rejected
        {"overlap_size": 4000},  # bigger than target_chunk_size
        {"overlap_mode": "sentences"},  # not a supported mode
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        Settings(**overrides)


def test_zero_overlap_is_allowed():
    settings = Settings(overlap_size=0)
    assert settings.overlap_size == 0


def test_overlap_modes_are_chars_and_heading():
    assert OVERLAP_MODES == ("chars", "heading")


def test_settings_are_frozen():
    settings = Settings()
    with pytest.raises(FrozenInstanceError):
        settings.target_chunk_size = 10  # type: ignore[misc]


def test_loading_does_not_require_a_contact_address(no_dotenv):
    # Parsing never touches the network, so it has no contact concept at all.
    settings = load(no_dotenv, environ={})
    assert not hasattr(settings, "contact")
