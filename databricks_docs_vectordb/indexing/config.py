import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from dotenv import dotenv_values, find_dotenv

from databricks_docs_vectordb.scraping.config import DATA_DIR_ENV_VAR

QDRANT_URL_ENV_VAR = "QDRANT_URL"
QDRANT_API_KEY_ENV_VAR = "QDRANT_API_KEY"

# The Qdrant sidecar's compose service address, as seen from the dev
# container (vector_db_integration.md's "Vector database and hosting").
DEFAULT_QDRANT_URL = "http://qdrant:6333"

# Only similarity metrics: higher is better under both (Q27). Spelled the
# way Qdrant's own Distance values are.
DISTANCES = ("Cosine", "Dot")


@dataclass(frozen=True)
class Settings:
    # Sent to Qdrant on every call; kept out of repr so it can't leak into
    # logs.
    api_key: str = field(repr=False)
    url: str = DEFAULT_QDRANT_URL
    data_dir: Path = Path("data")

    # The embedding model is not set here: it comes from embedding/config.py,
    # so there is one place to change it, and the collection name is built
    # from it (Record and collection schema).
    collection_prefix: str = "databricks_docs"

    # Under "Dot", Qdrant stores vectors as given and normalizes no query, so
    # whatever embeds queries in Phase 5 has to normalize them the same way.
    distance: str = "Cosine"
    normalize: bool = True

    batch_size: int = 64  # qdrant-client's own default (Q33)
    request_timeout: float = 30.0

    # Retry connection errors and timeouts, with exponential backoff and
    # jitter, as in Phases 1 and 3 (Q34).
    max_attempts: int = 10
    backoff_base: float = 1.0
    backoff_max: float = 60.0

    # `index check`'s recall@10 of default search against exact search
    # (Q28).
    recall_sample_size: int = 200
    recall_warn_below: float = 0.95

    progress_every: int = 1000  # log a progress line every N points

    def __post_init__(self) -> None:
        for name in ("api_key", "url", "collection_prefix"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if self.distance not in DISTANCES:
            raise ValueError(
                f"distance must be one of {DISTANCES}, got {self.distance!r}"
            )
        positive = (
            "batch_size",
            "request_timeout",
            "max_attempts",
            "backoff_base",
            "backoff_max",
            "recall_sample_size",
            "progress_every",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        if not 0 < self.recall_warn_below <= 1:
            raise ValueError(
                f"recall_warn_below must be in (0, 1], got {self.recall_warn_below!r}"
            )

    @property
    def manifest_path(self) -> Path:
        """Phase 1's manifest: pages not `ok` there are removed from Qdrant."""
        return self.data_dir / "manifest.sqlite"

    @property
    def parsed_path(self) -> Path:
        """Phase 2's chunk store, read only for the orphan check."""
        return self.data_dir / "parsed.sqlite"

    @property
    def embeddings_path(self) -> Path:
        """Phase 3's output: the embeddings store this phase loads from."""
        return self.data_dir / "embeddings.sqlite"

    @property
    def logs_dir(self) -> Path:
        """Shared with scraping and embedding: `index run` logs beside them."""
        return self.data_dir / "logs"

    @property
    def snapshots_dir(self) -> Path:
        """Where `index run` saves old collections before replacing them."""
        return self.data_dir / "qdrant_snapshots"

    @classmethod
    def load(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        dotenv_path: Path | None = None,
        **overrides: Any,
    ) -> Self:
        """Build settings from the environment, falling back to `.env`.

        Real environment variables win over `.env`. Nothing is written to
        `os.environ`. Overrides set to None are ignored, so CLI flags that
        were not given can be passed straight through. Shares
        `DATABRICKS_DOCS_DATA_DIR` with the other phases. A missing or blank
        `QDRANT_API_KEY` is a hard error: every `index` command talks to
        Qdrant, and the sidecar refuses calls without it.
        """
        env_file = dotenv_path if dotenv_path is not None else find_dotenv(usecwd=True)
        env = {
            **dotenv_values(env_file),
            **(os.environ if environ is None else environ),
        }

        api_key = (env.get(QDRANT_API_KEY_ENV_VAR) or "").strip()
        if not api_key:
            raise RuntimeError(
                f"{QDRANT_API_KEY_ENV_VAR} is not set. Add it to .env (see "
                ".env.example); the Qdrant sidecar uses it as its API key."
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "url": (env.get(QDRANT_URL_ENV_VAR) or "").strip() or DEFAULT_QDRANT_URL,
            "data_dir": Path(env.get(DATA_DIR_ENV_VAR) or "data"),
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)
