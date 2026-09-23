import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Self

from dotenv import dotenv_values, find_dotenv

CONTACT_ENV_VAR = "DATABRICKS_DOCS_CONTACT"
DATA_DIR_ENV_VAR = "DATABRICKS_DOCS_DATA_DIR"


@dataclass(frozen=True)
class Source:
    name: str
    sitemap_url: str
    url_prefix: str


SOURCES = (
    Source(
        name="aws-en",
        sitemap_url="https://docs.databricks.com/aws/en/sitemap.xml",
        url_prefix="https://docs.databricks.com/aws/en/",
    ),
    Source(
        name="api",
        sitemap_url="https://docs.databricks.com/api/sitemap.xml",
        url_prefix="https://docs.databricks.com/api/",
    ),
)


@dataclass(frozen=True)
class Settings:
    # Sent to Databricks in the User-Agent; kept out of repr so it can't leak
    # into logs.
    contact: str = field(repr=False)
    data_dir: Path = Path("data")
    sources: tuple[Source, ...] = SOURCES

    rate: float = 2.0  # requests per second, global
    concurrency: int = 4
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_redirects: int = 5

    max_attempts: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    retry_after_max: float = 300.0  # longest Retry-After we'll wait out

    breaker_threshold: int = 10  # consecutive failures before it trips
    breaker_cooldown: float = 600.0
    breaker_max_trips: int = 3

    progress_every: int = 100

    def __post_init__(self) -> None:
        positive = (
            "rate",
            "concurrency",
            "connect_timeout",
            "read_timeout",
            "max_attempts",
            "backoff_base",
            "backoff_max",
            "retry_after_max",
            "breaker_threshold",
            "breaker_max_trips",
            "progress_every",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        for name in ("max_redirects", "breaker_cooldown"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if not self.sources:
            raise ValueError("sources must not be empty")

    @property
    def user_agent(self) -> str:
        return f"databricks-docs-vectordb/{version('databricks-docs-vectordb')} (+{self.contact})"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.sqlite"

    @property
    def sitemaps_dir(self) -> Path:
        return self.data_dir / "sitemaps"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def robots_path(self) -> Path:
        return self.data_dir / "robots.txt"

    @classmethod
    def load(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        dotenv_path: Path | None = None,
        require_contact: bool = True,
        **overrides: Any,
    ) -> Self:
        """Build settings from the environment, falling back to `.env`.

        Real environment variables win over `.env`. Nothing is written to
        `os.environ`. Overrides set to None are ignored, so CLI flags that were
        not given can be passed straight through. Commands that never touch the
        network can pass `require_contact=False`; their settings then have an
        empty contact and must not be used to build a User-Agent.
        """
        env_file = dotenv_path if dotenv_path is not None else find_dotenv(usecwd=True)
        env = {
            **dotenv_values(env_file),
            **(os.environ if environ is None else environ),
        }

        contact = (env.get(CONTACT_ENV_VAR) or "").strip()
        if not contact and require_contact:
            raise RuntimeError(
                f"{CONTACT_ENV_VAR} is not set. Add it to .env (see .env.example) "
                "so requests to Databricks identify who is crawling."
            )
        if any(ch in contact for ch in "\r\n\0"):
            raise RuntimeError(f"{CONTACT_ENV_VAR} must be a single line")

        kwargs: dict[str, Any] = {
            "contact": contact,
            "data_dir": Path(env.get(DATA_DIR_ENV_VAR) or "data"),
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)
