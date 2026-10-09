import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from dotenv import dotenv_values, find_dotenv

from databricks_docs_vectordb.scraping.config import DATA_DIR_ENV_VAR


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")

    # Passed straight to litellm.embedding() -- swapping provider or model
    # is a one-line change here, per chunking_and_embedding.md's "Design >
    # Provider and model". Default is OpenAI's standard embedding model.
    model: str = "text-embedding-3-small"

    # The configured model's real per-input token limit (its own tokenizer,
    # not a char-based guess) -- 8,191 for text-embedding-3-small/-large.
    # Moves together with `model`: swapping to a model with a different
    # limit means updating this too.
    model_token_limit: int = 8191

    # How many tokens of the previous slice are repeated at the start of the
    # next one when an oversized chunk is split for embedding (Design >
    # Oversized chunks). Independent of parsing/config.py's overlap_size,
    # which is characters and belongs to a different phase's chunk store.
    split_overlap_tokens: int = 200

    # Batching is bounded by both a count and a summed token budget,
    # whichever binds first (Design > Batching, concurrency, and rate
    # limits).
    batch_size: int = 100
    batch_token_budget: int = 200_000

    # Rate limiting, both dimensions OpenAI enforces (RPM and TPM).
    # Conservative starting points, not yet tuned against a real account
    # tier -- see chunking_and_embedding.md's "Rate/batch defaults are
    # guesses" revisit trigger.
    rate_rpm: float = 500.0
    rate_tpm: float = 1_000_000.0
    concurrency: int = 4

    # Retry, mirroring scraping/config.py's shape and defaults.
    max_attempts: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    retry_after_max: float = 300.0  # longest Retry-After we'll wait out

    # Circuit breaker, mirroring scraping/config.py's shape and defaults.
    breaker_threshold: int = 10  # consecutive failures before it trips
    breaker_cooldown: float = 600.0
    breaker_max_trips: int = 3

    # Consecutive non-retryable errors (a bad API key, an unknown model, a
    # malformed request) that end the run early. These don't feed the
    # breaker -- the provider *is* answering -- but a run of them almost
    # always means every remaining batch would fail the same way.
    fatal_error_threshold: int = 5

    progress_every: int = 1000  # log a progress line every N chunks

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        positive = (
            "model_token_limit",
            "batch_size",
            "batch_token_budget",
            "rate_rpm",
            "rate_tpm",
            "concurrency",
            "max_attempts",
            "backoff_base",
            "backoff_max",
            "retry_after_max",
            "breaker_threshold",
            "breaker_max_trips",
            "fatal_error_threshold",
            "progress_every",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        for name in ("split_overlap_tokens", "breaker_cooldown"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if self.split_overlap_tokens >= self.model_token_limit:
            raise ValueError(
                "split_overlap_tokens must be smaller than model_token_limit, "
                f"got split_overlap_tokens={self.split_overlap_tokens!r}, "
                f"model_token_limit={self.model_token_limit!r}"
            )

    @property
    def parsed_path(self) -> Path:
        """Phase 2's output: the chunk store this phase reads from."""
        return self.data_dir / "parsed.sqlite"

    @property
    def embeddings_path(self) -> Path:
        """This phase's own output: the embeddings store."""
        return self.data_dir / "embeddings.sqlite"

    @property
    def logs_dir(self) -> Path:
        """Shared with scraping: `embed run` logs beside `scrape fetch`."""
        return self.data_dir / "logs"

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
        `DATABRICKS_DOCS_DATA_DIR` with scraping/config.py and
        parsing/config.py, since all three phases read and write under the
        same `data/` directory -- `embeddings_path` is derived from it, not
        its own separately overridable path. No provider API key is read or
        required here: `litellm.embedding()` (Step 5) reads one from the
        environment only when a call is actually made.
        """
        env_file = dotenv_path if dotenv_path is not None else find_dotenv(usecwd=True)
        env = {
            **dotenv_values(env_file),
            **(os.environ if environ is None else environ),
        }

        kwargs: dict[str, Any] = {
            "data_dir": Path(env.get(DATA_DIR_ENV_VAR) or "data"),
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)
