import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from dotenv import dotenv_values, find_dotenv

from databricks_docs_vectordb.scraping.config import DATA_DIR_ENV_VAR

OVERLAP_MODES = ("chars", "heading")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")

    # Target size of a chunk's rendered markdown, in characters. Characters
    # (not tokens) so this has no dependency on picking an embedding model's
    # tokenizer up front. A starting guess, not a final answer -- see
    # parsing.md's "chunk size/overlap tuning" revisit trigger.
    target_chunk_size: int = 3000

    # How much trailing context is repeated at the start of the next chunk
    # when a page is split. "chars": repeat the last `overlap_size`
    # characters of the previous chunk. "heading": ignore `overlap_size` and
    # instead restate the enclosing heading(s) at the top of the next chunk.
    overlap_size: int = 300
    overlap_mode: str = "chars"

    def __post_init__(self) -> None:
        if self.target_chunk_size <= 0:
            raise ValueError(
                f"target_chunk_size must be > 0, got {self.target_chunk_size!r}"
            )
        if self.overlap_size < 0:
            raise ValueError(f"overlap_size must be >= 0, got {self.overlap_size!r}")
        if self.overlap_size >= self.target_chunk_size:
            raise ValueError(
                "overlap_size must be smaller than target_chunk_size, got "
                f"overlap_size={self.overlap_size!r}, "
                f"target_chunk_size={self.target_chunk_size!r}"
            )
        if self.overlap_mode not in OVERLAP_MODES:
            raise ValueError(
                f"overlap_mode must be one of {OVERLAP_MODES}, got {self.overlap_mode!r}"
            )

    @property
    def raw_dir(self) -> Path:
        """Where Phase 1 (scraping) wrote the HTML this phase reads."""
        return self.data_dir / "raw"

    @property
    def manifest_path(self) -> Path:
        """The Phase 1 manifest this phase reads `ok` rows from."""
        return self.data_dir / "manifest.sqlite"

    @property
    def parsed_path(self) -> Path:
        """This phase's own output: the chunk store."""
        return self.data_dir / "parsed.sqlite"

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
        `DATABRICKS_DOCS_DATA_DIR` with scraping/config.py, since both
        phases read and write under the same `data/` directory -- parsing
        needs no contact address, because it never touches the network.
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
