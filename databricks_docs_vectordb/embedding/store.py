import array
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import NamedTuple, Self

SCHEMA_VERSION = 1
STATUSES = ("ok", "failed", "rejected")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id        TEXT NOT NULL,
    sub_index       INTEGER NOT NULL DEFAULT 0,
    sub_count       INTEGER NOT NULL DEFAULT 1,
    url             TEXT NOT NULL,
    source          TEXT NOT NULL,
    cloud           TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    prev_chunk_id   TEXT,
    next_chunk_id   TEXT,
    heading_path    TEXT NOT NULL,
    section_path    TEXT NOT NULL,
    doc_title       TEXT NOT NULL,
    description     TEXT,
    last_updated    TEXT,
    page_sha256     TEXT NOT NULL,
    chunk_text_hash TEXT NOT NULL,
    embedded_text   TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    model_version   TEXT,
    dimensions      INTEGER,
    embedding       BLOB,
    status          TEXT NOT NULL CHECK (status IN ({", ".join(f"'{s}'" for s in STATUSES)})),
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    embedded_at     TEXT NOT NULL,
    PRIMARY KEY (chunk_id, sub_index),
    -- A successful row must carry a vector and know its shape; a failed or
    -- rejected attempt has neither (there's nothing the provider returned).
    CHECK (status != 'ok' OR (embedding IS NOT NULL AND dimensions IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS idx_embeddings_url ON embeddings (url);
CREATE INDEX IF NOT EXISTS idx_embeddings_status ON embeddings (status);
PRAGMA user_version = {SCHEMA_VERSION};
"""


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def pack_embedding(vector: Sequence[float]) -> bytes:
    """Encode a vector as the little-endian float32 BLOB the `embedding`
    column stores -- stdlib `array`, so no numpy dependency is needed."""
    return array.array("f", vector).tobytes()


def unpack_embedding(blob: bytes) -> array.array:
    """Decode an `embedding` BLOB back into a float32 `array`."""
    values = array.array("f")
    values.frombytes(blob)
    return values


@dataclass(frozen=True)
class EmbeddingRow:
    chunk_id: str
    sub_index: int
    sub_count: int
    url: str
    source: str
    cloud: str
    chunk_index: int
    prev_chunk_id: str | None
    next_chunk_id: str | None
    heading_path: str
    section_path: str
    doc_title: str
    description: str | None
    last_updated: str | None
    page_sha256: str
    chunk_text_hash: str
    embedded_text: str
    token_count: int
    provider: str
    model: str
    model_version: str | None
    dimensions: int | None
    embedding: bytes | None
    status: str
    attempts: int
    error: str | None
    # Stamped by EmbeddingsStore.upsert, not the runner; whatever value is
    # set here when constructing a row to write is ignored.
    embedded_at: str | None = None


class ModelInfo(NamedTuple):
    provider: str
    model: str
    model_version: str | None


class ModelKey(NamedTuple):
    provider: str
    model: str
    model_version: str | None
    dimensions: int | None


class EmbeddingCounts(NamedTuple):
    chunk_ids: int  # distinct chunk_id values with at least one row
    rows: int  # total rows (== vectors stored, counting oversized splits)
    ok: int
    failed: int
    rejected: int


class StoredChunk(NamedTuple):
    """What's stored for one chunk_id, across all its sub_index rows."""

    chunk_text_hash: str
    model_info: ModelInfo
    any_failed: bool  # at least one row has status='failed'


class ChunkProblem(NamedTuple):
    chunk_id: str
    url: str
    error: str | None


class EmbeddingSummary(NamedTuple):
    chunk_id: str
    sub_index: int
    sub_count: int
    status: str
    provider: str
    model: str
    model_version: str | None
    dimensions: int | None
    blob_len: int | None  # LENGTH(embedding); None when embedding is NULL


class EmbeddingsStore:
    """SQLite record of every embedding this phase has produced from Phase
    2's chunk store.

    One connection, used from one thread. `upsert` is the only write path:
    one chunk_id's whole sub_index set is replaced atomically, so a chunk
    that used to split into several embedding-only slices and now needs a
    different split (or none) never leaves rows from the old set behind.
    The database runs in WAL mode, so `embed status` can read it while an
    embedding run is writing.
    """

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row

        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            self._conn.close()
            raise RuntimeError(
                f"{path} has embeddings-store schema version {version}; "
                f"this code understands version {SCHEMA_VERSION}"
            )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- writes ------------------------------------------------------------

    def upsert(
        self, chunk_id: str, rows: Sequence[EmbeddingRow], *, now: str | None = None
    ) -> None:
        """Replace everything stored for `chunk_id` with `rows`, atomically.

        Delete-then-insert in one transaction, so a chunk that used to split
        into N embedding-only slices and now needs a different N (or just
        one) never leaves rows from the old split behind. An empty `rows`
        just clears the chunk.

        Every row must belong to `chunk_id`; `sub_index` values must be
        exactly `0..len(rows)-1` with no duplicates or gaps; and every row's
        `sub_count` must equal `len(rows)` -- one chunk's rows always
        describe the same split. All of this is checked before anything is
        written.
        """
        stamp = now or utc_now()

        for row in rows:
            if row.chunk_id != chunk_id:
                raise ValueError(
                    f"row has chunk_id {row.chunk_id!r}, expected {chunk_id!r}"
                )
            if row.sub_count != len(rows):
                raise ValueError(
                    f"chunk {chunk_id!r}#{row.sub_index} has sub_count "
                    f"{row.sub_count!r}, expected {len(rows)!r} (== len(rows))"
                )
        sub_indices = sorted(row.sub_index for row in rows)
        if sub_indices != list(range(len(rows))):
            raise ValueError(
                f"chunk {chunk_id!r}'s sub_index values must be exactly "
                f"0..{len(rows) - 1} with no duplicates or gaps, got {sub_indices}"
            )

        with self._conn:
            self._conn.execute("DELETE FROM embeddings WHERE chunk_id = ?", (chunk_id,))
            self._conn.executemany(
                "INSERT INTO embeddings ("
                " chunk_id, sub_index, sub_count, url, source, cloud,"
                " chunk_index, prev_chunk_id, next_chunk_id, heading_path,"
                " section_path, doc_title, description, last_updated,"
                " page_sha256, chunk_text_hash, embedded_text, token_count,"
                " provider, model, model_version, dimensions, embedding,"
                " status, attempts, error, embedded_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,"
                " ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        r.chunk_id,
                        r.sub_index,
                        r.sub_count,
                        r.url,
                        r.source,
                        r.cloud,
                        r.chunk_index,
                        r.prev_chunk_id,
                        r.next_chunk_id,
                        r.heading_path,
                        r.section_path,
                        r.doc_title,
                        r.description,
                        r.last_updated,
                        r.page_sha256,
                        r.chunk_text_hash,
                        r.embedded_text,
                        r.token_count,
                        r.provider,
                        r.model,
                        r.model_version,
                        r.dimensions,
                        r.embedding,
                        r.status,
                        r.attempts,
                        r.error,
                        stamp,
                    )
                    for r in rows
                ],
            )

    # -- reads ---------------------------------------------------------------

    def get_hash(self, chunk_id: str) -> str | None:
        """`chunk_text_hash` for `chunk_id`, or None if it has no rows.

        Shared across all of a chunk_id's sub_index rows, so any one of them
        answers this -- used to decide whether a chunk needs re-embedding
        (Resolved question 4).
        """
        row = self._conn.execute(
            "SELECT chunk_text_hash FROM embeddings WHERE chunk_id = ? LIMIT 1",
            (chunk_id,),
        ).fetchone()
        return row["chunk_text_hash"] if row else None

    def get_model_info(self, chunk_id: str) -> ModelInfo | None:
        """`(provider, model, model_version)` for `chunk_id`, or None if it
        has no rows.

        Shared across all of a chunk_id's sub_index rows (they're always
        embedded together, in the same call) -- used alongside `get_hash` to
        decide whether a chunk needs re-embedding under the currently
        configured model (Resolved question 8).
        """
        row = self._conn.execute(
            "SELECT provider, model, model_version FROM embeddings"
            " WHERE chunk_id = ? LIMIT 1",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return ModelInfo(row["provider"], row["model"], row["model_version"])

    def stored_chunks(self) -> dict[str, StoredChunk]:
        """A `StoredChunk` per chunk_id, in one query -- the bulk form of
        `get_hash` + `get_model_info` (plus whether any row failed), for
        the runner's candidate selection over every chunk in the corpus.

        `chunk_text_hash` and the model columns are shared across a
        chunk_id's rows (they're written together by one `upsert`), so
        SQLite's pick of which row to read them from doesn't matter.
        """
        rows = self._conn.execute(
            "SELECT chunk_id, chunk_text_hash, provider, model, model_version,"
            " MAX(status = 'failed') AS any_failed"
            " FROM embeddings GROUP BY chunk_id"
        )
        return {
            row["chunk_id"]: StoredChunk(
                row["chunk_text_hash"],
                ModelInfo(row["provider"], row["model"], row["model_version"]),
                bool(row["any_failed"]),
            )
            for row in rows
        }

    def chunks_with_status(
        self, status: str, limit: int | None = None
    ) -> tuple[int, list[ChunkProblem]]:
        """How many distinct chunks have a row with `status`, and the first
        `limit` of them (all if None) with their error -- for `embed
        status`'s failed/rejected listing. One entry per chunk_id, not per
        row: a split chunk's rows share one outcome and one error."""
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        total = self._conn.execute(
            "SELECT COUNT(DISTINCT chunk_id) FROM embeddings WHERE status = ?",
            (status,),
        ).fetchone()[0]
        rows = self._conn.execute(
            "SELECT chunk_id, url, MIN(error) AS error FROM embeddings"
            " WHERE status = ? GROUP BY chunk_id ORDER BY chunk_id LIMIT ?",
            (status, -1 if limit is None else limit),
        )
        return total, [ChunkProblem(r["chunk_id"], r["url"], r["error"]) for r in rows]

    def all_summaries(self) -> list[EmbeddingSummary]:
        """A lightweight `EmbeddingSummary` per row, for `embed status` and
        `embed verify`.

        Deliberately excludes `embedded_text` and the `embedding` BLOB
        itself -- there can be tens of thousands of rows in the full corpus,
        and neither is needed to group by status or by model, or to check
        `sub_index`/`sub_count` consistency and a BLOB's length against its
        `dimensions`. `LENGTH(embedding)` is computed by SQLite itself, so
        the BLOB's bytes never cross into Python.
        """
        rows = self._conn.execute(
            "SELECT chunk_id, sub_index, sub_count, status, provider, model,"
            " model_version, dimensions, LENGTH(embedding) AS blob_len"
            " FROM embeddings"
        )
        return [
            EmbeddingSummary(
                row["chunk_id"],
                row["sub_index"],
                row["sub_count"],
                row["status"],
                row["provider"],
                row["model"],
                row["model_version"],
                row["dimensions"],
                row["blob_len"],
            )
            for row in rows
        ]

    def counts(self) -> EmbeddingCounts:
        """Distinct chunks embedded, total rows, and rows by status."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT chunk_id) AS chunk_ids, COUNT(*) AS rows,"
            " COALESCE(SUM(status = 'ok'), 0) AS ok,"
            " COALESCE(SUM(status = 'failed'), 0) AS failed,"
            " COALESCE(SUM(status = 'rejected'), 0) AS rejected"
            " FROM embeddings"
        ).fetchone()
        return EmbeddingCounts(
            row["chunk_ids"], row["rows"], row["ok"], row["failed"], row["rejected"]
        )

    def counts_by_model(self) -> dict[ModelKey, EmbeddingCounts]:
        """The same counts as `counts()`, broken out per `(provider, model,
        model_version, dimensions)` -- how `embed status`/`embed verify`
        show a model swap as a trackable partial migration (Resolved
        question 8)."""
        rows = self._conn.execute(
            "SELECT provider, model, model_version, dimensions,"
            " COUNT(DISTINCT chunk_id) AS chunk_ids, COUNT(*) AS rows,"
            " COALESCE(SUM(status = 'ok'), 0) AS ok,"
            " COALESCE(SUM(status = 'failed'), 0) AS failed,"
            " COALESCE(SUM(status = 'rejected'), 0) AS rejected"
            " FROM embeddings GROUP BY 1, 2, 3, 4"
        )
        return {
            ModelKey(
                row["provider"], row["model"], row["model_version"], row["dimensions"]
            ): EmbeddingCounts(
                row["chunk_ids"], row["rows"], row["ok"], row["failed"], row["rejected"]
            )
            for row in rows
        }
