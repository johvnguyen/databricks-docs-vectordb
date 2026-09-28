import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import NamedTuple, Self

SCHEMA_VERSION = 2

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    url               TEXT NOT NULL,
    source            TEXT NOT NULL,
    cloud             TEXT NOT NULL,
    chunk_index       INTEGER NOT NULL,
    prev_chunk_id     TEXT,
    next_chunk_id     TEXT,
    heading_path      TEXT NOT NULL,
    section_path      TEXT NOT NULL,
    doc_title         TEXT NOT NULL,
    description       TEXT,
    text              TEXT NOT NULL,
    char_count        INTEGER NOT NULL,
    token_count       INTEGER,
    token_count_model TEXT,
    is_oversized      INTEGER NOT NULL DEFAULT 0 CHECK (is_oversized IN (0, 1)),
    last_updated      TEXT,
    page_sha256       TEXT NOT NULL,
    fetched_at        TEXT,
    parsed_at         TEXT NOT NULL,
    UNIQUE (url, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_url ON chunks (url);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks (source);
PRAGMA user_version = {SCHEMA_VERSION};
"""


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
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
    text: str
    char_count: int
    token_count: int | None
    # Which embedding model's tokenizer produced token_count -- None until
    # a backfill has run, so Phase 3 can tell "never tokenized" and "tagged
    # by a since-swapped model" apart from "current" (see
    # chunking_and_embedding.md's Step 3 and its token_count/model-swap
    # revisit trigger). Not the chunker's concern to set.
    token_count_model: str | None
    is_oversized: bool
    last_updated: str | None
    page_sha256: str
    fetched_at: str | None
    # Stamped by ChunkStore.replace_page_chunks, not the chunker; whatever
    # value is set here when constructing a Chunk to write is ignored.
    parsed_at: str | None = None


class ChunkCounts(NamedTuple):
    pages: int  # distinct urls with at least one chunk
    chunks: int  # total chunk rows
    oversized: int  # chunks flagged is_oversized


class ChunkSummary(NamedTuple):
    url: str
    source: str
    char_count: int
    is_oversized: bool


class ChunkText(NamedTuple):
    chunk_id: str
    text: str


class ChunkStore:
    """SQLite record of every chunk this phase has produced from Phase 1's
    snapshot.

    One connection, used from one thread. `replace_page_chunks` is the only
    write path: a page's whole chunk set is replaced atomically, so a page's
    chunk count changing between runs (a chunking-rule tweak, a resized
    target) never leaves orphaned rows. The database runs in WAL mode, so
    `parse status` can read it while a parse run is writing.
    """

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row

        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, SCHEMA_VERSION):
            self._conn.close()
            raise RuntimeError(
                f"{path} has chunk-store schema version {version}; "
                f"this code understands version {SCHEMA_VERSION}"
            )
        self._conn.execute("PRAGMA journal_mode=WAL")
        if version == 1:
            # Version 1 predates token_count_model (version 2), so Phase 3
            # can tell which model tokenized a chunk and re-tokenize on a
            # model swap. CREATE TABLE IF NOT EXISTS below won't add a
            # column to an existing table, so this needs an explicit ALTER.
            # Every pre-existing row's token_count came from an untracked
            # tokenizer, so token_count_model is left NULL here -- the next
            # backfill treats that the same as "never tokenized" and
            # recomputes it under the currently configured model.
            with self._conn:
                self._conn.execute(
                    "ALTER TABLE chunks ADD COLUMN token_count_model TEXT"
                )
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

    def replace_page_chunks(
        self, url: str, chunks: Sequence[Chunk], *, now: str | None = None
    ) -> None:
        """Replace everything stored for `url` with `chunks`, atomically.

        Delete-then-insert in one transaction, so re-parsing a page after a
        markdown-rendering or chunking-rule change (or a resized target)
        never leaves chunks behind from the old set, however its chunk count
        changed. An empty `chunks` just clears the page.

        Every chunk must belong to `url`, `chunk_id` and `chunk_index` must
        each be unique within the batch, and any `prev_chunk_id`/
        `next_chunk_id` must point at another chunk in this same batch (or be
        None) -- a chunk can only be adjacent to a chunk of its own page, and
        a page's whole chunk set arrives in one call. All of this is checked
        before anything is written.
        """
        stamp = now or utc_now()

        ids = [c.chunk_id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate chunk_id values for {url!r}: {ids}")
        indices = [c.chunk_index for c in chunks]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate chunk_index values for {url!r}: {indices}")

        id_set = set(ids)
        for chunk in chunks:
            if chunk.url != url:
                raise ValueError(
                    f"chunk {chunk.chunk_id!r} has url {chunk.url!r}, expected {url!r}"
                )
            for neighbor in (chunk.prev_chunk_id, chunk.next_chunk_id):
                if neighbor is not None and neighbor not in id_set:
                    raise ValueError(
                        f"chunk {chunk.chunk_id!r} points at {neighbor!r}, "
                        "which is not in this batch"
                    )

        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE url = ?", (url,))
            self._conn.executemany(
                "INSERT INTO chunks ("
                " chunk_id, url, source, cloud, chunk_index, prev_chunk_id,"
                " next_chunk_id, heading_path, section_path, doc_title,"
                " description, text, char_count, token_count,"
                " token_count_model, is_oversized, last_updated, page_sha256,"
                " fetched_at, parsed_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.chunk_id,
                        c.url,
                        c.source,
                        c.cloud,
                        c.chunk_index,
                        c.prev_chunk_id,
                        c.next_chunk_id,
                        c.heading_path,
                        c.section_path,
                        c.doc_title,
                        c.description,
                        c.text,
                        c.char_count,
                        c.token_count,
                        c.token_count_model,
                        int(c.is_oversized),
                        c.last_updated,
                        c.page_sha256,
                        c.fetched_at,
                        stamp,
                    )
                    for c in chunks
                ],
            )

    def set_token_counts(self, counts: Mapping[str, int], model: str) -> int:
        """Write a real `token_count`, tagged with the `model` that produced
        it, for each `chunk_id` in `counts`.

        For Phase 3's tiktoken backfill (`embedding/tokens.py`) -- Phase 2
        never picks a tokenizer (`parsing.md`'s Resolved question 6), so
        this is the one write path into these two columns, separate from
        `replace_page_chunks`'s full-chunk replace. One transaction, one
        `UPDATE` per row. A `chunk_id` not currently in the store (e.g. a
        concurrent `parse run` replaced its page between reading and
        writing) is silently skipped, not an error. Returns how many rows
        were actually updated, which may be fewer than `len(counts)`.
        """
        if not counts:
            return 0
        with self._conn:
            cur = self._conn.executemany(
                "UPDATE chunks SET token_count = ?, token_count_model = ?"
                " WHERE chunk_id = ?",
                [(count, model, chunk_id) for chunk_id, count in counts.items()],
            )
        return cur.rowcount

    # -- reads ---------------------------------------------------------------

    def get_chunks(self, url: str) -> list[Chunk]:
        """A page's chunks, ordered by `chunk_index`."""
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE url = ? ORDER BY chunk_index", (url,)
        )
        return [_row_to_chunk(row) for row in rows]

    def counts(self) -> ChunkCounts:
        """Pages parsed, chunks produced, and how many are oversized."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT url) AS pages, COUNT(*) AS chunks,"
            " COALESCE(SUM(is_oversized), 0) AS oversized FROM chunks"
        ).fetchone()
        return ChunkCounts(row["pages"], row["chunks"], row["oversized"])

    def counts_by_source(self) -> dict[str, ChunkCounts]:
        """The same counts as `counts()`, broken out per source."""
        rows = self._conn.execute(
            "SELECT source, COUNT(DISTINCT url) AS pages, COUNT(*) AS chunks,"
            " COALESCE(SUM(is_oversized), 0) AS oversized"
            " FROM chunks GROUP BY 1"
        )
        return {
            row["source"]: ChunkCounts(row["pages"], row["chunks"], row["oversized"])
            for row in rows
        }

    def all_summaries(self) -> list[ChunkSummary]:
        """`(url, source, char_count, is_oversized)` for every chunk.

        For reporting (`parse status`'s per-source/section breakdown and
        chunk-size distribution) -- deliberately not `Chunk`'s full row,
        since a report only needs these four fields, and there can be tens
        of thousands of chunks in the full corpus.
        """
        rows = self._conn.execute(
            "SELECT url, source, char_count, is_oversized FROM chunks"
        )
        return [
            ChunkSummary(
                row["url"], row["source"], row["char_count"], bool(row["is_oversized"])
            )
            for row in rows
        ]

    def chunks_needing_token_count(self, model: str) -> list[ChunkText]:
        """`(chunk_id, text)` for every chunk with no `token_count` tagged
        as coming from `model`.

        Covers three cases with one predicate: never tokenized at all
        (`token_count_model IS NULL`), tokenized before this column existed
        (also `NULL`, per the version-1-to-2 migration), and tokenized
        under a *different*, since-swapped model -- `IS NOT` treats `NULL`
        as its own value, unlike `!=`, so all three match without a
        separate `OR ... IS NULL` clause. This is also how you test a
        different tokenizer: pass a different `model` and every chunk
        looks "needing" again.

        For Phase 3's tiktoken backfill (`embedding/tokens.py`) --
        deliberately just these two fields, not the full `Chunk` row, since
        there can be tens of thousands of chunks in the full corpus and
        nothing else is needed to compute a count.
        """
        rows = self._conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE token_count_model IS NOT ?",
            (model,),
        )
        return [ChunkText(row["chunk_id"], row["text"]) for row in rows]


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    data = dict(row)
    data["is_oversized"] = bool(data["is_oversized"])
    return Chunk(**data)
