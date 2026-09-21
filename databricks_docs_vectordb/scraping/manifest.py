import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import NamedTuple, Self

from databricks_docs_vectordb.scraping.urls import url_to_rel_path

SCHEMA_VERSION = 1
STATUSES = ("pending", "ok", "gone", "failed", "removed")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS pages (
    url           TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    rel_path      TEXT NOT NULL UNIQUE,
    status        TEXT NOT NULL CHECK (status IN ({", ".join(f"'{s}'" for s in STATUSES)})),
    http_status   INTEGER,
    final_url     TEXT,
    etag          TEXT,
    last_modified TEXT,
    sha256        TEXT,
    size_bytes    INTEGER,
    first_seen_at TEXT NOT NULL,
    fetched_at    TEXT,
    checked_at    TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_pages_status ON pages (status);
PRAGMA user_version = {SCHEMA_VERSION};
"""


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Page:
    url: str
    source: str
    rel_path: str
    status: str
    http_status: int | None
    final_url: str | None
    etag: str | None
    last_modified: str | None
    sha256: str | None
    size_bytes: int | None
    first_seen_at: str
    fetched_at: str | None
    checked_at: str | None
    attempts: int
    error: str | None


class AddResult(NamedTuple):
    added: int  # URLs never seen before
    revived: int  # URLs that had been marked removed and are back


class Manifest:
    """SQLite record of every page: where it came from and what we fetched.

    One connection, used from one thread. Each method is its own transaction,
    so a crash never leaves a half-applied update. The database runs in WAL
    mode, so `scrape status` can read it while a crawl is writing.
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
                f"{path} has manifest schema version {version}; "
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

    # -- enumerate -------------------------------------------------------

    def add_pending(self, source: str, urls: Iterable[str]) -> AddResult:
        """Insert URLs we haven't seen as `pending`; leave existing rows alone.

        Idempotent: existing rows keep their status and validators. A row that
        had been marked `removed` and is back in the sitemap returns to
        `pending`, keeping its validators so the next fetch can revalidate it
        with a cheap conditional GET.

        All-or-nothing. Raises ValueError for a URL outside the configured
        sources, and sqlite3.IntegrityError if two URLs map to the same file.
        """
        rows = [(url, source, url_to_rel_path(url)) for url in dict.fromkeys(urls)]
        now = utc_now()
        with self._conn:
            before = self._conn.total_changes
            self._conn.executemany(
                "INSERT INTO pages (url, source, rel_path, status, first_seen_at)"
                " VALUES (?, ?, ?, 'pending', ?) ON CONFLICT (url) DO NOTHING",
                [(url, src, path, now) for url, src, path in rows],
            )
            added = self._conn.total_changes - before

            before = self._conn.total_changes
            self._conn.executemany(
                "UPDATE pages SET status = 'pending', attempts = 0, error = NULL"
                " WHERE url = ? AND status = 'removed'",
                [(url,) for url, _, _ in rows],
            )
            revived = self._conn.total_changes - before
        return AddResult(added, revived)

    def mark_removed(self, source: str, current_urls: Iterable[str]) -> int:
        """Mark this source's rows that are no longer in the sitemap `removed`.

        `current_urls` is what the sitemap lists now (after robots filtering).
        Files and validators are kept; deciding what to drop downstream is
        Phase 7's job. Returns how many rows changed.
        """
        current = set(current_urls)
        if not current:
            raise ValueError(
                "refusing to mark every page removed: the sitemap URL list is empty"
            )
        with self._conn:
            known = self._conn.execute(
                "SELECT url FROM pages WHERE source = ? AND status != 'removed'",
                (source,),
            ).fetchall()
            gone = [(row["url"],) for row in known if row["url"] not in current]
            self._conn.executemany(
                "UPDATE pages SET status = 'removed' WHERE url = ?", gone
            )
        return len(gone)

    # -- fetch -----------------------------------------------------------

    def next_batch(
        self,
        refresh: bool = False,
        limit: int | None = None,
        stale_before: str | None = None,
    ) -> list[Page]:
        """Rows that need fetching: `pending`, plus `ok` rows when refreshing.

        Pending rows come first, then ok rows oldest-checked first. With
        `stale_before` (a UTC timestamp, normally the run's start time), only
        ok rows not checked since then are returned, so an interrupted
        refresh resumes instead of starting over.
        """
        clauses = ["status = 'pending'"]
        params: list[str | int] = []
        if refresh:
            ok = "status = 'ok'"
            if stale_before is not None:
                ok += " AND (checked_at IS NULL OR checked_at < ?)"
                params.append(stale_before)
            clauses.append(ok)
        sql = (
            f"SELECT * FROM pages WHERE {' OR '.join(f'({c})' for c in clauses)}"
            " ORDER BY status = 'ok', COALESCE(checked_at, ''), url"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [Page(**dict(row)) for row in self._conn.execute(sql, params)]

    def record_ok(
        self,
        url: str,
        *,
        http_status: int,
        sha256: str,
        size_bytes: int,
        etag: str | None = None,
        last_modified: str | None = None,
        final_url: str | None = None,
        now: str | None = None,
    ) -> None:
        """A 200: the body was written; it is new or changed."""
        now = now or utc_now()
        self._update(
            url,
            status="ok",
            http_status=http_status,
            final_url=final_url if final_url != url else None,
            etag=etag,
            last_modified=last_modified,
            sha256=sha256,
            size_bytes=size_bytes,
            fetched_at=now,
            checked_at=now,
            attempts=0,
            error=None,
        )

    def record_not_modified(
        self,
        url: str,
        *,
        http_status: int = 304,
        etag: str | None = None,
        last_modified: str | None = None,
        now: str | None = None,
    ) -> None:
        """The stored body is still current, so only `checked_at` advances.

        Validators the server sent with the response replace the stored ones;
        omitted ones are left alone. This matters when a server re-issues an
        ETag for unchanged content: keeping the old one would make every later
        conditional request miss.
        """
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE pages SET status = 'ok', http_status = ?,"
                " etag = COALESCE(?, etag), last_modified = COALESCE(?, last_modified),"
                " checked_at = ?, attempts = 0, error = NULL WHERE url = ?",
                (http_status, etag, last_modified, now or utc_now(), url),
            )
        if cursor.rowcount == 0:
            raise KeyError(url)

    def record_gone(
        self, url: str, *, http_status: int = 404, now: str | None = None
    ) -> None:
        """A 404/410: the page no longer exists upstream."""
        self._update(
            url,
            status="gone",
            http_status=http_status,
            checked_at=now or utc_now(),
            attempts=0,
            error=None,
        )

    def record_failed(
        self,
        url: str,
        *,
        error: str,
        attempts: int,
        http_status: int | None = None,
    ) -> None:
        """Retries were exhausted.

        A page that already has a good body stays `ok` (stale) and just gets
        the error recorded, so a transient failure during a refresh doesn't
        turn good pages into `failed`.
        """
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE pages SET"
                " status = CASE WHEN status = 'ok' THEN 'ok' ELSE 'failed' END,"
                " error = ?, attempts = ?, http_status = ? WHERE url = ?",
                (error, attempts, http_status, url),
            )
        if cursor.rowcount == 0:
            raise KeyError(url)

    def requeue_failed(self) -> int:
        """Put `failed` rows back to `pending` so the next fetch retries them."""
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE pages SET status = 'pending', attempts = 0, error = NULL"
                " WHERE status = 'failed'"
            )
        return cursor.rowcount

    # -- inspection ------------------------------------------------------

    def get(self, url: str) -> Page | None:
        row = self._conn.execute("SELECT * FROM pages WHERE url = ?", (url,)).fetchone()
        return Page(**dict(row)) if row else None

    def pages_with_status(self, status: str, limit: int | None = None) -> list[Page]:
        """Rows in one status, ordered by URL (e.g. to show what failed)."""
        sql = "SELECT * FROM pages WHERE status = ? ORDER BY url"
        params: list[str | int] = [status]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [Page(**dict(row)) for row in self._conn.execute(sql, params)]

    def counts(self) -> Counter[str]:
        """Rows per status."""
        rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM pages GROUP BY 1")
        return Counter({row["status"]: row["n"] for row in rows})

    def counts_by_source(self) -> dict[str, Counter[str]]:
        """Rows per status, for each source."""
        rows = self._conn.execute(
            "SELECT source, status, COUNT(*) AS n FROM pages GROUP BY 1, 2"
        )
        result: dict[str, Counter[str]] = {}
        for row in rows:
            result.setdefault(row["source"], Counter())[row["status"]] = row["n"]
        return result

    # -- internals -------------------------------------------------------

    def _update(self, url: str, **columns: object) -> None:
        assignments = ", ".join(f"{name} = ?" for name in columns)
        with self._conn:
            cursor = self._conn.execute(
                f"UPDATE pages SET {assignments} WHERE url = ?",
                (*columns.values(), url),
            )
        if cursor.rowcount == 0:
            raise KeyError(url)
