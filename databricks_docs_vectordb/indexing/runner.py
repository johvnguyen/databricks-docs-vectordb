"""`index run`: load the configured model's embeddings rows into Qdrant
(vector_db_integration.md's "Idempotency, deletions and failure handling").

So far this holds the read side (Step 4), shared with `index verify`: the
orphan check, and the mapping from an embeddings row to the point it
becomes. Then the loader (Step 5): comparing the rows with what Qdrant
holds to decide what a run is to do, the question whether to save what it
would change, both before anything is written, then the writes, and the
`Runner` that does all of it with retries, a summary and an exit status.
"""

import array
import functools
import hashlib
import logging
import math
import random
import signal
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Container, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import httpx
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from databricks_docs_vectordb.embedding.store import (
    EmbeddingsStore,
    ModelInfo,
    StoredVector,
    unpack_embedding,
)
from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.store import Point, QdrantStore, collection_name
from databricks_docs_vectordb.parsing.report import section_of
from databricks_docs_vectordb.parsing.store import ChunkStore
from databricks_docs_vectordb.scraping.config import SOURCES, Source
from databricks_docs_vectordb.scraping.manifest import Manifest

logger = logging.getLogger(__name__)

RunStatus = Literal["completed", "interrupted", "failed"]

# `embed run`'s exit codes: 130 is the shell's convention for Ctrl-C.
EXIT_CODES: dict[RunStatus, int] = {"completed": 0, "interrupted": 130, "failed": 1}

# What a retry can help with (Q34): the connection failed or timed out.
# qdrant-client wraps these in ResponseHandlingException; `save_snapshot`'s
# download raises them as they are.
_RETRYABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)

# How many orphaned rows the summary's warning names before it just counts.
ORPHANS_NAMED = 10

# Every point ID is derived from this, so changing it changes them all (Q7).
POINT_ID_NAMESPACE = uuid.UUID("2405f98a-c138-459f-b780-53782478d987")


def skip_orphans(
    rows: Iterable[StoredVector],
    chunk_ids: Container[str],
    skipped: list[StoredVector] | None = None,
) -> Iterator[StoredVector]:
    """Yield, in order, the rows whose `chunk_id` is in `chunk_ids`.

    `chunk_ids` is `ChunkStore.chunk_ids()`: every chunk in
    `parsed.sqlite`, as `embed verify` sees them. Any other row is an
    orphan, left behind when its page was re-parsed into fewer chunks or
    removed, since nothing deletes it from `embeddings.sqlite`. Each one is
    skipped with a warning naming it, and appended to `skipped` when given,
    for the run's summary line (Q11, Q19).

    `index run` refuses to start while `embed verify --complete` reports
    orphans (Q32), so this is a safeguard and should find none.
    """
    for row in rows:
        if row.chunk_id in chunk_ids:
            yield row
            continue
        logger.warning(
            "skipping orphaned row %s (sub_index %d): its chunk_id is not in "
            "parsed.sqlite",
            row.chunk_id,
            row.sub_index,
        )
        if skipped is not None:
            skipped.append(row)


def point_id(chunk_id: str, sub_index: int) -> str:
    """UUID5 of `<chunk_id>|<sub_index>` under `POINT_ID_NAMESPACE` (Q7).

    Only the natural key goes in, nothing about chunking settings, content
    or model, so a row keeps its ID when it is re-embedded.
    """
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{chunk_id}|{sub_index}"))


def unit_vector(values: Sequence[float]) -> list[float]:
    """`values` scaled to unit length, each component rounded to float32.

    Qdrant stores vectors as float32 and leaves one that is already unit
    length as it is, under `Cosine` too, so a point read back holds exactly
    these values once its components are also taken as float32.
    """
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("a zero vector has no direction to keep")
    return array.array("f", [x / norm for x in values]).tolist()


def point_section(row: StoredVector, sources: Sequence[Source] = SOURCES) -> str:
    """`<source>/<section>`, the section as `parse status` reports it
    (`section_of`): `aws-en/sql`, `api/jobs`, or `aws-en/(top-level)` for a
    top-level page."""
    return f"{row.source}/{section_of(row.url, row.source, sources)}"


def to_point(
    row: StoredVector, *, normalize: bool, sources: Sequence[Source] = SOURCES
) -> Point:
    """The point `row` becomes: its ID, its vector (the BLOB unchanged
    unless `normalize`), and the four payload fields (Q9).

    `section` is `point_section(row)`.
    """
    # Unnormalized vectors only round-trip under `Dot`; `Cosine` re-normalizes them.
    values = unpack_embedding(row.embedding)
    try:
        vector = unit_vector(values) if normalize else values.tolist()
    except ValueError as e:
        raise ValueError(f"{row.chunk_id} (sub_index {row.sub_index}): {e}") from e
    return Point(
        id=point_id(row.chunk_id, row.sub_index),
        vector=vector,
        payload={
            "chunk_id": row.chunk_id,
            "sub_index": row.sub_index,
            "url": row.url,
            "section": point_section(row, sources),
        },
    )


@dataclass(frozen=True)
class Plan:
    """What `index run` is to do, decided before anything is written. Each
    page set holds URLs."""

    collection: str
    dimensions: int  # the rows', and so the collection's vector size
    exists: bool  # whether `collection` is already on the server
    # Drop and recreate `collection`, then load every row: `--refresh`, or
    # `mismatches` found it differs from the configured settings.
    refresh: bool
    mismatches: tuple[str, ...]  # how `collection` differs, from settings_mismatches
    current: frozenset[str]  # points already match the rows: left alone
    new: frozenset[str]  # rows and no points: loaded
    changed: frozenset[str]  # points deleted, then the rows loaded
    removed: frozenset[str]  # points deleted, nothing loaded
    not_ok: frozenset[str]  # rows not loaded: the page is not `ok` in the manifest
    old_collections: tuple[str, ...]  # other models' collections, to be deleted

    @property
    def affected_collections(self) -> list[str]:
        """The collections this run would change or delete: `collection` if
        it would replace or remove a page, then the old ones."""
        affected = [self.collection] if self.changed or self.removed else []
        return affected + list(self.old_collections)

    @property
    def replaces_anything(self) -> bool:
        """Whether this run would replace or remove a page, or delete a
        collection: when the save question is asked (Q29, Q40)."""
        return bool(self.affected_collections)


def compare(
    qdrant: QdrantStore,
    collection: str,
    rows: Iterable[StoredVector],
    ok_pages: Container[str],
    settings: Settings,
    *,
    dimensions: int,
    refresh: bool = False,
) -> Plan:
    """Compare `rows`, the configured model's current rows (after
    `skip_orphans`), with what `collection` holds, page by page (Q30).

    A page with points has changed if its point IDs differ from its rows',
    or any stored vector differs from its row's mapped vector (the
    normalized BLOB). A page whose URL is not in `ok_pages` (the manifest's
    `ok` pages) counts as having no rows, so its points are removed and its
    rows are not loaded (Q31). Every other `<prefix>_*` collection is a
    previous model's (Q37). With `refresh`, every page with points counts
    as changed, so no vectors are read back.

    An existing `collection` whose settings differ from what
    `create_collection` would make of it now (the distance was changed in
    config, say) is rebuilt as with `refresh`. `dimensions` is the rows'.

    Only a hash of each vector is kept, not the vector, and only pages that
    have points are mapped.
    """
    stored: dict[str, dict[str, bytes | None]] = {}  # url -> point ID -> hash
    exists = qdrant.collection_exists(collection)
    mismatches = (
        qdrant.settings_mismatches(collection, dimensions=dimensions) if exists else []
    )
    if mismatches:
        logger.warning(
            "%s does not match the settings (%s): rebuilding it, as with --refresh",
            collection,
            "; ".join(mismatches),
        )
    refresh = refresh or bool(mismatches)
    offset = None
    while exists:
        points, offset = qdrant.scroll(
            collection, offset=offset, with_vectors=not refresh
        )
        for point in points:
            stored.setdefault(point.url, {})[point.id] = (
                None if point.vector is None else vector_hash(point.vector)
            )
        if offset is None:
            break

    expected: dict[str, dict[str, bytes | None]] = {}
    not_ok: set[str] = set()
    for row in rows:
        if row.url not in ok_pages:
            not_ok.add(row.url)
            continue
        page = expected.setdefault(row.url, {})
        if row.url in stored and not refresh:
            point = to_point(row, normalize=settings.normalize)
            page[point.id] = vector_hash(point.vector)

    both = stored.keys() & expected.keys()
    changed = both if refresh else {u for u in both if stored[u] != expected[u]}
    prefix = f"{settings.collection_prefix}_"
    plan = Plan(
        collection=collection,
        dimensions=dimensions,
        exists=exists,
        refresh=refresh,
        mismatches=tuple(mismatches),
        current=frozenset(both - changed),
        new=frozenset(expected.keys() - stored.keys()),
        changed=frozenset(changed),
        removed=frozenset(stored.keys() - expected.keys()),
        not_ok=frozenset(not_ok),
        old_collections=tuple(
            sorted(
                name
                for name in qdrant.collection_names()
                if name.startswith(prefix) and name != collection
            )
        ),
    )
    logger.info(
        "compared %s with the rows: %d pages current, %d new, %d changed, "
        "%d removed, %d not ok in the manifest; %d old collection(s)",
        collection,
        len(plan.current),
        len(plan.new),
        len(plan.changed),
        len(plan.removed),
        len(plan.not_ok),
        len(plan.old_collections),
    )
    return plan


def vector_hash(vector: Sequence[float]) -> bytes:
    """A hash of `vector` as float32, the precision Qdrant stores, so a
    vector read back hashes the same as the one written."""
    return hashlib.blake2b(array.array("f", vector).tobytes(), digest_size=16).digest()


class Written(NamedTuple):
    pages_deleted: int  # changed and removed pages whose points were deleted
    pages_loaded: int  # new and changed pages whose rows were upserted
    points: int  # points upserted


def write(
    qdrant: QdrantStore,
    plan: Plan,
    rows: Iterable[StoredVector],
    settings: Settings,
) -> Written:
    """Carry out `plan`, given the same rows `compare` was given.

    With `plan.refresh`, the collection is dropped and recreated (Step 3)
    and every row is loaded. Otherwise the collection is created if it is
    missing, every changed and removed page's points are deleted by `url`,
    and the new and changed pages' rows are upserted. Current pages are
    left alone, and old models' collections are not touched here.

    All deletes come before any upsert, and points go in batches of
    `settings.batch_size` that may span pages, so a run stopped part-way
    leaves each page current, without points (new next time), or partly
    loaded (changed next time).
    """
    deleted = plan.changed | plan.removed
    if plan.refresh or not plan.exists:
        qdrant.recreate_collection(plan.collection, dimensions=plan.dimensions)
    else:
        for urls in batched(sorted(deleted), settings.batch_size):
            qdrant.delete_pages(plan.collection, urls)

    to_load = plan.new | plan.changed
    mapped = (
        to_point(row, normalize=settings.normalize)
        for row in rows
        if row.url in to_load
    )
    points = 0
    for batch in batched(mapped, settings.batch_size):
        qdrant.upsert(plan.collection, batch)
        before, points = points, points + len(batch)
        if points // settings.progress_every > before // settings.progress_every:
            logger.info("upserted %d points", points)
    written = Written(len(deleted), len(to_load), points)
    logger.info(
        "wrote %s: deleted %d pages' points, loaded %d pages (%d points)",
        plan.collection,
        written.pages_deleted,
        written.pages_loaded,
        written.points,
    )
    return written


class SaveQuestionUnanswered(Exception):
    """The run would replace, remove or delete something, but the save
    question could not be asked or was not answered."""


def should_save(
    plan: Plan,
    settings: Settings,
    save_old: bool | None,
    *,
    ask: Callable[[str], str] = input,
    interactive: bool | None = None,
) -> bool:
    """Whether to save the affected collections before writing anything:
    the save question (Q29, Q40), asked at most once.

    Not asked when nothing would be replaced, removed or deleted (a first
    load, or a run with nothing to do), nor when `save_old`
    (`--save-old`/`--no-save-old`) answers it in advance. Otherwise it is
    asked on the terminal, or, with no terminal (`interactive` defaults to
    whether stdin is one), `SaveQuestionUnanswered` is raised.
    """
    affected = plan.affected_collections
    if not affected:
        return False
    if save_old is not None:
        return save_old
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise SaveQuestionUnanswered(
            f"this run would change {', '.join(affected)}, and there is no "
            "terminal to ask whether to save them first: pass --save-old or "
            "--no-save-old"
        )
    question = (
        f"This run would {_changes(plan)}.\n"
        f"Save a snapshot of {', '.join(affected)} to {settings.snapshots_dir}/ "
        "first? [y/n] "
    )
    while True:
        try:
            answer = ask(question).strip().lower()
        except EOFError as e:
            raise SaveQuestionUnanswered("the save question was not answered") from e
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        question = "Please answer y or n: "


def save_affected(qdrant: QdrantStore, plan: Plan, settings: Settings) -> list[Path]:
    """Save a snapshot of each affected collection to
    `settings.snapshots_dir`, before the run writes anything."""
    paths = []
    for name in plan.affected_collections:
        path = qdrant.save_snapshot(name, settings.snapshots_dir)
        logger.info("saved %s to %s (%d MB)", name, path, path.stat().st_size // 10**6)
        paths.append(path)
    return paths


def _changes(plan: Plan) -> str:
    """What `plan` would replace, remove or delete, for the save question."""
    parts = []
    if plan.refresh and (plan.changed or plan.removed):
        pages = len(plan.changed) + len(plan.removed)
        parts.append(f"drop and rebuild {plan.collection}, which holds {pages} page(s)")
    else:
        if plan.changed:
            parts.append(f"replace {len(plan.changed)} page(s) in {plan.collection}")
        if plan.removed:
            parts.append(f"remove {len(plan.removed)} page(s) from {plan.collection}")
    if plan.old_collections:
        parts.append(f"delete the old collection(s) {', '.join(plan.old_collections)}")
    return "; ".join(parts)


class QdrantUnreachable(Exception):
    """A Qdrant call still failed to connect, or timed out, on its last
    attempt."""


class RunFailed(Exception):
    """`index run` cannot go ahead; the message says why."""


class RetryingStore:
    """A `QdrantStore` whose every method call is retried on a connection
    error or a timeout (Q34): up to `settings.max_attempts` attempts, with
    exponential backoff and jitter, each failed attempt logged. After the
    last, `QdrantUnreachable` is raised. Any other error is raised at once.

    A whole method call is one attempt, so it relies on every store method
    a run calls being safe to repeat.
    """

    def __init__(
        self,
        store: QdrantStore,
        settings: Settings,
        *,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._store = store
        self._settings = settings
        self._sleep = sleep
        self._rng = rng
        self.retries = 0

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._store, name)

        @functools.wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            attempts = self._settings.max_attempts
            for attempt in range(1, attempts + 1):
                try:
                    return method(*args, **kwargs)
                except Exception as e:
                    cause = _retryable_cause(e)
                    if cause is None:
                        raise
                    reason = f"{type(cause).__name__}: {cause}"
                    if attempt == attempts:
                        logger.error(
                            "qdrant %s: %s; giving up (attempt %d/%d)",
                            name,
                            reason,
                            attempt,
                            attempts,
                        )
                        raise QdrantUnreachable(
                            f"Qdrant at {self._settings.url} did not answer "
                            f"{name} in {attempts} attempts (last: {reason})"
                        ) from e
                    delay = _backoff(attempt, self._settings, self._rng())
                    logger.warning(
                        "qdrant %s: %s; retrying in %.1fs (attempt %d/%d)",
                        name,
                        reason,
                        delay,
                        attempt,
                        attempts,
                    )
                    self.retries += 1
                    self._sleep(delay)

        return call


@dataclass(frozen=True)
class RunSummary:
    status: RunStatus
    collection: str | None  # None if the run stopped before naming it
    plan: Plan | None  # None if the run stopped before the compare finished
    saved: tuple[Path, ...]  # snapshots saved before writing
    written: Written | None  # None if the run stopped before the write finished
    orphans: tuple[tuple[str, int], ...]  # (chunk_id, sub_index) of rows skipped
    retries: int
    elapsed: float
    deleted: tuple[str, ...] = ()  # old models' collections deleted (Q37)
    error: str | None = None  # why a failed run failed

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


class Runner:
    """`index run`: compare the configured model's rows with what Qdrant
    holds, ask the save question, save, write, then delete any old models'
    collections, once the new one passes `index verify --complete` (Q37).

    - Every Qdrant call goes through `RetryingStore`. There is no rate
      limiter or circuit breaker (Q19).
    - Ctrl-C or SIGTERM stops the run where it is (`interrupted`). That is
      safe anywhere: nothing is written before the save question, and
      `write` orders its deletes and batches so that every page is left
      current or due for replacement, which the next run's compare finds.
    - A connection that still fails after its retries, any other Qdrant
      error, an unanswered save question, or a new collection that fails
      verification while old ones wait to be deleted ends the run as
      `failed`. In the last case every old collection is kept.
    - Orphans are warned about as the compare reads them (`skip_orphans`),
      and again, by name, after the summary line.

    A Runner does one run.
    """

    def __init__(
        self,
        settings: Settings,
        model: ModelInfo,
        embeddings: EmbeddingsStore,
        chunks: ChunkStore,
        manifest: Manifest,
        qdrant: QdrantStore,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        ask: Callable[[str], str] = input,
        interactive: bool | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._embeddings = embeddings
        self._chunks = chunks
        self._manifest = manifest
        self._retrying = RetryingStore(qdrant, settings, sleep=sleep, rng=rng)
        # Every QdrantStore method, each call retried.
        self._qdrant = cast(QdrantStore, self._retrying)
        self._clock = clock
        self._ask = ask
        self._interactive = interactive

    def run(
        self,
        *,
        refresh: bool = False,
        save_old: bool | None = None,
        handle_signals: bool = True,
    ) -> RunSummary:
        """One `index run [--refresh] [--save-old | --no-save-old]`;
        `save_old` is None when neither flag was given."""
        started = self._clock()
        orphans: list[StoredVector] = []
        collection: str | None = None
        plan: Plan | None = None
        saved: list[Path] = []
        written: Written | None = None
        deleted: list[str] = []
        status: RunStatus = "completed"
        error: str | None = None
        previous = _raise_on_sigterm() if handle_signals else None
        try:
            dimensions = configured_dimensions(self._embeddings, self._model)
            collection = collection_name(
                self._settings.collection_prefix, self._model, dimensions
            )
            chunk_ids = self._chunks.chunk_ids()
            ok_pages = {page.url for page in self._manifest.pages_with_status("ok")}
            logger.info(
                "starting: %s/%s rows into %s at %s%s",
                self._model.provider,
                self._model.model,
                collection,
                self._settings.url,
                ", rebuilding it (--refresh)" if refresh else "",
            )
            plan = compare(
                self._qdrant,
                collection,
                skip_orphans(self._rows(), chunk_ids, orphans),
                ok_pages,
                self._settings,
                dimensions=dimensions,
                refresh=refresh,
            )
            if should_save(
                plan,
                self._settings,
                save_old,
                ask=self._ask,
                interactive=self._interactive,
            ):
                saved = save_affected(self._qdrant, plan, self._settings)
            # The compare warned about each orphan; skipped quietly here.
            rows = (row for row in self._rows() if row.chunk_id in chunk_ids)
            written = write(self._qdrant, plan, rows, self._settings)
            if plan.old_collections:
                self._delete_old_collections(plan, deleted)
        except KeyboardInterrupt:
            status = "interrupted"
            logger.warning("interrupted: stopping now; run it again to finish")
        except (
            RunFailed,
            SaveQuestionUnanswered,
            QdrantUnreachable,
            ApiException,
            httpx.HTTPError,
        ) as e:
            status, error = "failed", str(e) or type(e).__name__
            logger.error("failed: %s", error)
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)

        summary = RunSummary(
            status=status,
            collection=collection,
            plan=plan,
            saved=tuple(saved),
            written=written,
            orphans=tuple((row.chunk_id, row.sub_index) for row in orphans),
            retries=self._retrying.retries,
            elapsed=self._clock() - started,
            deleted=tuple(deleted),
            error=error,
        )
        _log_summary(summary)
        return summary

    def _rows(self) -> Iterator[StoredVector]:
        return self._embeddings.ok_vectors(self._model)

    def _delete_old_collections(self, plan: Plan, deleted: list[str]) -> None:
        """Delete `plan.old_collections`, appending each to `deleted`, but
        only once `plan.collection` passes the `index verify --complete`
        checks (Q37). If it doesn't, raise `RunFailed` and keep them all."""
        # Imported here, not at the top to prevent circular imports
        from databricks_docs_vectordb.indexing.verify import verify_collection

        logger.info(
            "verifying %s before deleting the old collection(s) %s",
            plan.collection,
            ", ".join(plan.old_collections),
        )
        report = verify_collection(
            self._qdrant,
            self._settings,
            self._model,
            self._embeddings,
            self._chunks,
            self._manifest,
            complete=True,
        )
        if not report.ok:
            kinds = Counter(f.kind for f in report.violations)
            raise RunFailed(
                f"{plan.collection} failed `index verify --complete` ("
                + ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
                + "), so the old collection(s) were kept: "
                + ", ".join(plan.old_collections)
                + "; `index verify --complete` lists the problems"
            )
        for name in plan.old_collections:
            self._qdrant.delete_collection(name)
            deleted.append(name)
            logger.info("deleted the old collection %s", name)


def configured_dimensions(embeddings: EmbeddingsStore, model: ModelInfo) -> int:
    """The vector size of `model`'s `ok` rows, which names its collection.
    Raises `RunFailed` if there are none, or if they disagree."""
    found = {
        key.dimensions: counts.ok
        for key, counts in embeddings.counts_by_model().items()
        if (key.provider, key.model, key.model_version) == tuple(model) and counts.ok
    }
    name = f"{model.provider}/{model.model}"
    if not found:
        raise RunFailed(
            f"embeddings.sqlite has no ok rows of {name}: run `embed run` first"
        )
    if len(found) > 1:
        raise RunFailed(
            f"the {name} rows disagree on dimensions: "
            + ", ".join(f"{n} rows at {d}" for d, n in sorted(found.items()))
        )
    [dimensions] = found
    assert dimensions is not None  # an ok row always has its dimensions
    return dimensions


def _raise_on_sigterm() -> Any:
    """Make SIGTERM stop the run the way Ctrl-C does. Returns the handler to
    put back, or None where one can't be set (not the main thread)."""

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        return signal.signal(signal.SIGTERM, interrupt)
    except ValueError:
        return None


def _retryable_cause(error: Exception) -> Exception | None:
    """The connection error or timeout behind `error`, or None if it is
    something a retry won't fix."""
    if isinstance(error, ResponseHandlingException):
        error = error.source
    return error if isinstance(error, _RETRYABLE) else None


def _backoff(attempt: int, settings: Settings, unit: float) -> float:
    """Exponential backoff with jitter (50-100% of the nominal delay). Same
    curve as `scraping/fetcher.py`'s and `embedding/runner.py`'s."""
    nominal = min(settings.backoff_max, settings.backoff_base * 2 ** (attempt - 1))
    return nominal * (0.5 + unit / 2)


def _log_summary(summary: RunSummary) -> None:
    """The summary line, like `embed run`'s, then a warning naming any
    orphans."""
    parts = []
    if summary.error:
        parts.append(summary.error)
    if summary.plan is not None:
        plan = summary.plan
        parts.append(
            f"pages: {len(plan.new)} new, {len(plan.changed)} changed, "
            f"{len(plan.removed)} removed, {len(plan.current)} current, "
            f"{len(plan.not_ok)} not ok in the manifest"
        )
        if summary.written is not None:
            parts.append(
                f"{summary.written.points} points upserted, "
                f"{summary.written.pages_deleted} pages' points deleted"
            )
        else:
            parts.append("the write did not finish; the next run picks it up")
        if summary.deleted:
            parts.append(f"old collections deleted: {', '.join(summary.deleted)}")
        kept = [c for c in plan.old_collections if c not in summary.deleted]
        if kept:
            parts.append(f"old collections kept: {', '.join(kept)}")
    if summary.saved:
        parts.append(f"{len(summary.saved)} snapshot(s) saved")
    parts.append(f"{summary.retries} retries")
    parts.append(f"{len(summary.orphans)} orphaned rows skipped")
    logger.info(
        "%s in %.0fs%s: %s",
        summary.status,
        summary.elapsed,
        f" ({summary.collection})" if summary.collection else "",
        "; ".join(parts),
    )
    if summary.orphans:
        named = [f"{c} (sub_index {i})" for c, i in summary.orphans[:ORPHANS_NAMED]]
        more = len(summary.orphans) - len(named)
        logger.warning(
            "skipped %d orphaned row(s), whose chunk_id is not in parsed.sqlite: %s%s",
            len(summary.orphans),
            ", ".join(named),
            f", and {more} more" if more else "",
        )
