"""`index run`: load the configured model's embeddings rows into Qdrant
(vector_db_integration.md's "Idempotency, deletions and failure handling").

So far this holds the read side (Step 4), shared with `index verify`: the
orphan check, and the mapping from an embeddings row to the point it
becomes. Then the loader (Step 5): comparing the rows with what Qdrant
holds to decide what a run is to do, the question whether to save what it
would change, both before anything is written, and then the writes.
"""

import array
import hashlib
import logging
import math
import sys
import uuid
from collections.abc import Callable, Container, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import NamedTuple

from databricks_docs_vectordb.embedding.store import StoredVector, unpack_embedding
from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.store import Point, QdrantStore
from databricks_docs_vectordb.parsing.report import section_of
from databricks_docs_vectordb.scraping.config import SOURCES, Source

logger = logging.getLogger(__name__)

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


def to_point(
    row: StoredVector, *, normalize: bool, sources: Sequence[Source] = SOURCES
) -> Point:
    """The point `row` becomes: its ID, its vector (the BLOB unchanged
    unless `normalize`), and the four payload fields (Q9).

    `section` is `<source>/<section>`, the section as `parse status`
    reports it (`section_of`): `aws-en/sql`, `api/jobs`, or
    `aws-en/(top-level)` for a top-level page.
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
            "section": f"{row.source}/{section_of(row.url, row.source, sources)}",
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
                None if point.vector is None else _vector_hash(point.vector)
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
            page[point.id] = _vector_hash(point.vector)

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


def _vector_hash(vector: Sequence[float]) -> bytes:
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
    if plan.refresh:
        if plan.exists:
            qdrant.delete_collection(plan.collection)
        qdrant.create_collection(plan.collection, dimensions=plan.dimensions)
    else:
        if not plan.exists:
            qdrant.create_collection(plan.collection, dimensions=plan.dimensions)
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
