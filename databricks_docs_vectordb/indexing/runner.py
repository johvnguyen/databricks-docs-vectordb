"""`index run`: load the configured model's embeddings rows into Qdrant
(vector_db_integration.md's "Idempotency, deletions and failure handling").

So far this holds the read side (Step 4), shared with `index verify`: the
orphan check, and the mapping from an embeddings row to the point it
becomes. Then the loader's first stage (Step 5): comparing the rows with
what Qdrant holds, to decide what a run is to do before it writes anything.
"""

import array
import hashlib
import logging
import math
import uuid
from collections.abc import Container, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from databricks_docs_vectordb.embedding.store import StoredVector, unpack_embedding
from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.store import QdrantStore
from databricks_docs_vectordb.parsing.report import section_of
from databricks_docs_vectordb.scraping.config import SOURCES, Source

logger = logging.getLogger(__name__)

# Every point ID is derived from this, so changing it changes them all (Q7).
POINT_ID_NAMESPACE = uuid.UUID("2405f98a-c138-459f-b780-53782478d987")


class Point(NamedTuple):
    """One embeddings row as a Qdrant point ("Record and collection
    schema"): what `index run` writes and what `index verify` expects."""

    id: str
    vector: list[float]
    payload: dict[str, str | int]  # chunk_id, sub_index, url, section


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
    def replaces_anything(self) -> bool:
        """Whether this run would replace or remove a page, or delete a
        collection: when the save question is asked (Q29, Q40)."""
        return bool(self.changed or self.removed or self.old_collections)


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
