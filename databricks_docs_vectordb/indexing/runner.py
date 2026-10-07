"""`index run`: load the configured model's embeddings rows into Qdrant
(vector_db_integration.md's "Idempotency, deletions and failure handling").

So far this holds the read side (Step 4), shared with `index verify`: the
orphan check, and the mapping from an embeddings row to the point it
becomes. The loader itself is Step 5.
"""

import array
import logging
import math
import uuid
from collections.abc import Container, Iterable, Iterator, Sequence
from typing import NamedTuple

from databricks_docs_vectordb.embedding.store import StoredVector, unpack_embedding
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
