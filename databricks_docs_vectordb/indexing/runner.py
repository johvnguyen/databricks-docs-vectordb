"""`index run`: load the configured model's embeddings rows into Qdrant
(vector_db_integration.md's "Idempotency, deletions and failure handling").

So far this holds the read side's orphan check (Step 4), shared with
`index verify`; the loader itself is Step 5.
"""

import logging
from collections.abc import Container, Iterable, Iterator

from databricks_docs_vectordb.embedding.store import StoredVector

logger = logging.getLogger(__name__)


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
