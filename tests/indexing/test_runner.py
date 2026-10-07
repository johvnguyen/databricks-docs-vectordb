import logging

import pytest

from databricks_docs_vectordb.embedding.store import (
    EmbeddingRow,
    EmbeddingsStore,
    ModelInfo,
    StoredVector,
    pack_embedding,
)
from databricks_docs_vectordb.indexing.runner import skip_orphans
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore

MODEL = ModelInfo("openai", "text-embedding-3-small", None)
PAGE = "https://docs.databricks.com/aws/en/delta/"
VECTOR = [0.6, 0.8]


@pytest.fixture
def warnings_logged(caplog):
    caplog.set_level(logging.WARNING, logger="databricks_docs_vectordb.indexing")
    return caplog


def stored(index: int, sub_index: int = 0) -> StoredVector:
    return StoredVector(
        f"{PAGE}#{index}", sub_index, PAGE, "aws-en", 2, pack_embedding(VECTOR)
    )


def make_chunk(index: int) -> Chunk:
    text = f"chunk {index} text"
    return Chunk(
        chunk_id=f"{PAGE}#{index}",
        url=PAGE,
        source="aws-en",
        cloud="aws",
        chunk_index=index,
        prev_chunk_id=None,
        next_chunk_id=None,
        heading_path="",
        section_path="Delta Lake",
        doc_title="Delta Lake",
        description=None,
        text=text,
        char_count=len(text),
        token_count=None,
        token_count_model=None,
        is_oversized=False,
        last_updated=None,
        page_sha256="ab" * 32,
        fetched_at=None,
    )


def make_row(
    index: int, *, status: str = "ok", model: ModelInfo = MODEL
) -> EmbeddingRow:
    ok = status == "ok"
    return EmbeddingRow(
        chunk_id=f"{PAGE}#{index}",
        sub_index=0,
        sub_count=1,
        url=PAGE,
        source="aws-en",
        cloud="aws",
        chunk_index=index,
        prev_chunk_id=None,
        next_chunk_id=None,
        heading_path="",
        section_path="Delta Lake",
        doc_title="Delta Lake",
        description=None,
        last_updated=None,
        page_sha256="ab" * 32,
        chunk_text_hash="cd" * 32,
        embedded_text=f"Delta Lake\n\nchunk {index} text",
        token_count=8,
        provider=model.provider,
        model=model.model,
        model_version=model.model_version,
        dimensions=len(VECTOR) if ok else None,
        embedding=pack_embedding(VECTOR) if ok else None,
        status=status,
        attempts=1,
        error=None if ok else "boom",
    )


# -- skip_orphans ----------------------------------------------------------------


def test_rows_still_in_parsed_pass_through_in_order(warnings_logged):
    rows = [stored(0), stored(1, 0), stored(1, 1)]
    skipped: list[StoredVector] = []
    assert list(skip_orphans(rows, {f"{PAGE}#0", f"{PAGE}#1"}, skipped)) == rows
    assert skipped == []
    assert warnings_logged.records == []


def test_an_orphan_is_skipped_with_a_warning_naming_it(warnings_logged):
    orphan = stored(7, sub_index=1)
    skipped: list[StoredVector] = []
    rows = [stored(0), orphan, stored(1)]
    assert list(skip_orphans(rows, {f"{PAGE}#0", f"{PAGE}#1"}, skipped)) == [
        stored(0),
        stored(1),
    ]
    assert skipped == [orphan]
    [record] = warnings_logged.records
    assert record.levelno == logging.WARNING
    assert f"{PAGE}#7 (sub_index 1)" in record.getMessage()


def test_every_slice_of_an_orphaned_split_chunk_is_skipped(warnings_logged):
    slices = [stored(7, i) for i in range(3)]
    skipped: list[StoredVector] = []
    assert list(skip_orphans(slices, set(), skipped)) == []
    assert skipped == slices
    assert len(warnings_logged.records) == 3


def test_skipped_is_optional(warnings_logged):
    assert list(skip_orphans([stored(7)], set())) == []
    assert len(warnings_logged.records) == 1


# -- with both stores -------------------------------------------------------------


def test_only_the_orphan_warns_among_the_rows_left_out(tmp_path, warnings_logged):
    """Step 4's Verify: a `failed` row, another model's row and an orphan
    are each left out, and only the orphan produces a warning."""
    with ChunkStore(tmp_path / "parsed.sqlite") as chunks:
        # The page was re-parsed into three chunks, leaving #3 behind.
        chunks.replace_page_chunks(PAGE, [make_chunk(i) for i in range(3)])
        chunk_ids = chunks.chunk_ids()

    other_model = ModelInfo("openai", "text-embedding-3-large", None)
    with EmbeddingsStore(tmp_path / "embeddings.sqlite") as embeddings:
        embeddings.upsert(f"{PAGE}#0", [make_row(0)])
        embeddings.upsert(f"{PAGE}#1", [make_row(1, status="failed")])
        embeddings.upsert(f"{PAGE}#2", [make_row(2, model=other_model)])
        embeddings.upsert(f"{PAGE}#3", [make_row(3)])
        kept = list(skip_orphans(embeddings.ok_vectors(MODEL), chunk_ids))

    assert [row.chunk_id for row in kept] == [f"{PAGE}#0"]
    [record] = warnings_logged.records
    assert f"{PAGE}#3" in record.getMessage()
