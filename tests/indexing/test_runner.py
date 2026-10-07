import array
import logging
import math
import random
import re
from dataclasses import replace

import pytest
from qdrant_client import models

from databricks_docs_vectordb.embedding.store import (
    EmbeddingRow,
    EmbeddingsStore,
    ModelInfo,
    StoredVector,
    pack_embedding,
    unpack_embedding,
)
from databricks_docs_vectordb.indexing.runner import (
    point_id,
    skip_orphans,
    to_point,
    unit_vector,
)
from databricks_docs_vectordb.indexing.store import QdrantStore
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


def row_of(
    url: str = PAGE,
    source: str = "aws-en",
    *,
    index: int = 0,
    sub_index: int = 0,
    vector: list[float] = VECTOR,
) -> StoredVector:
    return StoredVector(
        f"{url}#{index}", sub_index, url, source, len(vector), pack_embedding(vector)
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


# -- point_id / unit_vector / to_point ------------------------------------------


def test_point_id_is_pinned():
    """Changing the namespace or the key's format would change every ID."""
    assert point_id(f"{PAGE}#0", 0) == "3ebb2d4a-9f33-50b5-9e39-63242bb011fe"


def test_point_ids_differ_per_chunk_and_slice():
    ids = {point_id(f"{PAGE}#{i}", s) for i in range(3) for s in range(3)}
    assert len(ids) == 9


def test_the_id_depends_only_on_the_natural_key():
    """A re-embed (new vector, new dimensions) keeps the row's ID."""
    before = to_point(row_of(vector=[3.0, 4.0]), normalize=True)
    after = to_point(row_of(vector=[1.0, 2.0, 2.0]), normalize=True)
    assert before.id == after.id == point_id(f"{PAGE}#0", 0)


def test_every_slice_of_a_split_chunk_gets_its_own_point():
    points = [to_point(row_of(sub_index=i), normalize=True) for i in range(3)]
    assert len({p.id for p in points}) == 3
    assert [p.payload["sub_index"] for p in points] == [0, 1, 2]


def test_a_normalized_vector_is_the_blob_scaled_to_unit_length():
    point = to_point(row_of(vector=[3.0, 4.0]), normalize=True)
    assert point.vector == array.array("f", [0.6, 0.8]).tolist()


def test_without_normalization_the_vector_is_the_blob_unchanged():
    vector = [3.0, 4.0, 0.1]
    point = to_point(row_of(vector=vector), normalize=False)
    assert point.vector == unpack_embedding(pack_embedding(vector)).tolist()


@pytest.mark.parametrize(
    "vector",
    [
        [3.0, 4.0],
        [1e-3] * 1536,
        [random.Random(0).gauss(0, 1) for _ in range(1536)],
    ],
    ids=["3-4-5", "tiny", "gaussian"],
)
def test_unit_vector_is_unit_length_in_float32(vector):
    unit = unit_vector(vector)
    assert array.array("f", unit).tolist() == unit
    assert math.fsum(x * x for x in unit) == pytest.approx(1, abs=1e-6)


def test_a_zero_vector_raises_naming_the_row():
    with pytest.raises(ValueError, match=rf"{re.escape(PAGE)}#0 \(sub_index 2\)"):
        to_point(row_of(sub_index=2, vector=[0.0, 0.0]), normalize=True)


def test_the_payload_is_the_lookup_key_and_the_two_filters():
    point = to_point(row_of(index=4, sub_index=1), normalize=True)
    assert point.payload == {
        "chunk_id": f"{PAGE}#4",
        "sub_index": 1,
        "url": PAGE,
        "section": "aws-en/delta",
    }


@pytest.mark.parametrize(
    ("url", "source", "section"),
    [
        (
            "https://docs.databricks.com/aws/en/sql/language-manual/functions/abs",
            "aws-en",
            "aws-en/sql",
        ),
        ("https://docs.databricks.com/api/jobs/v2/run", "api", "api/jobs"),
        (
            "https://docs.databricks.com/aws/en/introduction",
            "aws-en",
            "aws-en/(top-level)",
        ),
    ],
)
def test_the_section_is_the_source_and_its_section(url, source, section):
    assert to_point(row_of(url, source), normalize=True).payload["section"] == section


@pytest.mark.network
@pytest.mark.parametrize("distance", ["Cosine", "Dot"])
def test_a_normalized_point_reads_back_unchanged_as_float32(
    settings, client, name, distance
):
    """What Step 5's change detection and `index verify` rely on: Qdrant
    stores a unit vector as given, under `Cosine` too, and its components
    read back equal once taken as float32."""
    rng = random.Random(0)
    rows = [
        row_of(index=i, vector=[rng.gauss(0, 1) for _ in range(64)]) for i in range(20)
    ]
    points = [to_point(r, normalize=True) for r in rows]
    with QdrantStore(replace(settings, distance=distance)) as store:
        store.create_collection(name, dimensions=64)
    client.upsert(
        name,
        points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ],
        wait=True,
    )
    back = {
        str(record.id): record
        for record in client.retrieve(
            name, [p.id for p in points], with_vectors=True, with_payload=True
        )
    }
    assert set(back) == {p.id for p in points}
    for p in points:
        assert array.array("f", back[p.id].vector) == array.array("f", p.vector)
        assert back[p.id].payload == p.payload
