from pathlib import Path

import pytest

from databricks_docs_vectordb.embedding.runner import text_hash
from databricks_docs_vectordb.embedding.store import (
    EmbeddingRow,
    EmbeddingsStore,
    ModelInfo,
    pack_embedding,
)
from databricks_docs_vectordb.embedding.verify import (
    LIST_LIMIT,
    format_report,
    verify_embeddings,
)
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore

MODEL = ModelInfo("openai", "text-embedding-3-small", None)
AWS = "https://docs.databricks.com/aws/en/"


def page_url(n: int) -> str:
    return f"{AWS}p{n:03d}"


def make_chunk(url: str, index: int = 0, text: str | None = None) -> Chunk:
    text = text if text is not None else f"text of {url}#{index}"
    return Chunk(
        chunk_id=f"{url}#{index}",
        url=url,
        source="aws-en",
        cloud="aws",
        chunk_index=index,
        prev_chunk_id=None,
        next_chunk_id=None,
        heading_path="Heading",
        section_path="Section",
        doc_title="Title",
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


def make_rows(
    chunk: Chunk,
    n: int = 1,
    *,
    status: str = "ok",
    model: ModelInfo = MODEL,
    dimensions: int = 3,
    chunk_hash: str | None = None,
) -> list[EmbeddingRow]:
    ok = status == "ok"
    return [
        EmbeddingRow(
            chunk_id=chunk.chunk_id,
            sub_index=i,
            sub_count=n,
            url=chunk.url,
            source=chunk.source,
            cloud=chunk.cloud,
            chunk_index=chunk.chunk_index,
            prev_chunk_id=None,
            next_chunk_id=None,
            heading_path=chunk.heading_path,
            section_path=chunk.section_path,
            doc_title=chunk.doc_title,
            description=None,
            last_updated=None,
            page_sha256=chunk.page_sha256,
            chunk_text_hash=chunk_hash or text_hash(chunk.text),
            embedded_text=f"slice {i}",
            token_count=2,
            provider=model.provider,
            model=model.model,
            model_version=model.model_version,
            dimensions=dimensions if ok else None,
            embedding=pack_embedding([0.5] * dimensions) if ok else None,
            status=status,
            attempts=1,
            error=None if ok else "boom",
        )
        for i in range(n)
    ]


@pytest.fixture
def stores(tmp_path: Path):
    with (
        ChunkStore(tmp_path / "parsed.sqlite") as chunks,
        EmbeddingsStore(tmp_path / "embeddings.sqlite") as embeddings,
    ):
        yield chunks, embeddings


def seed(stores, pages: int = 3, *, split: int = 1) -> list[Chunk]:
    """`pages` one-chunk pages, each embedded cleanly (`split` rows each)."""
    chunks, embeddings = stores
    made = []
    for p in range(pages):
        chunk = make_chunk(page_url(p))
        chunks.replace_page_chunks(chunk.url, [chunk])
        embeddings.upsert(chunk.chunk_id, make_rows(chunk, split))
        made.append(chunk)
    return made


def verify(stores, **kwargs):
    chunks, embeddings = stores
    return verify_embeddings(chunks, embeddings, MODEL, **kwargs)


def kinds(report) -> list[str]:
    return [f.kind for f in report.violations]


def sql(stores, statement: str, *params) -> None:
    """Corrupt the store directly: `upsert` refuses malformed row sets."""
    _, embeddings = stores
    with embeddings._conn:
        embeddings._conn.execute(statement, params)


# =========================================================================
# clean stores
# =========================================================================


def test_a_clean_store_has_no_violations(stores):
    seed(stores, 3)
    report = verify(stores, complete=True)
    assert report.ok
    assert report.chunks == 3
    assert report.coverage["ok"] == 3
    assert report.gaps == 0


def test_a_clean_store_with_split_chunks_has_no_violations(stores):
    seed(stores, 2, split=4)
    report = verify(stores, complete=True)
    assert report.ok
    assert report.counts.rows == 8


def test_a_clean_store_formats_as_ok(stores):
    seed(stores, 2)
    text = format_report(verify(stores), "embeddings.sqlite")
    assert text.endswith("OK")
    assert "coverage (chunks): 2 ok, 0 rejected, 0 failed, 0 not-embedded" in text


def test_empty_stores_are_clean(stores):
    report = verify(stores, complete=True)
    assert report.ok and report.chunks == 0


def test_no_embeddings_store_means_every_chunk_is_not_embedded(stores):
    chunks, _ = stores
    chunks.replace_page_chunks(page_url(0), [make_chunk(page_url(0))])
    report = verify_embeddings(chunks, None, MODEL)
    assert report.ok
    assert report.coverage["not-embedded"] == 1
    assert report.counts.rows == 0


# =========================================================================
# coverage: gaps are notes by default, violations with complete
# =========================================================================


def test_a_missing_row_set_is_a_note_by_default(stores):
    chunks, _ = stores
    seed(stores, 2)
    chunks.replace_page_chunks(page_url(9), [make_chunk(page_url(9))])
    report = verify(stores)
    assert report.ok
    assert report.coverage["not-embedded"] == 1
    text = format_report(report, "embeddings.sqlite")
    assert "note: 1 chunk(s) not embedded or not current" in text


def test_a_missing_row_set_is_a_violation_with_complete(stores):
    chunks, _ = stores
    seed(stores, 2)
    chunks.replace_page_chunks(page_url(9), [make_chunk(page_url(9))])
    report = verify(stores, complete=True)
    assert kinds(report) == ["not-embedded"]
    assert report.violations[0].where == f"{page_url(9)}#0"
    assert "note:" not in format_report(report, "embeddings.sqlite")


def test_a_stale_hash_is_flagged(stores):
    chunks, _ = stores
    seed(stores, 2)
    # re-parsed with new text since it was embedded
    chunks.replace_page_chunks(page_url(1), [make_chunk(page_url(1), text="new text")])
    assert verify(stores).coverage["stale-hash"] == 1
    report = verify(stores, complete=True)
    assert kinds(report) == ["stale-hash"]
    assert report.violations[0].where == f"{page_url(1)}#0"


def test_a_model_mismatch_is_flagged(stores):
    chunks, embeddings = stores
    seed(stores, 1)
    chunk = make_chunk(page_url(5))
    chunks.replace_page_chunks(chunk.url, [chunk])
    large = ModelInfo("openai", "text-embedding-3-large", None)
    embeddings.upsert(chunk.chunk_id, make_rows(chunk, model=large))

    report = verify(stores, complete=True)
    assert kinds(report) == ["model-mismatch"]
    assert "openai/text-embedding-3-large/-" in report.violations[0].detail


@pytest.mark.parametrize("status", ["failed", "rejected"])
def test_failed_and_rejected_row_sets_count_as_covered_even_when_complete(
    stores, status
):
    chunks, embeddings = stores
    chunk = make_chunk(page_url(0))
    chunks.replace_page_chunks(chunk.url, [chunk])
    embeddings.upsert(chunk.chunk_id, make_rows(chunk, 2, status=status))
    report = verify(stores, complete=True)
    assert report.ok
    assert report.coverage[status] == 1


def test_a_stale_failed_chunk_counts_as_stale_not_failed(stores):
    chunks, embeddings = stores
    chunk = make_chunk(page_url(0))
    chunks.replace_page_chunks(chunk.url, [chunk])
    embeddings.upsert(
        chunk.chunk_id, make_rows(chunk, status="failed", chunk_hash="0" * 64)
    )
    assert verify(stores).coverage == {"stale-hash": 1}


# =========================================================================
# integrity: always violations
# =========================================================================


def test_an_orphaned_chunk_id_is_a_violation(stores):
    chunks, _ = stores
    seed(stores, 3)
    # page 1 re-parsed into... nothing; its embeddings rows stay behind
    chunks.replace_page_chunks(page_url(1), [])
    report = verify(stores)
    assert kinds(report) == ["orphaned-rows"]
    assert report.violations[0].where == f"{page_url(1)}#0"
    assert report.chunks == 2


def test_orphans_from_a_page_reparsed_into_fewer_chunks(stores):
    chunks, embeddings = stores
    url = page_url(0)
    three = [make_chunk(url, i) for i in range(3)]
    chunks.replace_page_chunks(url, three)
    for c in three:
        embeddings.upsert(c.chunk_id, make_rows(c))
    chunks.replace_page_chunks(url, three[:1])

    report = verify(stores)
    assert kinds(report) == ["orphaned-rows", "orphaned-rows"]
    assert [f.where for f in report.violations] == [f"{url}#1", f"{url}#2"]


def test_a_corrupt_blob_length_is_a_violation(stores):
    made = seed(stores, 2)
    sql(
        stores,
        "UPDATE embeddings SET embedding = ? WHERE chunk_id = ?",
        b"\x00" * 10,
        made[0].chunk_id,
    )
    report = verify(stores)
    assert kinds(report) == ["bad-blob-length"]
    assert "10 bytes, expected 12" in report.violations[0].detail


def test_dimensions_disagreeing_with_the_blob_is_a_violation(stores):
    made = seed(stores, 1)
    sql(
        stores,
        "UPDATE embeddings SET dimensions = 4 WHERE chunk_id = ?",
        made[0].chunk_id,
    )
    assert kinds(verify(stores)) == ["bad-blob-length"]


def test_a_gap_in_sub_index_is_a_violation(stores):
    made = seed(stores, 1, split=3)
    sql(
        stores,
        "DELETE FROM embeddings WHERE chunk_id = ? AND sub_index = 1",
        made[0].chunk_id,
    )
    report = verify(stores)
    assert kinds(report) == ["bad-split"]
    assert "sub_index [0, 2]" in report.violations[0].detail


def test_a_wrong_sub_count_is_a_violation(stores):
    made = seed(stores, 1, split=2)
    sql(
        stores,
        "UPDATE embeddings SET sub_count = 5 WHERE chunk_id = ?",
        made[0].chunk_id,
    )
    assert kinds(verify(stores)) == ["bad-split"]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("chunk_text_hash", "f" * 64),
        ("model", "text-embedding-3-large"),
        ("model_version", "2027-01"),
    ],
)
def test_rows_of_one_chunk_disagreeing_is_a_violation(stores, column, value):
    made = seed(stores, 1, split=3)
    sql(
        stores,
        f"UPDATE embeddings SET {column} = ? WHERE chunk_id = ? AND sub_index = 2",
        value,
        made[0].chunk_id,
    )
    report = verify(stores)
    assert "inconsistent-rows" in kinds(report)


def test_mixed_statuses_within_a_chunk_is_a_violation(stores):
    made = seed(stores, 1, split=2)
    sql(
        stores,
        "UPDATE embeddings SET status = 'failed', embedding = NULL, dimensions = NULL"
        " WHERE chunk_id = ? AND sub_index = 1",
        made[0].chunk_id,
    )
    report = verify(stores)
    assert kinds(report) == ["inconsistent-rows"]
    assert "status" in report.violations[0].detail


def test_mixed_dimensions_across_ok_rows_is_a_violation(stores):
    made = seed(stores, 1, split=2)
    sql(
        stores,
        "UPDATE embeddings SET dimensions = 4, embedding = ?"
        " WHERE chunk_id = ? AND sub_index = 1",
        pack_embedding([0.5] * 4),
        made[0].chunk_id,
    )
    report = verify(stores)
    assert kinds(report) == ["inconsistent-rows"]
    assert "dimensions" in report.violations[0].detail


def test_a_failed_row_carrying_a_vector_is_a_violation(stores):
    chunks, embeddings = stores
    chunk = make_chunk(page_url(0))
    chunks.replace_page_chunks(chunk.url, [chunk])
    embeddings.upsert(chunk.chunk_id, make_rows(chunk, status="failed"))
    sql(stores, "UPDATE embeddings SET embedding = ?", pack_embedding([1.0]))
    assert kinds(verify(stores)) == ["unexpected-vector"]


def test_integrity_violations_are_flagged_without_complete(stores):
    seed(stores, 1)
    sql(stores, "UPDATE embeddings SET embedding = ?", b"\x00")
    assert kinds(verify(stores, complete=False)) == ["bad-blob-length"]


def test_every_violation_is_reported_together(stores):
    chunks, _ = stores
    made = seed(stores, 5, split=2)
    chunks.replace_page_chunks(page_url(0), [])  # orphan
    chunks.replace_page_chunks(
        page_url(1), [make_chunk(page_url(1), text="new")]
    )  # stale
    sql(
        stores,
        "UPDATE embeddings SET embedding = ? WHERE chunk_id = ?",
        b"x",
        made[2].chunk_id,
    )
    sql(
        stores,
        "DELETE FROM embeddings WHERE chunk_id = ? AND sub_index = 0",
        made[3].chunk_id,
    )
    chunks.replace_page_chunks(page_url(9), [make_chunk(page_url(9))])  # missing

    report = verify(stores, complete=True)
    assert sorted(set(kinds(report))) == [
        "bad-blob-length",
        "bad-split",
        "not-embedded",
        "orphaned-rows",
        "stale-hash",
    ]
    text = format_report(report, "embeddings.sqlite")
    assert text.endswith(f"FAILED: {len(report.violations)} violations")


# =========================================================================
# reporting
# =========================================================================


def test_counts_by_status_and_by_model_are_reported(stores):
    chunks, embeddings = stores
    seed(stores, 2, split=2)
    failed = make_chunk(page_url(7))
    chunks.replace_page_chunks(failed.url, [failed])
    embeddings.upsert(failed.chunk_id, make_rows(failed, status="failed"))
    old = make_chunk(page_url(8))
    chunks.replace_page_chunks(old.url, [old])
    large = ModelInfo("openai", "text-embedding-3-large", None)
    embeddings.upsert(old.chunk_id, make_rows(old, model=large, dimensions=5))

    text = format_report(verify(stores), "embeddings.sqlite")
    assert (
        "rows by status: 5 ok, 1 failed, 0 rejected (6 row(s), 4 chunk_id(s))" in text
    )
    assert (
        "  openai/text-embedding-3-large/- (5): 1 chunk(s), 1 row(s): "
        "1 ok, 0 failed, 0 rejected\n"
    ) in text
    assert (
        "  openai/text-embedding-3-small/- (3): 2 chunk(s), 4 row(s): "
        "4 ok, 0 failed, 0 rejected  <- configured"
    ) in text
    assert "  openai/text-embedding-3-small/- (-): 1 chunk(s), 1 row(s): " in text
    assert (
        "coverage (chunks): 2 ok, 0 rejected, 1 failed, 0 not-embedded, "
        "0 stale-hash, 1 model-mismatch"
    ) in text


def test_long_lists_of_one_kind_are_truncated(stores):
    chunks, _ = stores
    for p in range(LIST_LIMIT + 5):
        chunks.replace_page_chunks(page_url(p), [make_chunk(page_url(p))])
    report = verify(stores, complete=True)
    text = format_report(report, "embeddings.sqlite")
    assert len(report.violations) == LIST_LIMIT + 5
    assert text.count("[not-embedded] https://") == LIST_LIMIT
    assert "[not-embedded] ... and 5 more" in text
