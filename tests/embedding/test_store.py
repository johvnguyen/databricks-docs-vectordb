import sqlite3
from pathlib import Path

import pytest

from databricks_docs_vectordb.embedding.store import (
    SCHEMA_VERSION,
    EmbeddingCounts,
    EmbeddingRow,
    EmbeddingsStore,
    EmbeddingSummary,
    ModelInfo,
    ModelKey,
    pack_embedding,
    unpack_embedding,
)

CHUNK_A = "https://docs.databricks.com/aws/en/delta/#0"
CHUNK_B = "https://docs.databricks.com/aws/en/delta/#1"
T1, T2 = "2026-09-27T01:00:00Z", "2026-09-27T02:00:00Z"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "embeddings.sqlite"


@pytest.fixture
def store(db_path):
    with EmbeddingsStore(db_path) as s:
        yield s


def make_row(
    chunk_id: str, sub_index: int = 0, sub_count: int = 1, **overrides
) -> EmbeddingRow:
    defaults = {
        "chunk_id": chunk_id,
        "sub_index": sub_index,
        "sub_count": sub_count,
        "url": chunk_id.split("#")[0],
        "source": "aws-en",
        "cloud": "aws",
        "chunk_index": 0,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "heading_path": "Delta Lake",
        "section_path": "Tables > Table formats > Delta Lake",
        "doc_title": "What is Delta Lake?",
        "description": "Delta Lake is the optimized storage layer.",
        "last_updated": "2026-09-11",
        "page_sha256": "ab" * 32,
        "chunk_text_hash": "cd" * 32,
        "embedded_text": "What is Delta Lake?\nDelta Lake\n\nchunk text",
        "token_count": 10,
        "provider": "openai",
        "model": "text-embedding-3-small",
        "model_version": None,
        "dimensions": 4,
        "embedding": pack_embedding([0.1, 0.2, 0.3, 0.4]),
        "status": "ok",
        "attempts": 1,
        "error": None,
    }
    defaults.update(overrides)
    return EmbeddingRow(**defaults)


def make_split(chunk_id: str, n: int, **overrides) -> list[EmbeddingRow]:
    """`n` sub_index rows for `chunk_id`, sharing one sub_count."""
    return [make_row(chunk_id, i, sub_count=n, **overrides) for i in range(n)]


# -- schema and persistence ------------------------------------------------


def test_opening_creates_the_database_and_its_parent_directory(db_path):
    assert not db_path.parent.exists()
    with EmbeddingsStore(db_path) as s:
        assert s.counts() == EmbeddingCounts(0, 0, 0, 0, 0)
    assert db_path.exists()


def test_reopening_from_disk_returns_the_same_rows(db_path):
    with EmbeddingsStore(db_path) as s:
        s.upsert(CHUNK_A, make_split(CHUNK_A, 2), now=T1)
    with EmbeddingsStore(db_path) as s:
        assert s.counts() == EmbeddingCounts(1, 2, 2, 0, 0)
        assert s.get_hash(CHUNK_A) == "cd" * 32


def test_schema_version_is_recorded_and_checked(db_path):
    with EmbeddingsStore(db_path):
        pass
    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="schema version 99"):
        EmbeddingsStore(db_path)


def test_status_column_rejects_unknown_values(store):
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO embeddings (chunk_id, sub_index, sub_count, url,"
            " source, cloud, chunk_index, heading_path, section_path,"
            " doc_title, page_sha256, chunk_text_hash, embedded_text,"
            " token_count, provider, model, status, embedded_at)"
            " VALUES ('x', 0, 1, 'u', 's', 'c', 0, 'h', 'p', 'd', 'sha',"
            " 'hash', 'text', 1, 'openai', 'm', 'pending', 'now')"
        )


def test_ok_status_without_a_vector_is_rejected_by_schema(store):
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO embeddings (chunk_id, sub_index, sub_count, url,"
            " source, cloud, chunk_index, heading_path, section_path,"
            " doc_title, page_sha256, chunk_text_hash, embedded_text,"
            " token_count, provider, model, dimensions, embedding, status,"
            " embedded_at)"
            " VALUES ('x', 0, 1, 'u', 's', 'c', 0, 'h', 'p', 'd', 'sha',"
            " 'hash', 'text', 1, 'openai', 'm', 4, NULL, 'ok', 'now')"
        )


def test_failed_row_with_no_vector_is_allowed(store):
    store.upsert(
        CHUNK_A,
        [
            make_row(
                CHUNK_A,
                status="failed",
                dimensions=None,
                embedding=None,
                attempts=3,
                error="rate limited",
            )
        ],
    )
    summary = store.all_summaries()[0]
    assert summary.status == "failed"
    assert summary.dimensions is None
    assert summary.blob_len is None


# -- upsert -----------------------------------------------------------------


def test_replacing_with_a_different_sub_count_leaves_no_orphaned_rows(store):
    store.upsert(CHUNK_A, make_split(CHUNK_A, 3), now=T1)
    assert store.counts() == EmbeddingCounts(1, 3, 3, 0, 0)

    store.upsert(CHUNK_A, make_split(CHUNK_A, 1), now=T2)
    assert store.counts() == EmbeddingCounts(1, 1, 1, 0, 0)
    summaries = store.all_summaries()
    assert len(summaries) == 1
    assert summaries[0].sub_index == 0
    assert summaries[0].sub_count == 1


def test_replacing_with_empty_rows_clears_the_chunk(store):
    store.upsert(CHUNK_A, make_split(CHUNK_A, 2), now=T1)
    store.upsert(CHUNK_A, [], now=T2)
    assert store.get_hash(CHUNK_A) is None
    assert store.counts() == EmbeddingCounts(0, 0, 0, 0, 0)


def test_other_chunks_are_untouched(store):
    store.upsert(CHUNK_A, make_split(CHUNK_A, 2), now=T1)
    store.upsert(CHUNK_B, make_split(CHUNK_B, 1), now=T1)
    store.upsert(CHUNK_A, make_split(CHUNK_A, 1), now=T2)
    assert store.counts() == EmbeddingCounts(2, 2, 2, 0, 0)
    assert store.get_hash(CHUNK_B) is not None


def test_embedded_at_is_stamped_by_the_store_not_the_caller(store):
    store.upsert(CHUNK_A, [make_row(CHUNK_A)], now=T1)
    row = store._conn.execute(
        "SELECT embedded_at FROM embeddings WHERE chunk_id = ?", (CHUNK_A,)
    ).fetchone()
    assert row["embedded_at"] == T1


def test_mismatched_chunk_id_is_rejected(store):
    bad = make_row(CHUNK_B)
    with pytest.raises(ValueError, match="expected"):
        store.upsert(CHUNK_A, [bad])


def test_wrong_sub_count_is_rejected(store):
    # sub_count says 2 but only one row is passed.
    bad = make_row(CHUNK_A, sub_index=0, sub_count=2)
    with pytest.raises(ValueError, match="sub_count"):
        store.upsert(CHUNK_A, [bad])


def test_duplicate_sub_index_is_rejected(store):
    a = make_row(CHUNK_A, sub_index=0, sub_count=2)
    b = make_row(CHUNK_A, sub_index=0, sub_count=2)
    with pytest.raises(ValueError, match="0..1"):
        store.upsert(CHUNK_A, [a, b])


def test_gap_in_sub_index_is_rejected(store):
    a = make_row(CHUNK_A, sub_index=0, sub_count=2)
    b = make_row(CHUNK_A, sub_index=2, sub_count=2)
    with pytest.raises(ValueError, match="0..1"):
        store.upsert(CHUNK_A, [a, b])


def test_nothing_is_written_when_validation_fails(store):
    store.upsert(CHUNK_A, make_split(CHUNK_A, 2), now=T1)
    bad = [make_row(CHUNK_A, sub_index=0, sub_count=2)] * 1  # missing sub_index 1
    with pytest.raises(ValueError):
        store.upsert(CHUNK_A, bad)
    # The original two rows are still there -- the failed replace didn't even
    # get as far as deleting them.
    assert store.counts() == EmbeddingCounts(1, 2, 2, 0, 0)


# -- get_hash / get_model_info ----------------------------------------------


def test_get_hash_round_trips(store):
    store.upsert(CHUNK_A, [make_row(CHUNK_A, chunk_text_hash="deadbeef")])
    assert store.get_hash(CHUNK_A) == "deadbeef"


def test_get_hash_returns_none_for_an_unknown_chunk(store):
    assert store.get_hash("nope") is None


def test_get_hash_is_shared_across_sub_index_rows(store):
    store.upsert(CHUNK_A, make_split(CHUNK_A, 3, chunk_text_hash="shared-hash"))
    assert store.get_hash(CHUNK_A) == "shared-hash"


def test_get_model_info_round_trips(store):
    store.upsert(
        CHUNK_A,
        [
            make_row(
                CHUNK_A,
                provider="openai",
                model="text-embedding-3-large",
                model_version="2026-01-01",
            )
        ],
    )
    assert store.get_model_info(CHUNK_A) == ModelInfo(
        "openai", "text-embedding-3-large", "2026-01-01"
    )


def test_get_model_info_returns_none_for_an_unknown_chunk(store):
    assert store.get_model_info("nope") is None


# -- all_summaries -----------------------------------------------------------


def test_all_summaries_is_empty_for_an_empty_store(store):
    assert store.all_summaries() == []


def test_all_summaries_covers_every_row_including_splits(store):
    store.upsert(CHUNK_A, make_split(CHUNK_A, 3), now=T1)
    store.upsert(CHUNK_B, make_split(CHUNK_B, 1), now=T1)
    summaries = store.all_summaries()
    assert len(summaries) == 4
    assert {s.chunk_id for s in summaries} == {CHUNK_A, CHUNK_B}
    assert sorted(s.sub_index for s in summaries if s.chunk_id == CHUNK_A) == [0, 1, 2]


def test_all_summaries_blob_len_matches_the_packed_vector(store):
    store.upsert(
        CHUNK_A,
        [
            make_row(
                CHUNK_A, dimensions=4, embedding=pack_embedding([1.0, 2.0, 3.0, 4.0])
            )
        ],
    )
    summary = store.all_summaries()[0]
    assert summary.blob_len == 16  # 4 float32s
    assert summary.dimensions == 4


def test_all_summaries_after_a_replace_reflects_only_the_current_rows(store):
    store.upsert(CHUNK_A, make_split(CHUNK_A, 3), now=T1)
    store.upsert(CHUNK_A, make_split(CHUNK_A, 1), now=T2)
    assert len(store.all_summaries()) == 1


# -- counts / counts_by_model -------------------------------------------------


def test_counts_across_chunks_and_splits(store):
    store.upsert(CHUNK_A, make_split(CHUNK_A, 2), now=T1)
    store.upsert(CHUNK_B, make_split(CHUNK_B, 1), now=T1)
    assert store.counts() == EmbeddingCounts(
        chunk_ids=2, rows=3, ok=3, failed=0, rejected=0
    )


def test_counts_by_status(store):
    store.upsert(CHUNK_A, [make_row(CHUNK_A, status="ok")], now=T1)
    store.upsert(
        CHUNK_B,
        [
            make_row(
                CHUNK_B,
                status="failed",
                dimensions=None,
                embedding=None,
                attempts=5,
                error="timeout",
            )
        ],
        now=T1,
    )
    assert store.counts() == EmbeddingCounts(
        chunk_ids=2, rows=2, ok=1, failed=1, rejected=0
    )


def test_counts_by_model_groups_by_provider_model_version_dimensions(store):
    store.upsert(
        CHUNK_A,
        [
            make_row(
                CHUNK_A,
                provider="openai",
                model="text-embedding-3-small",
                model_version=None,
                dimensions=4,
            )
        ],
        now=T1,
    )
    store.upsert(
        CHUNK_B,
        [
            make_row(
                CHUNK_B,
                provider="openai",
                model="text-embedding-3-large",
                model_version=None,
                dimensions=4,
            )
        ],
        now=T1,
    )
    by_model = store.counts_by_model()
    key_small = ModelKey("openai", "text-embedding-3-small", None, 4)
    key_large = ModelKey("openai", "text-embedding-3-large", None, 4)
    assert by_model[key_small] == EmbeddingCounts(1, 1, 1, 0, 0)
    assert by_model[key_large] == EmbeddingCounts(1, 1, 1, 0, 0)


# -- pack/unpack_embedding ----------------------------------------------------


def test_pack_and_unpack_round_trip():
    vector = [0.5, -1.25, 3.0, 0.0]
    blob = pack_embedding(vector)
    assert isinstance(blob, bytes)
    assert len(blob) == 16
    restored = unpack_embedding(blob)
    assert list(restored) == pytest.approx(vector)


def test_embedding_summary_is_a_plain_namedtuple():
    # Sanity check the type used above in equality assertions.
    assert EmbeddingSummary._fields == (
        "chunk_id",
        "sub_index",
        "sub_count",
        "status",
        "provider",
        "model",
        "model_version",
        "dimensions",
        "blob_len",
        "chunk_text_hash",
    )
