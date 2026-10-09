import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from databricks_docs_vectordb.parsing.store import (
    SCHEMA_VERSION,
    Chunk,
    ChunkCounts,
    ChunkStore,
    ChunkSummary,
    ChunkText,
)

URL_A = "https://docs.databricks.com/aws/en/delta/"
URL_B = "https://docs.databricks.com/aws/en/delta/merge"
T1, T2 = "2026-09-23T01:00:00Z", "2026-09-23T02:00:00Z"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "parsed.sqlite"


@pytest.fixture
def store(db_path):
    with ChunkStore(db_path) as s:
        yield s


def make_chunk(url: str, index: int, chunk_id: str | None = None, **overrides) -> Chunk:
    defaults = {
        "chunk_id": chunk_id or f"{url}#{index}",
        "url": url,
        "source": "aws-en",
        "cloud": "aws",
        "chunk_index": index,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "heading_path": "Delta Lake",
        "section_path": "Tables > Table formats > Delta Lake",
        "doc_title": "What is Delta Lake?",
        "description": "Delta Lake is the optimized storage layer.",
        "text": f"chunk {index} text",
        "char_count": len(f"chunk {index} text"),
        "token_count": None,
        "token_count_model": None,
        "is_oversized": False,
        "last_updated": "2026-09-11",
        "page_sha256": "ab" * 32,
        "fetched_at": T1,
    }
    defaults.update(overrides)
    return Chunk(**defaults)


def make_page(url: str, n: int, **overrides) -> list[Chunk]:
    """`n` chunks for `url`, correctly linked into a chain."""
    ids = [f"{url}#{i}" for i in range(n)]
    chunks = []
    for i in range(n):
        chunks.append(
            make_chunk(
                url,
                i,
                chunk_id=ids[i],
                prev_chunk_id=ids[i - 1] if i > 0 else None,
                next_chunk_id=ids[i + 1] if i < n - 1 else None,
                **overrides,
            )
        )
    return chunks


# -- schema and persistence ----------------------------------------------


def test_opening_creates_the_database_and_its_parent_directory(db_path):
    assert not db_path.parent.exists()
    with ChunkStore(db_path) as s:
        assert s.counts() == ChunkCounts(0, 0, 0)
    assert db_path.exists()


def test_reopening_from_disk_returns_the_same_chunks(db_path):
    with ChunkStore(db_path) as s:
        s.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    with ChunkStore(db_path) as s:
        assert s.counts() == ChunkCounts(1, 2, 0)
        assert [c.chunk_id for c in s.get_chunks(URL_A)] == [
            f"{URL_A}#0",
            f"{URL_A}#1",
        ]


def test_schema_version_is_recorded_and_checked(db_path):
    with ChunkStore(db_path):
        pass
    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="schema version 99"):
        ChunkStore(db_path)


def test_is_oversized_column_rejects_values_outside_zero_or_one(store):
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "INSERT INTO chunks (chunk_id, url, source, cloud, chunk_index,"
            " heading_path, section_path, doc_title, text, char_count,"
            " is_oversized, page_sha256, parsed_at)"
            " VALUES ('x', 'u', 's', 'c', 0, 'h', 'p', 't', 'txt', 3, 2, 'sha', 'now')"
        )


# -- replace_page_chunks ---------------------------------------------------


def test_replacing_with_a_different_count_leaves_only_the_new_set(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 3), now=T1)
    assert store.counts() == ChunkCounts(1, 3, 0)

    store.replace_page_chunks(URL_A, make_page(URL_A, 1), now=T2)
    chunks = store.get_chunks(URL_A)
    assert store.counts() == ChunkCounts(1, 1, 0)
    assert [c.chunk_id for c in chunks] == [f"{URL_A}#0"]
    assert chunks[0].parsed_at == T2


def test_replacing_with_an_empty_list_clears_the_page(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    store.replace_page_chunks(URL_A, [], now=T2)
    assert store.get_chunks(URL_A) == []
    assert store.counts() == ChunkCounts(0, 0, 0)


def test_other_pages_are_untouched(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    store.replace_page_chunks(URL_B, make_page(URL_B, 1), now=T1)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1), now=T2)
    assert len(store.get_chunks(URL_A)) == 1
    assert len(store.get_chunks(URL_B)) == 1


def test_parsed_at_is_stamped_by_the_store_not_the_caller(store):
    chunks = make_page(URL_A, 1)
    object.__setattr__(chunks[0], "parsed_at", "whatever-the-caller-put-here")
    store.replace_page_chunks(URL_A, chunks, now=T1)
    assert store.get_chunks(URL_A)[0].parsed_at == T1


def test_prev_next_round_trip_and_are_null_at_the_boundaries(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 3), now=T1)
    first, middle, last = store.get_chunks(URL_A)

    assert first.prev_chunk_id is None
    assert first.next_chunk_id == middle.chunk_id
    assert middle.prev_chunk_id == first.chunk_id
    assert middle.next_chunk_id == last.chunk_id
    assert last.prev_chunk_id == middle.chunk_id
    assert last.next_chunk_id is None


def test_prev_next_always_point_at_a_chunk_of_the_same_url(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    for chunk in store.get_chunks(URL_A):
        for neighbor_id in (chunk.prev_chunk_id, chunk.next_chunk_id):
            if neighbor_id is not None:
                assert neighbor_id.startswith(URL_A)


def test_url_and_chunk_index_uniqueness_is_enforced_by_the_schema(store):
    # Bypass replace_page_chunks' own pre-check (tested separately below) to
    # confirm the schema itself, not just application code, rejects this.
    columns = (
        "chunk_id, url, source, cloud, chunk_index, heading_path,"
        " section_path, doc_title, text, char_count, page_sha256, parsed_at"
    )
    values = "(?, ?, ?, ?, ?, 'h', 'p', 't', 'txt', 3, 'sha', 'now')"
    store._conn.execute(
        f"INSERT INTO chunks ({columns}) VALUES {values}",
        ("a", URL_A, "aws-en", "aws", 0),
    )
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            f"INSERT INTO chunks ({columns}) VALUES {values}",
            ("b", URL_A, "aws-en", "aws", 0),  # same url, same chunk_index
        )


def test_a_chunk_from_a_different_url_is_rejected(store):
    mismatched = make_chunk(URL_B, 0)
    with pytest.raises(ValueError, match="expected"):
        store.replace_page_chunks(URL_A, [mismatched])


def test_duplicate_chunk_ids_are_rejected(store):
    a = make_chunk(URL_A, 0, chunk_id="dup")
    b = make_chunk(URL_A, 1, chunk_id="dup")
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        store.replace_page_chunks(URL_A, [a, b])


def test_duplicate_chunk_index_across_different_ids_is_rejected(store):
    a = make_chunk(URL_A, 0, chunk_id="a")
    b = make_chunk(URL_A, 0, chunk_id="b")
    with pytest.raises(ValueError, match="duplicate chunk_index"):
        store.replace_page_chunks(URL_A, [a, b])


def test_a_dangling_neighbor_reference_is_rejected(store):
    chunk = make_chunk(URL_A, 0, next_chunk_id="does-not-exist")
    with pytest.raises(ValueError, match="not in this batch"):
        store.replace_page_chunks(URL_A, [chunk])


def test_nothing_is_written_when_validation_fails(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    bad = [make_chunk(URL_A, 0, chunk_id="a"), make_chunk(URL_A, 0, chunk_id="b")]
    with pytest.raises(ValueError):
        store.replace_page_chunks(URL_A, bad)
    # The original two chunks are still there -- the failed replace didn't
    # even get as far as deleting them.
    assert len(store.get_chunks(URL_A)) == 2


# -- counts ----------------------------------------------------------------


def test_counts_across_sources(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    store.replace_page_chunks(
        URL_B, make_page(URL_B, 1, source="api", cloud="aws"), now=T1
    )
    assert store.counts() == ChunkCounts(pages=2, chunks=3, oversized=0)


def test_counts_by_source(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    store.replace_page_chunks(URL_B, make_page(URL_B, 1, source="api"), now=T1)
    by_source = store.counts_by_source()
    assert by_source["aws-en"] == ChunkCounts(pages=1, chunks=2, oversized=0)
    assert by_source["api"] == ChunkCounts(pages=1, chunks=1, oversized=0)


def test_oversized_chunks_are_counted(store):
    chunks = make_page(URL_A, 2)
    chunks[1] = make_chunk(
        URL_A,
        1,
        chunk_id=chunks[1].chunk_id,
        prev_chunk_id=chunks[0].chunk_id,
        is_oversized=True,
    )
    store.replace_page_chunks(URL_A, chunks, now=T1)
    assert store.counts() == ChunkCounts(pages=1, chunks=2, oversized=1)


# -- all_summaries -----------------------------------------------------


def test_all_summaries_is_empty_for_an_empty_store(store):
    assert store.all_summaries() == []


def test_all_summaries_covers_every_chunk_across_pages(store):
    chunks_a = make_page(URL_A, 2, source="aws-en")
    chunks_b = make_page(URL_B, 1, source="api")
    store.replace_page_chunks(URL_A, chunks_a, now=T1)
    store.replace_page_chunks(URL_B, chunks_b, now=T1)

    summaries = store.all_summaries()
    assert len(summaries) == 3
    assert set(summaries) == {
        ChunkSummary(c.url, c.source, c.char_count, c.is_oversized)
        for c in (*chunks_a, *chunks_b)
    }


def test_all_summaries_reflects_is_oversized_as_a_real_bool(store):
    chunk = make_chunk(URL_A, 0, is_oversized=True)
    store.replace_page_chunks(URL_A, [chunk], now=T1)
    summary = store.all_summaries()[0]
    assert summary.is_oversized is True
    assert isinstance(summary.is_oversized, bool)


def test_all_summaries_after_a_replace_reflects_only_the_current_chunks(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 3), now=T1)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1), now=T2)
    assert len(store.all_summaries()) == 1


# -- chunk_ids -----------------------------------------------------------


def test_chunk_ids_is_empty_for_an_empty_store(store):
    assert store.chunk_ids() == set()


def test_chunk_ids_covers_every_chunk_iter_chunks_walks(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    store.replace_page_chunks(URL_B, make_page(URL_B, 1), now=T1)
    ids = store.chunk_ids()
    assert ids == {f"{URL_A}#0", f"{URL_A}#1", f"{URL_B}#0"}
    assert ids == {chunk.chunk_id for chunk in store.iter_chunks()}


def test_chunk_ids_after_a_replace_reflects_only_the_current_chunks(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 3), now=T1)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1), now=T2)
    assert store.chunk_ids() == {f"{URL_A}#0"}


# -- chunks_needing_token_count / set_token_counts ---------------------------

MODEL_A = "text-embedding-3-small"
MODEL_B = "text-embedding-3-large"


def test_chunks_needing_token_count_is_empty_for_an_empty_store(store):
    assert store.chunks_needing_token_count(MODEL_A) == []


def test_chunks_needing_token_count_lists_every_never_tokenized_row(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    needing = store.chunks_needing_token_count(MODEL_A)
    assert sorted(needing) == sorted(
        ChunkText(c.chunk_id, c.text) for c in store.get_chunks(URL_A)
    )


def test_chunks_needing_token_count_excludes_rows_already_tagged_with_that_model(store):
    chunks = make_page(URL_A, 2)
    chunks[0] = replace(chunks[0], token_count=5, token_count_model=MODEL_A)
    store.replace_page_chunks(URL_A, chunks, now=T1)
    needing = store.chunks_needing_token_count(MODEL_A)
    assert [n.chunk_id for n in needing] == [chunks[1].chunk_id]


def test_chunks_needing_token_count_includes_rows_tagged_with_a_different_model(store):
    chunks = make_page(URL_A, 2)
    chunks[0] = replace(chunks[0], token_count=5, token_count_model=MODEL_A)
    store.replace_page_chunks(URL_A, chunks, now=T1)
    # Asking for MODEL_B: the MODEL_A-tagged row needs (re-)tokenizing too --
    # this is what lets you test/compare a different tokenizer.
    needing = store.chunks_needing_token_count(MODEL_B)
    assert {n.chunk_id for n in needing} == {c.chunk_id for c in chunks}


def test_set_token_counts_writes_the_given_values_and_model(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    ids = [c.chunk_id for c in store.get_chunks(URL_A)]
    updated = store.set_token_counts({ids[0]: 7, ids[1]: 42}, MODEL_A)
    assert updated == 2
    chunks = {c.chunk_id: c for c in store.get_chunks(URL_A)}
    assert chunks[ids[0]].token_count == 7
    assert chunks[ids[0]].token_count_model == MODEL_A
    assert chunks[ids[1]].token_count == 42
    assert chunks[ids[1]].token_count_model == MODEL_A


def test_set_token_counts_clears_the_needing_list_for_that_model_only(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 2), now=T1)
    ids = [c.chunk_id for c in store.get_chunks(URL_A)]
    store.set_token_counts({i: 1 for i in ids}, MODEL_A)
    assert store.chunks_needing_token_count(MODEL_A) == []
    # But it's now "needing" again under a different model -- that's the
    # point: testing a different tokenizer re-tokenizes rather than reusing
    # a stale count.
    assert len(store.chunks_needing_token_count(MODEL_B)) == 2


def test_set_token_counts_skips_chunk_ids_no_longer_in_the_store(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 1), now=T1)
    updated = store.set_token_counts({"does-not-exist": 1}, MODEL_A)
    assert updated == 0


def test_set_token_counts_with_an_empty_mapping_does_nothing(store):
    store.replace_page_chunks(URL_A, make_page(URL_A, 1), now=T1)
    assert store.set_token_counts({}, MODEL_A) == 0
    assert len(store.chunks_needing_token_count(MODEL_A)) == 1


# -- version 1 -> 2 migration (adds token_count_model) -----------------------


def test_opening_a_version_1_store_adds_the_column_and_bumps_the_version(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE chunks ("
        " chunk_id TEXT PRIMARY KEY, url TEXT NOT NULL, source TEXT NOT NULL,"
        " cloud TEXT NOT NULL, chunk_index INTEGER NOT NULL,"
        " prev_chunk_id TEXT, next_chunk_id TEXT, heading_path TEXT NOT NULL,"
        " section_path TEXT NOT NULL, doc_title TEXT NOT NULL,"
        " description TEXT, text TEXT NOT NULL, char_count INTEGER NOT NULL,"
        " token_count INTEGER,"
        " is_oversized INTEGER NOT NULL DEFAULT 0, last_updated TEXT,"
        " page_sha256 TEXT NOT NULL, fetched_at TEXT, parsed_at TEXT NOT NULL,"
        " UNIQUE (url, chunk_index))"
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, url, source, cloud, chunk_index,"
        " heading_path, section_path, doc_title, text, char_count,"
        " token_count, page_sha256, parsed_at)"
        " VALUES ('x', 'u', 's', 'c', 0, 'h', 'p', 'd', 'txt', 3, 99, 'sha', 'now')"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    with ChunkStore(db_path) as store:
        chunk = store.get_chunks("u")[0]
        # The pre-existing token_count survives the migration untouched...
        assert chunk.token_count == 99
        # ...but with no tracked model, it's treated as needing (re-)
        # tokenizing under whatever model is asked for.
        assert chunk.token_count_model is None
        assert store.chunks_needing_token_count(MODEL_A) == [ChunkText("x", "txt")]

    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def test_reopening_a_migrated_store_is_a_no_op(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE chunks ("
        " chunk_id TEXT PRIMARY KEY, url TEXT NOT NULL, source TEXT NOT NULL,"
        " cloud TEXT NOT NULL, chunk_index INTEGER NOT NULL,"
        " prev_chunk_id TEXT, next_chunk_id TEXT, heading_path TEXT NOT NULL,"
        " section_path TEXT NOT NULL, doc_title TEXT NOT NULL,"
        " description TEXT, text TEXT NOT NULL, char_count INTEGER NOT NULL,"
        " token_count INTEGER,"
        " is_oversized INTEGER NOT NULL DEFAULT 0, last_updated TEXT,"
        " page_sha256 TEXT NOT NULL, fetched_at TEXT, parsed_at TEXT NOT NULL,"
        " UNIQUE (url, chunk_index))"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    with ChunkStore(db_path):
        pass
    with ChunkStore(db_path) as store:  # second open: already at SCHEMA_VERSION
        assert store.counts() == ChunkCounts(0, 0, 0)
