from pathlib import Path

import pytest

from databricks_docs_vectordb.cli import main
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore

URL_A = "https://docs.databricks.com/aws/en/delta/"


@pytest.fixture
def env(monkeypatch, tmp_path):
    """No real .env, data in a temp dir. Embedding's backfill-tokens never
    needs a contact address or a provider API key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABRICKS_DOCS_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def make_chunk(index: int, **overrides) -> Chunk:
    defaults = {
        "chunk_id": f"{URL_A}#{index}",
        "url": URL_A,
        "source": "aws-en",
        "cloud": "aws",
        "chunk_index": index,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "heading_path": "Delta Lake",
        "section_path": "Tables > Table formats > Delta Lake",
        "doc_title": "What is Delta Lake?",
        "description": "Delta Lake is the optimized storage layer.",
        "text": "hello world",
        "char_count": len("hello world"),
        "token_count": None,
        "is_oversized": False,
        "last_updated": "2026-09-11",
        "page_sha256": "ab" * 32,
        "fetched_at": "2026-09-11T00:00:00Z",
    }
    defaults.update(overrides)
    return Chunk(**defaults)


# -- embed backfill-tokens ----------------------------------------------


def test_backfill_tokens_without_a_chunk_store_is_a_clear_error(env, capsys):
    assert main(["embed", "backfill-tokens"]) == 1
    assert "parse run" in capsys.readouterr().err


def test_backfill_tokens_fills_in_every_null_row(env, capsys):
    with ChunkStore(env / "parsed.sqlite") as store:
        store.replace_page_chunks(URL_A, [make_chunk(0, text="hello world")])

    assert main(["embed", "backfill-tokens"]) == 0
    out = capsys.readouterr().out
    assert "1 chunk(s) given a real token_count" in out
    assert "text-embedding-3-small" in out

    with ChunkStore(env / "parsed.sqlite") as store:
        assert store.get_chunks(URL_A)[0].token_count == 2  # "hello world"


def test_backfill_tokens_leaves_text_and_boundaries_untouched(env, capsys):
    with ChunkStore(env / "parsed.sqlite") as store:
        store.replace_page_chunks(URL_A, [make_chunk(0, text="hello world")])

    main(["embed", "backfill-tokens"])

    with ChunkStore(env / "parsed.sqlite") as store:
        chunk = store.get_chunks(URL_A)[0]
        assert chunk.text == "hello world"
        assert chunk.char_count == len("hello world")


def test_running_backfill_tokens_twice_is_a_no_op_the_second_time(env, capsys):
    with ChunkStore(env / "parsed.sqlite") as store:
        store.replace_page_chunks(URL_A, [make_chunk(0, text="hello world")])

    main(["embed", "backfill-tokens"])
    capsys.readouterr()

    assert main(["embed", "backfill-tokens"]) == 0
    assert "nothing to backfill" in capsys.readouterr().out


def test_backfill_tokens_on_an_empty_store_says_so(env, capsys):
    with ChunkStore(env / "parsed.sqlite"):
        pass  # creates the (empty) store, nothing more
    assert main(["embed", "backfill-tokens"]) == 0
    assert "nothing to backfill" in capsys.readouterr().out
