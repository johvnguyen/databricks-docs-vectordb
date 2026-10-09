from pathlib import Path

import pytest
import tiktoken

from databricks_docs_vectordb.embedding.tokens import (
    backfill_token_counts,
    count_tokens,
    get_encoding,
)
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore

OPENAI_SMALL = "text-embedding-3-small"
OPENAI_LARGE = "text-embedding-3-large"
URL_A = "https://docs.databricks.com/aws/en/delta/"


@pytest.fixture
def store(tmp_path: Path):
    with ChunkStore(tmp_path / "parsed.sqlite") as s:
        yield s


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
        "token_count_model": None,
        "is_oversized": False,
        "last_updated": "2026-09-11",
        "page_sha256": "ab" * 32,
        "fetched_at": "2026-09-11T00:00:00Z",
    }
    defaults.update(overrides)
    return Chunk(**defaults)


def test_count_tokens_matches_known_tiktoken_counts():
    # cl100k_base, confirmed against the real tiktoken package.
    assert count_tokens("hello world", OPENAI_SMALL) == 2
    assert count_tokens("Delta Lake", OPENAI_SMALL) == 2
    assert count_tokens("tokenization", OPENAI_SMALL) == 2


def test_empty_string_has_zero_tokens():
    assert count_tokens("", OPENAI_SMALL) == 0


def test_get_encoding_resolves_openai_embedding_models_to_cl100k_base():
    assert get_encoding(OPENAI_SMALL).name == "cl100k_base"
    assert get_encoding(OPENAI_LARGE).name == "cl100k_base"


def test_unknown_model_falls_back_to_cl100k_base_instead_of_raising():
    # A non-OpenAI model routed through LiteLLM -- tiktoken doesn't know it
    # by name, but count_tokens must still return an approximation, not
    # raise (Resolved question 1's "configurable to different models").
    assert get_encoding("cohere/embed-english-v3.0").name == "cl100k_base"
    assert count_tokens("hello world", "cohere/embed-english-v3.0") == 2


def test_get_encoding_is_cached():
    assert get_encoding(OPENAI_SMALL) is get_encoding(OPENAI_SMALL)


def test_count_tokens_agrees_with_encoding_the_text_directly():
    text = "The quick brown fox jumps over the lazy dog."
    encoding = tiktoken.encoding_for_model(OPENAI_SMALL)
    assert count_tokens(text, OPENAI_SMALL) == len(encoding.encode(text))


# -- backfill_token_counts ---------------------------------------------------


def test_backfill_writes_real_counts_for_every_null_row(store):
    chunk = make_chunk(0, text="hello world")
    store.replace_page_chunks(URL_A, [chunk])
    updated = backfill_token_counts(store, OPENAI_SMALL)
    assert updated == 1
    assert store.get_chunks(URL_A)[0].token_count == 2


def test_backfill_on_an_empty_store_does_nothing(store):
    assert backfill_token_counts(store, OPENAI_SMALL) == 0


def test_backfill_leaves_rows_already_tagged_with_that_model_alone(store):
    chunk = make_chunk(
        0, text="hello world", token_count=999, token_count_model=OPENAI_SMALL
    )
    store.replace_page_chunks(URL_A, [chunk])
    updated = backfill_token_counts(store, OPENAI_SMALL)
    assert updated == 0
    assert store.get_chunks(URL_A)[0].token_count == 999


def test_backfill_re_tokenizes_under_a_different_model(store):
    # Testing a different tokenizer: a row already tagged with one model
    # is re-tokenized, not skipped, when asked for a different one.
    chunk = make_chunk(
        0, text="hello world", token_count=999, token_count_model=OPENAI_SMALL
    )
    store.replace_page_chunks(URL_A, [chunk])
    updated = backfill_token_counts(store, OPENAI_LARGE)
    assert updated == 1
    result = store.get_chunks(URL_A)[0]
    assert result.token_count == 2  # real count, not the old placeholder
    assert result.token_count_model == OPENAI_LARGE


def test_a_second_backfill_run_touches_nothing(store):
    chunk = make_chunk(0, text="hello world")
    store.replace_page_chunks(URL_A, [chunk])
    assert backfill_token_counts(store, OPENAI_SMALL) == 1
    assert backfill_token_counts(store, OPENAI_SMALL) == 0


def test_backfill_only_touches_the_pages_a_reparse_actually_changed(store):
    a = make_chunk(0, text="hello world")
    b = make_chunk(0, chunk_id=f"{URL_A}other#0", url=f"{URL_A}other", text="a b c")
    store.replace_page_chunks(URL_A, [a])
    store.replace_page_chunks(f"{URL_A}other", [b])
    assert backfill_token_counts(store, OPENAI_SMALL) == 2

    # Re-parsing just one page resets its token_count to NULL again
    # (chunker.py still doesn't set it); the other page's is untouched.
    store.replace_page_chunks(URL_A, [make_chunk(0, text="hello world, again")])
    assert [c.chunk_id for c in store.chunks_needing_token_count(OPENAI_SMALL)] == [
        a.chunk_id
    ]
    assert backfill_token_counts(store, OPENAI_SMALL) == 1
