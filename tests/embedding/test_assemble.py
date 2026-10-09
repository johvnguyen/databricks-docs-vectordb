from itertools import pairwise

import pytest

from databricks_docs_vectordb.embedding.assemble import (
    EmbeddingInput,
    assemble_inputs,
    build_context,
)
from databricks_docs_vectordb.embedding.tokens import count_tokens, get_encoding
from databricks_docs_vectordb.parsing.store import Chunk

MODEL = "text-embedding-3-small"
URL_A = "https://docs.databricks.com/aws/en/delta/"


def make_chunk(text: str = "hello world", **overrides) -> Chunk:
    defaults = {
        "chunk_id": f"{URL_A}#0",
        "url": URL_A,
        "source": "aws-en",
        "cloud": "aws",
        "chunk_index": 0,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "heading_path": "Delta Lake",
        "section_path": "Tables > Table formats > Delta Lake",
        "doc_title": "What is Delta Lake?",
        "description": "Delta Lake is the optimized storage layer.",
        "text": text,
        "char_count": len(text),
        "token_count": None,
        "token_count_model": None,
        "is_oversized": False,
        "last_updated": "2026-09-11",
        "page_sha256": "ab" * 32,
        "fetched_at": "2026-09-11T00:00:00Z",
    }
    defaults.update(overrides)
    return Chunk(**defaults)


def assemble(chunk: Chunk, token_limit: int = 8191, overlap: int = 200):
    return assemble_inputs(
        chunk, model=MODEL, token_limit=token_limit, overlap_tokens=overlap
    )


def numbered_text(n: int) -> str:
    # Distinct words, so each slice's position in the original is unambiguous.
    return " ".join(f"w{i}" for i in range(n))


def test_normal_chunk_is_context_blank_line_then_text():
    chunk = make_chunk("Delta Lake is great.")
    inputs = assemble(chunk)
    expected = (
        "What is Delta Lake?\n"
        "Delta Lake\n"
        "Tables > Table formats > Delta Lake\n"
        "\n"
        "Delta Lake is great."
    )
    assert inputs == [EmbeddingInput(expected, count_tokens(expected, MODEL))]


def test_chunk_exactly_at_limit_produces_one_input():
    chunk = make_chunk(numbered_text(300))
    whole = count_tokens(build_context(chunk) + chunk.text, MODEL)
    inputs = assemble(chunk, token_limit=whole, overlap=10)
    assert len(inputs) == 1
    assert inputs[0].token_count == whole


def test_chunk_one_token_over_limit_splits():
    chunk = make_chunk(numbered_text(300))
    whole = count_tokens(build_context(chunk) + chunk.text, MODEL)
    assert len(assemble(chunk, token_limit=whole - 1, overlap=10)) == 2


def test_near_empty_chunk_still_produces_one_nonempty_input():
    inputs = assemble(make_chunk("- "))
    assert len(inputs) == 1
    assert inputs[0].text.endswith("\n\n- ")
    assert inputs[0].token_count > 0


def test_oversized_chunk_slices_fit_limit_and_carry_context():
    chunk = make_chunk(numbered_text(5000))
    context = build_context(chunk)
    inputs = assemble(chunk, token_limit=500, overlap=50)
    assert len(inputs) > 1
    for piece in inputs:
        assert piece.text.startswith(context)
        assert piece.token_count == count_tokens(piece.text, MODEL)
        assert piece.token_count <= 500


def test_oversized_slices_cover_the_whole_text_in_order():
    chunk = make_chunk(numbered_text(5000))
    context = build_context(chunk)
    bodies = [p.text[len(context) :] for p in assemble(chunk, 500, 50)]
    assert chunk.text.startswith(bodies[0])
    assert chunk.text.endswith(bodies[-1])
    # Every slice is a verbatim substring, and each begins before the
    # previous one ends -- so nothing between them is dropped.
    position = 0
    for body in bodies:
        start = chunk.text.index(body)
        assert start <= position
        position = start + len(body)
    assert position == len(chunk.text)


def test_overlap_tokens_repeat_at_each_internal_boundary():
    chunk = make_chunk(numbered_text(5000))
    context = build_context(chunk)
    overlap = 50
    encoding = get_encoding(MODEL)
    bodies = [p.text[len(context) :] for p in assemble(chunk, 500, overlap)]
    for prev, nxt in pairwise(bodies):
        tail = encoding.decode(encoding.encode(prev)[-overlap:])
        assert nxt.startswith(tail)
        assert len(encoding.encode(tail)) == overlap


def test_zero_overlap_slices_partition_the_text():
    chunk = make_chunk(numbered_text(2000))
    context = build_context(chunk)
    bodies = [p.text[len(context) :] for p in assemble(chunk, 300, 0)]
    assert "".join(bodies) == chunk.text


def test_multibyte_text_is_never_cut_mid_character():
    # CJK characters span multiple tokens each under cl100k_base, so a
    # naive token-boundary cut would leave U+FFFD in a slice.
    chunk = make_chunk("日本語のテキスト。" * 800)
    inputs = assemble(chunk, token_limit=400, overlap=30)
    assert len(inputs) > 1
    for piece in inputs:
        assert "�" not in piece.text
        assert piece.token_count <= 400


def test_context_leaving_no_room_for_text_raises():
    chunk = make_chunk(numbered_text(1000), doc_title=numbered_text(200))
    with pytest.raises(ValueError, match="no room"):
        assemble(chunk, token_limit=300, overlap=200)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        assemble(make_chunk(), overlap=-1)
