import pytest

from databricks_docs_vectordb.parsing.chunker import chunk_page
from databricks_docs_vectordb.parsing.config import Settings
from databricks_docs_vectordb.parsing.html_walker import (
    CodeBlock,
    Heading,
    Paragraph,
    Table,
)
from databricks_docs_vectordb.parsing.metadata import PageMetadata

URL = "https://docs.databricks.com/aws/en/example/"


def meta(**overrides) -> PageMetadata:
    defaults = {
        "url": URL,
        "source": "aws-en",
        "cloud": "aws",
        "doc_title": "Example Page",
        "description": "An example page.",
        "section_path": "Examples > Example Page",
        "last_updated": "2026-09-11T00:00:00.000Z",
        "page_sha256": "ab" * 32,
        "fetched_at": "2026-09-11T00:00:00Z",
    }
    defaults.update(overrides)
    return PageMetadata(**defaults)


def settings(**overrides) -> Settings:
    defaults = {"target_chunk_size": 100, "overlap_size": 20, "overlap_mode": "chars"}
    defaults.update(overrides)
    return Settings(**defaults)


def h(level: int, text: str, id_: str | None = None) -> Heading:
    return Heading(level=level, id=id_ or text.lower().replace(" ", "-"), text=text)


def p(n_chars: int, marker: str = "x") -> Paragraph:
    """A paragraph of roughly `n_chars` characters (never renders empty)."""
    return Paragraph((marker * max(n_chars, 1))[:n_chars])


def assert_linked_sequence(chunks) -> None:
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.prev_chunk_id == (ids[i - 1] if i > 0 else None)
        assert c.next_chunk_id == (ids[i + 1] if i < len(ids) - 1 else None)


# -- empty input -----------------------------------------------------------


def test_no_nodes_produces_no_chunks():
    assert chunk_page([], meta(), settings()) == []


# -- one chunk per h2 section when everything fits --------------------------


def test_sections_under_target_size_produce_one_chunk_per_h2_section():
    nodes = [
        h(2, "First"),
        p(30),
        h(2, "Second"),
        p(30),
        h(2, "Third"),
        p(30),
    ]
    chunks = chunk_page(nodes, meta(), settings())
    assert len(chunks) == 3
    assert [c.heading_path for c in chunks] == ["First", "Second", "Third"]
    assert all(not c.is_oversized for c in chunks)
    assert_linked_sequence(chunks)


def test_preamble_before_the_first_h2_becomes_its_own_chunk():
    nodes = [p(20), h(2, "First"), p(20)]
    chunks = chunk_page(nodes, meta(), settings())
    assert len(chunks) == 2
    assert chunks[0].heading_path == ""
    assert chunks[1].heading_path == "First"


def test_a_page_with_no_h2_at_all_is_one_chunk():
    nodes = [p(20), p(20)]
    chunks = chunk_page(nodes, meta(), settings())
    assert len(chunks) == 1
    assert chunks[0].heading_path == ""
    assert not chunks[0].is_oversized


# -- shared metadata fields --------------------------------------------------


def test_every_chunk_carries_the_pages_metadata():
    nodes = [h(2, "First"), p(20), h(2, "Second"), p(20)]
    chunks = chunk_page(nodes, meta(), settings())
    for c in chunks:
        assert c.url == URL
        assert c.source == "aws-en"
        assert c.cloud == "aws"
        assert c.doc_title == "Example Page"
        assert c.description == "An example page."
        assert c.section_path == "Examples > Example Page"
        assert c.last_updated == "2026-09-11T00:00:00.000Z"
        assert c.page_sha256 == "ab" * 32
        assert c.fetched_at == "2026-09-11T00:00:00Z"


def test_chunk_id_is_derived_from_url_and_index():
    nodes = [h(2, "First"), p(20), h(2, "Second"), p(20)]
    chunks = chunk_page(nodes, meta(), settings())
    assert chunks[0].chunk_id == f"{URL}#0"
    assert chunks[1].chunk_id == f"{URL}#1"


# -- an oversized h2 section splits at its h3s -------------------------------


def test_an_oversized_h2_section_splits_at_its_h3s():
    nodes = [
        h(2, "Big Section"),
        h(3, "Sub A"),
        p(60),
        h(3, "Sub B"),
        p(60),
    ]
    chunks = chunk_page(nodes, meta(), settings())
    # The whole section (~150+ chars) exceeds the 100-char target, but each
    # h3 subsection (~70 chars) fits on its own.
    assert len(chunks) == 2
    assert chunks[0].heading_path == "Big Section > Sub A"
    assert chunks[1].heading_path == "Big Section > Sub B"
    assert all(not c.is_oversized for c in chunks)
    assert_linked_sequence(chunks)


def test_intro_content_before_the_first_h3_gets_its_own_chunk():
    nodes = [
        h(2, "Big Section"),
        p(70),  # real intro content, not just the heading
        h(3, "Sub A"),
        p(70),
    ]
    chunks = chunk_page(nodes, meta(), settings())
    assert len(chunks) == 2
    assert chunks[0].heading_path == "Big Section"
    assert chunks[1].heading_path == "Big Section > Sub A"


def test_a_heading_only_leading_h3_group_is_folded_into_the_first_h3_section():
    # No content directly under the <h2> before the first <h3> -- the <h2>
    # heading alone would otherwise become a near-empty chunk.
    nodes = [
        h(2, "Big Section"),
        h(3, "Sub A"),
        p(70),
        h(3, "Sub B"),
        p(70),
    ]
    chunks = chunk_page(nodes, meta(), settings())
    assert len(chunks) == 2
    assert chunks[0].heading_path == "Big Section > Sub A"
    assert "## Big Section" in chunks[0].text
    assert "### Sub A" in chunks[0].text


# -- an oversized table or code block becomes its own flagged chunk ---------


def test_an_oversized_code_block_is_never_split_and_is_flagged():
    big_code = CodeBlock(language="python", code="x = 1\n" * 40)  # well over 100 chars
    nodes = [h(2, "Section"), p(20), big_code, p(20)]
    chunks = chunk_page(nodes, meta(), settings())
    oversized = [c for c in chunks if c.is_oversized]
    assert len(oversized) == 1
    assert big_code.code in oversized[0].text
    # The surrounding normal-sized content is packed into separate chunks,
    # not swallowed into the oversized one.
    assert not any(c.is_oversized for c in chunks if c is not oversized[0])
    assert_linked_sequence(chunks)


def test_an_oversized_table_is_never_split_and_is_flagged():
    big_table = Table(
        headers=("A", "B"),
        rows=tuple(("x" * 20, "y" * 20) for _ in range(10)),
    )
    nodes = [h(2, "Section"), p(20), big_table, p(20)]
    chunks = chunk_page(nodes, meta(), settings())
    oversized = [c for c in chunks if c.is_oversized]
    assert len(oversized) == 1
    assert "| A | B |" in oversized[0].text
    assert_linked_sequence(chunks)


def test_an_oversized_node_shares_its_sections_heading_path():
    big_code = CodeBlock(language="python", code="x = 1\n" * 40)
    nodes = [h(2, "Section"), h(3, "Sub"), p(20), big_code]
    chunks = chunk_page(nodes, meta(), settings())
    assert all(c.heading_path == "Section > Sub" for c in chunks)


def test_all_chunks_from_packing_share_the_same_heading_path_even_unsplit_by_h3():
    # No <h3>s at all, but too much content to fit in one chunk: packing
    # kicks in directly under the <h2>, and every resulting chunk still
    # belongs to that one logical section.
    nodes = [h(2, "Section"), p(70), p(70), p(70)]
    chunks = chunk_page(nodes, meta(), settings())
    assert len(chunks) > 1
    assert all(c.heading_path == "Section" for c in chunks)


# -- prev/next linkage across the whole page --------------------------------


def test_prev_next_form_a_correct_sequence_null_only_at_the_ends():
    nodes = [
        h(2, "First"),
        p(70),
        p(70),
        p(70),
        h(2, "Second"),
        p(20),
    ]
    chunks = chunk_page(nodes, meta(), settings())
    assert len(chunks) > 2  # confirms this actually exercises multiple chunks
    assert_linked_sequence(chunks)
    assert chunks[0].prev_chunk_id is None
    assert chunks[-1].next_chunk_id is None
    for c in chunks[1:-1]:
        assert c.prev_chunk_id is not None
        assert c.next_chunk_id is not None


# -- overlap -----------------------------------------------------------


def test_overlap_content_appears_in_both_chunks_on_either_side_of_a_split():
    nodes = [h(2, "First"), p(80), h(2, "Second"), p(80)]
    chunks = chunk_page(nodes, meta(), settings(overlap_size=20))
    first_own_text = "## First\n\n" + "x" * 80
    assert chunks[0].text == first_own_text
    tail = first_own_text[-20:]
    assert chunks[1].text.startswith(tail)
    assert "## Second" in chunks[1].text


def test_the_first_chunk_has_no_overlap_prepended():
    nodes = [h(2, "First"), p(20)]
    chunks = chunk_page(nodes, meta(), settings())
    assert chunks[0].text == "## First\n\n" + "x" * 20


def test_overlap_size_zero_means_no_overlap_is_added():
    nodes = [h(2, "First"), p(80), h(2, "Second"), p(80)]
    chunks = chunk_page(nodes, meta(), settings(overlap_size=0))
    assert chunks[1].text == "## Second\n\n" + "x" * 80


def test_heading_overlap_mode_restates_the_previous_chunks_heading_path():
    nodes = [h(2, "First"), p(80), h(2, "Second"), p(80)]
    chunks = chunk_page(nodes, meta(), settings(overlap_mode="heading"))
    assert chunks[1].text.startswith('*(continued from "First")*\n\n## Second')


def test_heading_overlap_mode_adds_nothing_when_the_previous_chunk_has_no_heading():
    # The preamble chunk has heading_path "" -- nothing to restate.
    nodes = [p(80), h(2, "First"), p(80)]
    chunks = chunk_page(nodes, meta(), settings(overlap_mode="heading"))
    assert chunks[1].text == "## First\n\n" + "x" * 80


def test_char_count_reflects_the_final_text_including_overlap():
    nodes = [h(2, "First"), p(80), h(2, "Second"), p(80)]
    chunks = chunk_page(nodes, meta(), settings(overlap_size=20))
    assert chunks[1].char_count == len(chunks[1].text)
    assert chunks[1].char_count > len("## Second\n\n" + "x" * 80)


# -- token_count is left for a later phase -----------------------------


def test_token_count_is_none():
    nodes = [h(2, "First"), p(20)]
    chunks = chunk_page(nodes, meta(), settings())
    assert chunks[0].token_count is None


# -- real fixtures: no crashes, sane invariants ------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ("delta", "index.html"),
        ("oltp", "update-to-autoscaling-terraform.html"),
        ("api", "jobs", "v2", "run.html"),
        ("error-messages", "error-classes.html"),
        ("machine-learning", "sgc-yolo11n-detect-coco128.html"),
    ],
)
def test_real_fixtures_produce_a_valid_linked_chunk_sequence(path):
    from pathlib import Path

    from databricks_docs_vectordb.parsing.html_walker import walk
    from databricks_docs_vectordb.parsing.metadata import extract

    fixtures = Path(__file__).parent.parent / "fixtures" / "parsing"
    html = fixtures.joinpath(*path).read_text(encoding="utf-8")
    source = "api" if path[0] == "api" else "aws-en"
    page_meta = extract(
        html,
        url=f"https://docs.databricks.com/{'/'.join(path)}",
        source=source,
        sha256="ab" * 32,
        fetched_at="2026-09-11T00:00:00Z",
    )
    settings = Settings()
    chunks = chunk_page(walk(html), page_meta, settings)
    assert chunks
    assert_linked_sequence(chunks)
    assert all(c.text for c in chunks)
    # Every non-oversized chunk respects the configured target, up to the
    # overlap prepended from its predecessor (is_oversized reflects the
    # section's own content, computed before overlap is added).
    ceiling = settings.target_chunk_size + settings.overlap_size + len("\n\n")
    assert all(c.char_count <= ceiling or c.is_oversized for c in chunks)
