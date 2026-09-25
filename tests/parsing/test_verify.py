from pathlib import Path

import pytest

from databricks_docs_vectordb.parsing.config import Settings
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore
from databricks_docs_vectordb.parsing.verify import format_report, verify_store
from databricks_docs_vectordb.scraping.config import SOURCES
from databricks_docs_vectordb.scraping.manifest import Manifest

URL_A = "https://docs.databricks.com/aws/en/sql/a/"
URL_B = "https://docs.databricks.com/aws/en/pyspark/b/"


@pytest.fixture
def manifest(tmp_path: Path):
    with Manifest(tmp_path / "manifest.sqlite") as m:
        yield m


@pytest.fixture
def store(tmp_path: Path):
    with ChunkStore(tmp_path / "parsed.sqlite") as s:
        yield s


@pytest.fixture
def settings() -> Settings:
    return Settings(target_chunk_size=100, overlap_size=20)


def add_ok(manifest: Manifest, url: str, *, sha256: str = "ab" * 32) -> None:
    source = "api" if url.startswith("https://docs.databricks.com/api/") else "aws-en"
    manifest.add_pending(source, [url])
    manifest.record_ok(url, http_status=200, sha256=sha256, size_bytes=100)


def make_chunk(url: str, index: int, chunk_id: str | None = None, **overrides) -> Chunk:
    defaults = {
        "chunk_id": chunk_id or f"{url}#{index}",
        "url": url,
        "source": "aws-en",
        "cloud": "aws",
        "chunk_index": index,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "heading_path": "Section",
        "section_path": "Section",
        "doc_title": "Title",
        "description": "A page.",
        "text": f"chunk {index} text",
        "char_count": len(f"chunk {index} text"),
        "token_count": None,
        "is_oversized": False,
        "last_updated": "2026-09-11",
        "page_sha256": "ab" * 32,
        "fetched_at": "2026-09-11T00:00:00Z",
    }
    defaults.update(overrides)
    defaults["char_count"] = len(defaults["text"])  # keep it in sync with `text`
    return Chunk(**defaults)


def make_page(url: str, n: int, **overrides) -> list[Chunk]:
    ids = [f"{url}#{i}" for i in range(n)]
    return [
        make_chunk(
            url,
            i,
            chunk_id=ids[i],
            prev_chunk_id=ids[i - 1] if i > 0 else None,
            next_chunk_id=ids[i + 1] if i < n - 1 else None,
            **overrides,
        )
        for i in range(n)
    ]


def kinds(report) -> list[str]:
    return sorted(f.kind for f in report.violations)


# -- a clean store ---------------------------------------------------------


def test_a_clean_store_has_no_violations(manifest, store, settings):
    add_ok(manifest, URL_A)
    store.replace_page_chunks(URL_A, make_page(URL_A, 2))
    report = verify_store(manifest, store, settings, SOURCES)
    assert report.ok
    assert report.violations == []
    assert report.checked == 1
    assert report.total_chunks == 2


def test_a_clean_store_formats_as_ok(manifest, store, settings):
    add_ok(manifest, URL_A)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1))
    report = verify_store(manifest, store, settings, SOURCES)
    assert format_report(report, settings).splitlines()[-1] == "OK"


def test_an_empty_manifest_and_store_is_clean(manifest, store, settings):
    report = verify_store(manifest, store, settings, SOURCES)
    assert report.ok
    assert report.checked == 0
    assert report.total_chunks == 0


# -- no-chunks / --complete ----------------------------------------------


def test_by_default_an_unparsed_ok_page_is_not_a_violation(manifest, store, settings):
    # A pilot run (Step 9) only parses a sample on purpose; verify must not
    # treat the rest of the corpus as broken.
    add_ok(manifest, URL_A)
    report = verify_store(manifest, store, settings, SOURCES)
    assert report.ok
    assert report.unparsed == 1


def test_with_complete_an_unparsed_ok_page_is_a_violation(manifest, store, settings):
    add_ok(manifest, URL_A)
    report = verify_store(manifest, store, settings, SOURCES, complete=True)
    assert kinds(report) == ["no-chunks"]
    assert not report.ok


def test_unparsed_pages_get_an_informational_note_when_not_complete(
    manifest, store, settings
):
    add_ok(manifest, URL_A)
    report = verify_store(manifest, store, settings, SOURCES)
    out = format_report(report, settings)
    assert "note: 1 ok page(s) have not been parsed yet" in out
    assert "OK" in out.splitlines()[-1]


def test_no_note_when_every_ok_page_has_chunks(manifest, store, settings):
    add_ok(manifest, URL_A)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1))
    report = verify_store(manifest, store, settings, SOURCES)
    assert "note:" not in format_report(report, settings)


# -- empty-chunk -------------------------------------------------------


def test_an_empty_chunk_text_is_a_violation(manifest, store, settings):
    add_ok(manifest, URL_A)
    chunks = make_page(URL_A, 1)
    chunks[0] = make_chunk(URL_A, 0, chunk_id=chunks[0].chunk_id, text="")
    store.replace_page_chunks(URL_A, chunks)
    report = verify_store(manifest, store, settings, SOURCES)
    assert kinds(report) == ["empty-chunk"]


def test_a_whitespace_only_chunk_text_is_also_a_violation(manifest, store, settings):
    add_ok(manifest, URL_A)
    chunks = make_page(URL_A, 1)
    chunks[0] = make_chunk(URL_A, 0, chunk_id=chunks[0].chunk_id, text="   \n  ")
    store.replace_page_chunks(URL_A, chunks)
    report = verify_store(manifest, store, settings, SOURCES)
    assert kinds(report) == ["empty-chunk"]


# -- chunk-too-large -----------------------------------------------------


def test_an_oversized_chunk_not_flagged_is_a_violation(manifest, store, settings):
    add_ok(manifest, URL_A)
    chunks = make_page(URL_A, 1)
    big_text = "x" * 500  # far past target(100) + overlap(20)
    chunks[0] = make_chunk(
        URL_A, 0, chunk_id=chunks[0].chunk_id, text=big_text, is_oversized=False
    )
    store.replace_page_chunks(URL_A, chunks)
    report = verify_store(manifest, store, settings, SOURCES)
    assert kinds(report) == ["chunk-too-large"]


def test_an_oversized_chunk_correctly_flagged_is_not_a_violation(
    manifest, store, settings
):
    add_ok(manifest, URL_A)
    chunks = make_page(URL_A, 1)
    chunks[0] = make_chunk(
        URL_A, 0, chunk_id=chunks[0].chunk_id, text="x" * 500, is_oversized=True
    )
    store.replace_page_chunks(URL_A, chunks)
    report = verify_store(manifest, store, settings, SOURCES)
    assert report.ok


def test_a_chunk_within_the_overlap_margin_is_not_flagged(manifest, store, settings):
    # target=100, overlap=20 -> ceiling=122 (+2 for the "\n\n" chunker.py
    # joins overlap with, see chunker.overlap_ceiling); 115 is over target
    # but within the legitimate overlap margin, so not a violation.
    add_ok(manifest, URL_A)
    chunks = make_page(URL_A, 1)
    chunks[0] = make_chunk(URL_A, 0, chunk_id=chunks[0].chunk_id, text="x" * 115)
    store.replace_page_chunks(URL_A, chunks)
    report = verify_store(manifest, store, settings, SOURCES)
    assert report.ok


def test_a_chunk_just_past_the_ceiling_is_flagged(manifest, store, settings):
    add_ok(manifest, URL_A)
    chunks = make_page(URL_A, 1)
    chunks[0] = make_chunk(URL_A, 0, chunk_id=chunks[0].chunk_id, text="x" * 123)
    store.replace_page_chunks(URL_A, chunks)
    report = verify_store(manifest, store, settings, SOURCES)
    assert kinds(report) == ["chunk-too-large"]


def test_a_chunk_exactly_at_the_ceiling_is_not_flagged(manifest, store, settings):
    add_ok(manifest, URL_A)
    chunks = make_page(URL_A, 1)
    chunks[0] = make_chunk(URL_A, 0, chunk_id=chunks[0].chunk_id, text="x" * 122)
    store.replace_page_chunks(URL_A, chunks)
    report = verify_store(manifest, store, settings, SOURCES)
    assert report.ok


# -- dangling-neighbor (only reachable by bypassing the store's own -------
# -- write-time validation, i.e. real corruption) -------------------------


def test_a_dangling_next_chunk_id_is_a_violation(manifest, store, settings):
    add_ok(manifest, URL_A)
    store.replace_page_chunks(URL_A, make_page(URL_A, 2))
    store._conn.execute(
        "UPDATE chunks SET next_chunk_id = 'does-not-exist' WHERE chunk_id = ?",
        (f"{URL_A}#0",),
    )
    report = verify_store(manifest, store, settings, SOURCES)
    assert kinds(report) == ["dangling-neighbor"]
    assert "does-not-exist" in report.violations[0].detail


def test_a_dangling_prev_chunk_id_is_a_violation(manifest, store, settings):
    add_ok(manifest, URL_A)
    store.replace_page_chunks(URL_A, make_page(URL_A, 2))
    store._conn.execute(
        "UPDATE chunks SET prev_chunk_id = 'does-not-exist' WHERE chunk_id = ?",
        (f"{URL_A}#1",),
    )
    report = verify_store(manifest, store, settings, SOURCES)
    assert kinds(report) == ["dangling-neighbor"]


def test_a_neighbor_pointing_at_a_different_pages_chunk_is_a_violation(
    manifest, store, settings
):
    add_ok(manifest, URL_A)
    add_ok(manifest, URL_B)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1))
    store.replace_page_chunks(URL_B, make_page(URL_B, 1))
    store._conn.execute(
        "UPDATE chunks SET next_chunk_id = ? WHERE chunk_id = ?",
        (f"{URL_B}#0", f"{URL_A}#0"),
    )
    report = verify_store(manifest, store, settings, SOURCES)
    assert kinds(report) == ["dangling-neighbor"]


# -- missing-field (also only reachable via direct corruption) --------------


def test_a_chunk_with_an_empty_source_is_a_violation(manifest, store, settings):
    add_ok(manifest, URL_A)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1))
    store._conn.execute(
        "UPDATE chunks SET source = '' WHERE chunk_id = ?", (f"{URL_A}#0",)
    )
    report = verify_store(manifest, store, settings, SOURCES)
    assert kinds(report) == ["missing-field"]
    assert "source" in report.violations[0].detail


# -- everything at once --------------------------------------------------


def test_all_violations_are_reported_together_not_just_the_first(
    manifest, store, settings
):
    add_ok(manifest, URL_A)  # will have zero chunks
    add_ok(manifest, URL_B)
    chunks = make_page(URL_B, 2)
    chunks[0] = make_chunk(URL_B, 0, chunk_id=chunks[0].chunk_id, text="")
    chunks[1] = make_chunk(
        URL_B,
        1,
        chunk_id=chunks[1].chunk_id,
        prev_chunk_id=chunks[0].chunk_id,
        text="x" * 500,
        is_oversized=False,
    )
    store.replace_page_chunks(URL_B, chunks)
    store._conn.execute(
        "UPDATE chunks SET next_chunk_id = 'ghost' WHERE chunk_id = ?",
        (f"{URL_B}#1",),
    )

    report = verify_store(manifest, store, settings, SOURCES, complete=True)
    assert kinds(report) == [
        "chunk-too-large",
        "dangling-neighbor",
        "empty-chunk",
        "no-chunks",
    ]
    formatted = format_report(report, settings)
    assert "FAILED: 4 violations" in formatted
    for kind in ("no-chunks", "empty-chunk", "chunk-too-large", "dangling-neighbor"):
        assert f"[{kind}]" in formatted


# -- pages no longer ok are not checked or flagged as orphans ---------------


def test_chunks_for_a_page_no_longer_ok_are_not_checked(manifest, store, settings):
    manifest.add_pending("aws-en", [URL_A])
    manifest.record_ok(URL_A, http_status=200, sha256="a" * 64, size_bytes=1)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1))
    manifest.record_gone(URL_A)  # no longer ok; its stale chunks are untouched

    report = verify_store(manifest, store, settings, SOURCES)
    assert report.ok
    assert report.checked == 0
    assert report.total_chunks == 1  # still counted in the overall total


# -- --by-section --------------------------------------------------------


def test_by_section_table_is_included_only_when_requested(manifest, store, settings):
    add_ok(manifest, URL_A)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1))
    report = verify_store(manifest, store, settings, SOURCES)
    assert "per-section" not in format_report(report, settings)
    assert "per-section" in format_report(report, settings, by_section=True)


def test_by_section_table_reflects_the_stores_sections(manifest, store, settings):
    add_ok(manifest, URL_A)
    add_ok(manifest, URL_B)
    store.replace_page_chunks(URL_A, make_page(URL_A, 1))
    store.replace_page_chunks(URL_B, make_page(URL_B, 1))
    report = verify_store(manifest, store, settings, SOURCES)
    assert set(report.sections) == {("aws-en", "sql"), ("aws-en", "pyspark")}
    out = format_report(report, settings, by_section=True)
    assert "aws-en/sql" in out
    assert "aws-en/pyspark" in out
    assert "total" in out
