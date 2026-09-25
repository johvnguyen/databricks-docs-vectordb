from pathlib import Path

import pytest

from databricks_docs_vectordb.parsing.config import Settings
from databricks_docs_vectordb.parsing.pipeline import (
    PageResult,
    is_current,
    parse_page,
    parse_pages,
)
from databricks_docs_vectordb.parsing.store import ChunkStore
from databricks_docs_vectordb.scraping.manifest import Page

URL = "https://docs.databricks.com/aws/en/example/"


def page(
    rel_path: str = "aws/en/example/index.html",
    *,
    url: str = URL,
    source: str = "aws-en",
    sha256: str = "ab" * 32,
    fetched_at: str = "2026-09-11T00:00:00Z",
) -> Page:
    return Page(
        url=url,
        source=source,
        rel_path=rel_path,
        status="ok",
        http_status=200,
        final_url=None,
        etag=None,
        last_modified=None,
        sha256=sha256,
        size_bytes=1234,
        first_seen_at="2026-09-01T00:00:00Z",
        fetched_at=fetched_at,
        checked_at="2026-09-11T00:00:00Z",
        attempts=0,
        error=None,
    )


def make_html(title: str = "Example", *, h2_count: int = 2) -> str:
    sections = "".join(
        f'<h2 class="anchor" id="s{i}">Section {i}'
        f'<a href="#s{i}" class="hash-link">\u200b</a></h2>'
        f"<p>Content for section {i}.</p>"
        for i in range(h2_count)
    )
    return f"""<html><head>
<title>{title} | Databricks on AWS</title>
<meta name="description" content="An example page about {title}.">
<script type="application/ld+json">
{{"@type": "BreadcrumbList", "itemListElement": [
    {{"@type": "ListItem", "position": 1, "name": "{title}", "item": "x"}}
]}}
</script>
</head><body><article>
<nav class="theme-doc-breadcrumbs breadcrumbsContainer"></nav>
<div class="row theme-doc-footer-edit-meta-row">
<span class="theme-last-updated">Last updated on <b><time datetime="2026-09-11T00:00:00.000Z" itemprop="dateModified">Sep 11, 2026</time></b></span>
</div>
<div class="theme-doc-markdown markdown">
<header><h1>{title}</h1></header>
<p>Intro paragraph.</p>
{sections}
</div></article></body></html>"""


def write_page(raw_dir: Path, rel_path: str, html: str) -> None:
    path = raw_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return tmp_path / "raw"


@pytest.fixture
def store(tmp_path: Path):
    with ChunkStore(tmp_path / "parsed.sqlite") as s:
        yield s


@pytest.fixture
def settings() -> Settings:
    return Settings(target_chunk_size=10_000)  # small pages fit in one chunk


# -- a fresh page gets parsed -------------------------------------------


def test_a_never_parsed_page_is_parsed(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html())
    result = parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)
    assert result == PageResult(URL, "parsed", result.chunk_count)
    assert result.chunk_count > 0


def test_parsing_actually_writes_chunks_to_the_store(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html())
    parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)
    chunks = store.get_chunks(URL)
    assert chunks
    assert chunks[0].url == URL
    assert chunks[0].source == "aws-en"
    assert chunks[0].cloud == "aws"
    assert chunks[0].doc_title == "Example"
    assert chunks[0].page_sha256 == "ab" * 32
    assert chunks[0].fetched_at == "2026-09-11T00:00:00Z"


def test_a_page_with_multiple_h2_sections_produces_multiple_chunks(
    raw_dir, store, settings
):
    write_page(raw_dir, "aws/en/example/index.html", make_html(h2_count=3))
    result = parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)
    assert result.outcome == "parsed"
    assert result.chunk_count == 4  # preamble + 3 sections
    headings = [c.heading_path for c in store.get_chunks(URL)]
    assert headings == ["", "Section 0", "Section 1", "Section 2"]


# -- an unchanged page is skipped ------------------------------------


def test_an_already_current_page_is_skipped(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html())
    first = parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)
    assert first.outcome == "parsed"

    second = parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)
    assert second == PageResult(URL, "skipped", first.chunk_count)


def test_skipping_does_not_touch_the_stored_chunks(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html())
    parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)
    before = store.get_chunks(URL)

    parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)
    after = store.get_chunks(URL)
    assert before == after


def test_skipping_never_reads_the_raw_file(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html())
    parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)

    # Delete the raw file: if parse_page tried to re-read it, this would
    # raise. A skip must never touch the raw snapshot.
    (raw_dir / "aws/en/example/index.html").unlink()
    result = parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)
    assert result.outcome == "skipped"


# -- a changed page is reparsed ----------------------------------------


def test_a_changed_sha256_triggers_a_reparse(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html("Example"))
    first = parse_page(
        page(sha256="a" * 64), raw_dir=raw_dir, store=store, settings=settings
    )
    assert first.outcome == "parsed"

    write_page(raw_dir, "aws/en/example/index.html", make_html("Renamed Title"))
    second = parse_page(
        page(sha256="b" * 64), raw_dir=raw_dir, store=store, settings=settings
    )
    assert second.outcome == "parsed"
    assert store.get_chunks(URL)[0].doc_title == "Renamed Title"
    assert store.get_chunks(URL)[0].page_sha256 == "b" * 64


def test_reparsing_replaces_rather_than_appends(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html(h2_count=3))
    parse_page(page(sha256="a" * 64), raw_dir=raw_dir, store=store, settings=settings)
    before_count = len(store.get_chunks(URL))

    write_page(raw_dir, "aws/en/example/index.html", make_html(h2_count=1))
    parse_page(page(sha256="b" * 64), raw_dir=raw_dir, store=store, settings=settings)
    after = store.get_chunks(URL)
    assert len(after) < before_count
    assert len(after) == len({c.chunk_id for c in after})


# -- parse_pages: batch orchestration -----------------------------------


def test_parse_pages_yields_one_result_per_page_in_order(raw_dir, store, settings):
    write_page(raw_dir, "a/index.html", make_html("Page A"))
    write_page(raw_dir, "b/index.html", make_html("Page B"))
    pages = [
        page("a/index.html", url="https://docs.databricks.com/aws/en/a/"),
        page("b/index.html", url="https://docs.databricks.com/aws/en/b/"),
    ]
    results = list(parse_pages(pages, raw_dir=raw_dir, store=store, settings=settings))
    assert [r.url for r in results] == [p.url for p in pages]
    assert all(r.outcome == "parsed" for r in results)
    assert store.get_chunks(pages[0].url)[0].doc_title == "Page A"
    assert store.get_chunks(pages[1].url)[0].doc_title == "Page B"


def test_parse_pages_skips_pages_already_current_within_the_batch(
    raw_dir, store, settings
):
    write_page(raw_dir, "a/index.html", make_html("Page A"))
    p = page("a/index.html", url="https://docs.databricks.com/aws/en/a/")
    parse_page(p, raw_dir=raw_dir, store=store, settings=settings)

    results = list(parse_pages([p], raw_dir=raw_dir, store=store, settings=settings))
    assert results == [PageResult(p.url, "skipped", results[0].chunk_count)]


def test_parse_pages_on_an_empty_iterable_yields_nothing(raw_dir, store, settings):
    assert list(parse_pages([], raw_dir=raw_dir, store=store, settings=settings)) == []


# -- a missing raw file is a real error, not swallowed ------------------


def test_a_missing_raw_file_raises_rather_than_being_silently_skipped(
    raw_dir, store, settings
):
    with pytest.raises(FileNotFoundError):
        parse_page(page(), raw_dir=raw_dir, store=store, settings=settings)


# -- is_current: the same predicate the CLI uses to pick candidates ---------


def test_is_current_is_false_for_a_never_parsed_page(store):
    assert is_current(page(), store) is False


def test_is_current_is_true_once_parsed_with_a_matching_sha256(
    raw_dir, store, settings
):
    write_page(raw_dir, "aws/en/example/index.html", make_html())
    p = page(sha256="a" * 64)
    parse_page(p, raw_dir=raw_dir, store=store, settings=settings)
    assert is_current(p, store) is True


def test_is_current_is_false_once_the_sha256_changes(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html())
    parse_page(page(sha256="a" * 64), raw_dir=raw_dir, store=store, settings=settings)
    assert is_current(page(sha256="b" * 64), store) is False


# -- force=True (--refresh): reparse regardless of currency -----------------


def test_force_reparses_an_already_current_page(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html("Original"))
    p = page(sha256="a" * 64)
    parse_page(p, raw_dir=raw_dir, store=store, settings=settings)

    write_page(raw_dir, "aws/en/example/index.html", make_html("Original"))  # unchanged
    result = parse_page(p, raw_dir=raw_dir, store=store, settings=settings, force=True)
    assert result.outcome == "parsed"


def test_without_force_an_unchanged_page_is_still_skipped(raw_dir, store, settings):
    write_page(raw_dir, "aws/en/example/index.html", make_html())
    p = page(sha256="a" * 64)
    parse_page(p, raw_dir=raw_dir, store=store, settings=settings)
    result = parse_page(p, raw_dir=raw_dir, store=store, settings=settings, force=False)
    assert result.outcome == "skipped"


def test_parse_pages_force_applies_to_every_page_in_the_batch(raw_dir, store, settings):
    write_page(raw_dir, "a/index.html", make_html("Page A"))
    p = page(
        "a/index.html", url="https://docs.databricks.com/aws/en/a/", sha256="a" * 64
    )
    parse_page(p, raw_dir=raw_dir, store=store, settings=settings)

    results = list(
        parse_pages([p], raw_dir=raw_dir, store=store, settings=settings, force=True)
    )
    assert results[0].outcome == "parsed"
