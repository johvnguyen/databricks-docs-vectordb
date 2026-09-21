import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from databricks_docs_vectordb.scraping.config import Settings
from databricks_docs_vectordb.scraping.manifest import Manifest, Page
from databricks_docs_vectordb.scraping.urls import url_to_rel_path
from databricks_docs_vectordb.scraping.verify import (
    TOP_LEVEL,
    format_report,
    format_sections,
    section_of,
    verify_snapshot,
)

AWS = "https://docs.databricks.com/aws/en/"
API = "https://docs.databricks.com/api/"
CONTACT = "me@example.com"

PAGE = (
    b"<html><head><title>T</title></head>"
    b"<body><article><h1>Title</h1><p>text</p></article></body></html>"
)


def rich(features: str) -> bytes:
    """A docs page with the given features inside its article: h1, pre, table."""
    parts = []
    if "h1" in features:
        parts.append("<h1>Title</h1>")
    if "pre" in features:
        parts.append("<pre><code>x = 1</code></pre>")
    if "table" in features:
        parts.append("<table><tr><td>1</td></tr></table>")
    return (
        "<html><head><title>T</title></head><body>"
        f"<article>{''.join(parts)}</article></body></html>"
    ).encode()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(contact=CONTACT, data_dir=tmp_path / "data")


@pytest.fixture
def manifest(settings):
    with Manifest(settings.manifest_path) as m:
        yield m


def add_ok(
    manifest: Manifest,
    settings: Settings,
    url: str,
    body: bytes = PAGE,
    *,
    source: str | None = None,
    write: bool = True,
    **record,
) -> Path:
    """A page as the fetcher leaves it: a row and a matching file."""
    source = source or ("api" if url.startswith(API) else "aws-en")
    manifest.add_pending(source, [url])
    path = settings.raw_dir / url_to_rel_path(url)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    fields = dict(
        http_status=200,
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
        etag='"e"',
        now="2026-09-21T01:00:00Z",
    )
    manifest.record_ok(url, **{**fields, **record})
    return path


def kinds(findings) -> list[str]:
    return sorted(f.kind for f in findings)


def run(settings, manifest, **kwargs):
    return verify_snapshot(settings, manifest, **kwargs)


# -- a clean snapshot ----------------------------------------------------


def test_a_clean_snapshot_has_no_violations(settings, manifest):
    for url in (AWS + "delta/", AWS + "sql/a", API + "workspace/x"):
        add_ok(manifest, settings, url)
    report = run(settings, manifest)
    assert report.ok
    assert report.violations == [] and report.warnings == []
    assert report.checked == 3


def test_an_empty_manifest_is_clean(settings, manifest):
    report = run(settings, manifest)
    assert report.ok and report.checked == 0 and report.sections == {}


def test_a_snapshot_with_no_raw_directory_yet_is_fine_when_nothing_is_ok(
    settings, manifest
):
    manifest.add_pending("aws-en", [AWS + "a", AWS + "b"])
    assert not settings.raw_dir.exists()
    assert run(settings, manifest).ok


# -- the violations the doc lists ---------------------------------------


def test_an_ok_row_whose_file_is_missing(settings, manifest):
    add_ok(manifest, settings, AWS + "a", write=False)
    (v,) = run(settings, manifest).violations
    assert (v.kind, v.where) == ("missing-file", AWS + "a")
    assert "aws/en/a.html" in v.detail


def test_an_ok_row_whose_path_is_a_directory_counts_as_missing(settings, manifest):
    path = add_ok(manifest, settings, AWS + "a", write=False)
    path.mkdir(parents=True)
    assert kinds(run(settings, manifest).violations) == ["missing-file"]


def test_a_file_with_no_manifest_row(settings, manifest):
    add_ok(manifest, settings, AWS + "a")
    stray = settings.raw_dir / "aws/en/mystery/index.html"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(PAGE)

    (v,) = run(settings, manifest).violations

    assert (v.kind, v.where) == ("orphan-file", "aws/en/mystery/index.html")


def test_a_sha256_mismatch(settings, manifest):
    path = add_ok(manifest, settings, AWS + "a")
    path.write_bytes(PAGE.replace(b"text", b"TEXT"))  # same size, different bytes
    report = run(settings, manifest)
    assert kinds(report.violations) == ["hash-mismatch"]


def test_a_size_mismatch_is_reported_separately(settings, manifest):
    add_ok(manifest, settings, AWS + "a", size_bytes=1)
    assert kinds(run(settings, manifest).violations) == ["size-mismatch"]


@pytest.mark.parametrize(
    ("body", "detail"),
    [
        (
            b"<html><head><title>T</title></head><body>oops</body></html>",
            "no <article>",
        ),
        (b"<html><body><article>hi</article></body></html>", "no <title>"),
        (b"", "no <article>"),
    ],
    ids=["no-article", "no-title", "empty"],
)
def test_a_body_that_does_not_look_like_a_docs_page(settings, manifest, body, detail):
    add_ok(manifest, settings, AWS + "a", body)
    (v,) = run(settings, manifest).violations
    assert v.kind == "bad-content" and detail in v.detail


def test_two_rows_sharing_one_file(settings, manifest):
    class Stub:
        def __init__(self, pages):
            self._pages = pages

        def all_pages(self):
            return self._pages

    add_ok(manifest, settings, AWS + "a")
    first = manifest.get(AWS + "a")
    second = replace(first, url=AWS + "b")  # same rel_path as the first
    report = verify_snapshot(settings, Stub([first, second]))

    dup = [v for v in report.violations if v.kind == "duplicate-path"]
    assert len(dup) == 1
    assert dup[0].where == "aws/en/a.html"
    assert AWS + "a" in dup[0].detail and AWS + "b" in dup[0].detail


def test_pending_rows_after_a_supposedly_complete_run(settings, manifest):
    add_ok(manifest, settings, AWS + "done")
    manifest.add_pending("aws-en", [AWS + "todo1", AWS + "todo2"])

    assert run(settings, manifest).ok  # by default, pending is just a status
    report = run(settings, manifest, complete=True)

    (v,) = report.violations
    assert v.kind == "pending-remaining"
    assert "2 pages are still pending" in v.detail


def test_complete_is_satisfied_when_nothing_is_pending(settings, manifest):
    add_ok(manifest, settings, AWS + "done")
    assert run(settings, manifest, complete=True).ok


# -- other consistency checks -------------------------------------------


def test_an_ok_row_missing_its_hash_or_timestamps(settings, manifest):
    add_ok(manifest, settings, AWS + "a")
    manifest._conn.execute("UPDATE pages SET sha256 = NULL, checked_at = NULL")
    manifest._conn.commit()
    (v,) = run(settings, manifest).violations
    assert v.kind == "incomplete-row"
    assert "sha256" in v.detail and "checked_at" in v.detail


def test_a_rel_path_that_does_not_match_the_url(settings, manifest):
    add_ok(manifest, settings, AWS + "a")
    manifest._conn.execute("UPDATE pages SET rel_path = 'aws/en/other.html'")
    manifest._conn.commit()
    report = run(settings, manifest)
    assert "path-mismatch" in kinds(report.violations)


def test_a_rel_path_that_escapes_the_raw_directory_is_reported_and_never_read(
    settings, manifest, tmp_path
):
    add_ok(manifest, settings, AWS + "a")
    outside = settings.data_dir / "outside.html"  # settings.raw_dir/../outside.html
    outside.write_bytes(b"secret, and not even a docs page")
    manifest._conn.execute("UPDATE pages SET rel_path = '../outside.html'")
    manifest._conn.commit()

    report = run(settings, manifest)

    assert "path-escape" in kinds(report.violations)
    # the outside file was never opened: no hash or content findings about it
    assert not {"hash-mismatch", "bad-content", "size-mismatch"} & set(
        kinds(report.violations)
    )


def test_every_violation_is_reported_not_just_the_first(settings, manifest):
    p1 = add_ok(manifest, settings, AWS + "hash")
    add_ok(manifest, settings, AWS + "missing", write=False)
    add_ok(manifest, settings, AWS + "empty", b"")
    p1.write_bytes(PAGE.replace(b"text", b"TEXT"))
    (settings.raw_dir / "orphan.html").write_bytes(PAGE)
    manifest.add_pending("aws-en", [AWS + "todo"])

    report = run(settings, manifest, complete=True)

    assert kinds(report.violations) == sorted(
        [
            "hash-mismatch",
            "missing-file",
            "bad-content",
            "orphan-file",
            "pending-remaining",
        ]
    )


# -- things that are not violations -------------------------------------


def test_failed_and_gone_pages_are_not_violations_even_when_complete(
    settings, manifest
):
    add_ok(manifest, settings, AWS + "fine")
    manifest.add_pending("api", [API])  # the accepted empty shell
    manifest.record_failed(
        API, error="HTTP 200 but the page has no <article> element", attempts=5
    )
    manifest.add_pending("aws-en", [AWS + "old"])
    manifest.record_gone(AWS + "old")

    report = run(settings, manifest, complete=True)

    assert report.ok
    assert report.statuses["failed"] == 1 and report.statuses["gone"] == 1


def test_files_kept_for_removed_and_gone_pages_are_not_orphans(settings, manifest):
    add_ok(manifest, settings, AWS + "old")
    manifest.record_gone(AWS + "old")
    add_ok(manifest, settings, AWS + "dropped")
    manifest.mark_removed("aws-en", [AWS + "old"])
    assert run(settings, manifest).ok


def test_a_leftover_temp_file_is_a_warning_not_a_violation(settings, manifest):
    path = add_ok(manifest, settings, AWS + "a")
    (path.parent / ".a.html.k3j2x9").write_bytes(b"half a page")

    report = run(settings, manifest)

    assert report.ok
    (w,) = report.warnings
    assert (w.kind, w.where) == ("stray-temp-file", "aws/en/.a.html.k3j2x9")


def test_only_ok_rows_are_integrity_checked(settings, manifest):
    # A pending row whose file exists but is garbage: harmless (a crash between
    # writing the file and updating the manifest), and the row is not "ok".
    manifest.add_pending("aws-en", [AWS + "a"])
    path = settings.raw_dir / "aws/en/a.html"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not html at all")
    assert run(settings, manifest).ok


# -- per-section table ---------------------------------------------------


def test_sections_count_features_inside_the_article_only(settings, manifest):
    add_ok(manifest, settings, AWS + "sql/a", rich("h1 pre table"))
    add_ok(manifest, settings, AWS + "sql/b", rich("h1"))
    add_ok(manifest, settings, AWS + "pyspark/c", rich(""))
    add_ok(manifest, settings, API + "workspace/scopes", rich("h1 pre"))

    sections = run(settings, manifest).sections

    assert {k: vars(v) for k, v in sections.items()} == {
        ("aws-en", "sql"): dict(pages=2, article=2, h1=2, code=1, tables=1),
        ("aws-en", "pyspark"): dict(pages=1, article=1, h1=0, code=0, tables=0),
        ("api", "workspace"): dict(pages=1, article=1, h1=1, code=1, tables=0),
    }


def test_features_outside_the_article_do_not_count(settings, manifest):
    body = (
        b"<html><head><title>T</title></head><body>"
        b"<nav><h1>Site</h1><pre>x</pre><table></table></nav>"
        b"<article><p>plain</p></article>"
        b"<footer><h1>Foot</h1></footer></body></html>"
    )
    add_ok(manifest, settings, AWS + "sql/a", body)
    stats = run(settings, manifest).sections[("aws-en", "sql")]
    assert (stats.article, stats.h1, stats.code, stats.tables) == (1, 0, 0, 0)


def test_feature_matching_ignores_case_and_attributes(settings, manifest):
    body = (
        b"<HTML><TITLE>T</TITLE><ARTICLE class=x><H1 id=a>t</H1>"
        b"<PRE class=c>x</PRE><TABLE border=1></TABLE></ARTICLE></HTML>"
    )
    add_ok(manifest, settings, AWS + "sql/a", body)
    stats = run(settings, manifest).sections[("aws-en", "sql")]
    assert (stats.article, stats.h1, stats.code, stats.tables) == (1, 1, 1, 1)


def test_a_page_without_an_article_is_counted_but_has_no_features(settings, manifest):
    add_ok(
        manifest, settings, AWS + "sql/a", b"<html><title>T</title><h1>x</h1></html>"
    )
    stats = run(settings, manifest).sections[("aws-en", "sql")]
    assert (stats.pages, stats.article, stats.h1) == (1, 0, 0)


def test_section_counts_add_up_to_the_ok_rows(settings, manifest):
    urls = [
        AWS + "sql/a",
        AWS + "sql/b/c",
        AWS + "delta/",
        AWS + "introduction",
        AWS,
        API + "workspace/x",
        API + "iam/v2/y",
    ]
    for url in urls:
        add_ok(manifest, settings, url)
    manifest.add_pending("aws-en", [AWS + "still-pending"])  # not counted

    report = run(settings, manifest)

    assert report.checked == len(urls)
    assert sum(s.pages for s in report.sections.values()) == len(urls)
    assert set(report.sections) == {
        ("aws-en", "sql"),
        ("aws-en", "delta"),
        ("aws-en", TOP_LEVEL),
        ("api", "workspace"),
        ("api", "iam"),
    }


@pytest.mark.parametrize(
    ("url", "source", "expected"),
    [
        (AWS + "sql/language-manual/x", "aws-en", ("aws-en", "sql")),
        (AWS + "delta/", "aws-en", ("aws-en", "delta")),  # a section's index page
        (AWS + "delta", "aws-en", ("aws-en", TOP_LEVEL)),  # a lone top-level page
        (AWS, "aws-en", ("aws-en", TOP_LEVEL)),  # the root itself
        (API + "workspace/scopes", "api", ("api", "workspace")),
        (API, "api", ("api", TOP_LEVEL)),
        (AWS + "sql/a", "gcp-en", ("gcp-en", "(unknown)")),  # source not configured
    ],
)
def test_section_of(settings, url, source, expected):
    page = Page(
        url=url,
        source=source,
        rel_path="x",
        status="ok",
        http_status=200,
        final_url=None,
        etag=None,
        last_modified=None,
        sha256=None,
        size_bytes=None,
        first_seen_at="2026-09-21T00:00:00Z",
        fetched_at=None,
        checked_at=None,
        attempts=0,
        error=None,
    )
    assert section_of(page, settings.sources) == expected


# -- the printed report --------------------------------------------------


def test_a_clean_report_ends_with_ok(settings, manifest):
    add_ok(manifest, settings, AWS + "a")
    manifest.add_pending("aws-en", [AWS + "todo"])
    text = format_report(run(settings, manifest), settings)
    lines = text.splitlines()
    assert lines[0].startswith("checked 1 fetched pages against")
    assert "manifest: 1 pending, 1 ok" in text
    assert lines[-1] == "OK"
    assert "violations" not in text and "per-section" not in text


def test_a_failing_report_lists_every_violation_and_says_how_many(settings, manifest):
    add_ok(manifest, settings, AWS + "a", write=False)
    add_ok(manifest, settings, AWS + "b", b"")
    text = format_report(run(settings, manifest), settings)
    assert "violations (2):" in text
    assert f"[missing-file] {AWS}a:" in text
    assert f"[bad-content] {AWS}b:" in text
    assert text.splitlines()[-1] == "FAILED: 2 violations"
    assert "OK" not in text.splitlines()


def test_warnings_are_printed_but_do_not_fail_the_report(settings, manifest):
    path = add_ok(manifest, settings, AWS + "a")
    (path.parent / ".tmp").write_bytes(b"x")
    text = format_report(run(settings, manifest), settings)
    assert "warnings (1):" in text and "[stray-temp-file]" in text
    assert text.splitlines()[-1] == "OK"


def test_the_report_notes_failed_and_gone_pages_without_failing(settings, manifest):
    manifest.add_pending("api", [API])
    manifest.record_failed(API, error="x", attempts=5)
    text = format_report(run(settings, manifest), settings)
    assert "note: 1 pages the site would not give us" in text
    assert text.splitlines()[-1] == "OK"


def test_the_section_table_is_printed_only_on_request(settings, manifest):
    add_ok(manifest, settings, AWS + "sql/a", rich("h1 pre"))
    report = run(settings, manifest)
    assert "per-section" not in format_report(report, settings)
    table = format_report(report, settings, by_section=True)
    assert "per-section (fetched pages, and how many have each feature):" in table
    assert "aws-en/sql" in table


def test_the_section_table_has_aligned_columns_and_a_total(settings, manifest):
    add_ok(manifest, settings, AWS + "sql/a", rich("h1 pre table"))
    add_ok(manifest, settings, AWS + "sql/b", rich("h1"))
    add_ok(manifest, settings, API + "workspace/x", rich("h1 pre"))
    lines = format_sections(run(settings, manifest).sections)

    assert lines[1].split() == [
        "source/section",
        "pages",
        "<article",
        "<h1>",
        "code",
        "tables",
    ]
    rows = {line.split()[0]: line.split()[1:] for line in lines[2:] if line.strip("- ")}
    assert rows["aws-en/sql"] == ["2", "2", "2", "1", "1"]
    assert rows["api/workspace"] == ["1", "1", "1", "1", "0"]
    assert rows["total"] == ["3", "3", "3", "2", "1"]
    data_lines = [line for line in lines[1:] if line.strip("- ")]
    assert len({len(line) for line in data_lines}) == 1  # every row is the same width


def test_the_section_table_with_no_pages_says_so():
    assert format_sections({}) == ["per-section: no fetched pages"]
