from collections import Counter

from databricks_docs_vectordb.scraping.manifest import Page
from databricks_docs_vectordb.scraping.report import (
    format_counts,
    format_manifest,
    format_problems,
)

URL = "https://docs.databricks.com/aws/en/delta/"


def page(**overrides) -> Page:
    fields = dict(
        url=URL,
        source="aws-en",
        rel_path="aws/en/delta/index.html",
        status="failed",
        http_status=None,
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
    return Page(**{**fields, **overrides})


def test_counts_use_a_fixed_status_order_and_skip_zeros():
    counts = Counter(failed=1, ok=12, pending=3)
    assert format_counts(counts) == "3 pending, 12 ok, 1 failed"


def test_an_empty_counter_says_empty():
    assert format_counts(Counter()) == "empty"


def test_manifest_lines_add_a_total_only_for_several_sources():
    one = format_manifest({"aws-en": Counter(pending=2)})
    assert one == ["  aws-en: 2 pending"]

    two = format_manifest({"api": Counter(ok=1), "aws-en": Counter(ok=2, pending=1)})
    assert two == [
        "  api: 1 ok",
        "  aws-en: 1 pending, 2 ok",
        "  all: 1 pending, 3 ok",
    ]


def test_problems_show_the_error_or_the_http_status():
    lines = format_problems(
        "failed",
        [page(error="gave up: HTTP 500"), page(url=URL + "x", http_status=404)],
        total=2,
    )
    assert lines == [
        "failed (2):",
        f"  {URL}  gave up: HTTP 500",
        f"  {URL}x  HTTP 404",
    ]


def test_problems_flatten_and_truncate_long_errors():
    (_, line) = format_problems("failed", [page(error="a\nb" + "c" * 500)], total=1)
    assert "\n" not in line
    assert len(line) < len(URL) + 130


def test_problems_say_how_many_more_there_are():
    lines = format_problems("failed", [page()], total=4)
    assert lines[-1] == "  ... and 3 more"
