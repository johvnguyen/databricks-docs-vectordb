from pathlib import Path

import pytest

from databricks_docs_vectordb.cli import main
from databricks_docs_vectordb.parsing.store import ChunkStore
from databricks_docs_vectordb.scraping.manifest import Manifest

URL_A = "https://docs.databricks.com/aws/en/a/"
URL_B = "https://docs.databricks.com/aws/en/b/"
URL_C = "https://docs.databricks.com/aws/en/c/"


@pytest.fixture
def env(monkeypatch, tmp_path):
    """No real .env, data in a temp dir. Parsing never needs a contact."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABRICKS_DOCS_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def make_html(title: str = "Example") -> str:
    return f"""<html><head><title>{title} | Databricks on AWS</title>
<meta name="description" content="About {title}.">
</head><body><article><div class="theme-doc-markdown markdown">
<header><h1>{title}</h1></header>
<h2 id="s">Section</h2><p>Content for {title}.</p>
</div></article></body></html>"""


def seed(env: Path, pages: dict[str, str]) -> None:
    """Add each url (with its own sha256 stand-in) to the manifest as `ok`,
    and write matching raw HTML at the path the manifest expects."""
    with Manifest(env / "manifest.sqlite") as manifest:
        for i, (url, title) in enumerate(pages.items()):
            manifest.add_pending("aws-en", [url])
            manifest.record_ok(
                url,
                http_status=200,
                sha256=f"{i:064d}",
                size_bytes=100,
                now="2026-09-11T00:00:00Z",
            )
            rel_path = manifest.get(url).rel_path
            html_path = env / "raw" / rel_path
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(make_html(title), encoding="utf-8")


# -- parse run: no manifest yet -----------------------------------------


def test_parse_run_without_a_manifest_is_a_clear_error(env, capsys):
    assert main(["parse", "run"]) == 1
    assert "scrape enumerate" in capsys.readouterr().err


def test_parse_status_without_a_store_says_so_and_exits_zero(env, capsys):
    assert main(["parse", "status"]) == 0
    assert "run" in capsys.readouterr().out


# -- parse run: a fresh corpus -------------------------------------------


def test_parse_run_parses_every_ok_page(env, capsys):
    seed(env, {URL_A: "Page A", URL_B: "Page B"})
    assert main(["parse", "run"]) == 0
    out = capsys.readouterr().out
    assert "2 page(s) considered (0 already current): 2 parsed, 0 skipped" in out

    with ChunkStore(env / "parsed.sqlite") as store:
        assert store.counts().pages == 2


def test_running_parse_run_twice_skips_the_second_time(env, capsys):
    seed(env, {URL_A: "Page A"})
    main(["parse", "run"])
    capsys.readouterr()

    assert main(["parse", "run"]) == 0
    out = capsys.readouterr().out
    # The already-current page is filtered out before even being counted as
    # "considered" -- it was never re-touched, not "skipped" per page.
    assert "0 page(s) considered (1 already current): 0 parsed, 0 skipped" in out


def test_parse_run_only_considers_ok_manifest_rows(env, capsys):
    with Manifest(env / "manifest.sqlite") as manifest:
        manifest.add_pending("aws-en", [URL_A])  # left pending, never fetched
    assert main(["parse", "run"]) == 0
    assert "0 page(s) considered" in capsys.readouterr().out


# -- parse run --limit --------------------------------------------------


def test_parse_run_limit_caps_pages_per_invocation(env, capsys):
    seed(env, {URL_A: "Page A", URL_B: "Page B", URL_C: "Page C"})
    assert main(["parse", "run", "--limit", "2"]) == 0
    assert "2 page(s) considered" in capsys.readouterr().out
    with ChunkStore(env / "parsed.sqlite") as store:
        assert store.counts().pages == 2


def test_parse_run_limit_advances_through_remaining_candidates_across_runs(env, capsys):
    seed(env, {URL_A: "Page A", URL_B: "Page B", URL_C: "Page C"})
    main(["parse", "run", "--limit", "2"])
    capsys.readouterr()
    main(["parse", "run", "--limit", "2"])
    with ChunkStore(env / "parsed.sqlite") as store:
        # The second run's 2 candidates are drawn from the 1 remaining
        # not-yet-current page, so all 3 pages end up parsed, not 4.
        assert store.counts().pages == 3


# -- parse run --refresh -----------------------------------------------


def test_parse_run_refresh_reparses_already_current_pages(env, capsys):
    seed(env, {URL_A: "Page A"})
    main(["parse", "run"])
    capsys.readouterr()

    assert main(["parse", "run", "--refresh"]) == 0
    out = capsys.readouterr().out
    assert "1 page(s) considered (0 already current): 1 parsed, 0 skipped" in out


# -- parse run --sample --------------------------------------------------


def test_parse_run_sample_too_small_is_a_clear_error(env, capsys):
    seed(env, {URL_A: "Page A", URL_B: "Page B", URL_C: "Page C"})
    assert main(["parse", "run", "--sample", "1"]) == 1
    assert "too small" in capsys.readouterr().err


def test_parse_run_sample_covers_every_section(env, capsys):
    seed(env, {URL_A: "Page A", URL_B: "Page B", URL_C: "Page C"})
    assert main(["parse", "run", "--sample", "3"]) == 0
    with ChunkStore(env / "parsed.sqlite") as store:
        assert store.counts().pages == 3


def test_limit_and_sample_are_mutually_exclusive(env, capsys):
    with pytest.raises(SystemExit):
        main(["parse", "run", "--limit", "1", "--sample", "1"])
    assert "not allowed with" in capsys.readouterr().err


# -- parse status ---------------------------------------------------------


def test_parse_status_reports_counts_and_a_per_section_table(env, capsys):
    seed(env, {URL_A: "Page A", URL_B: "Page B"})
    main(["parse", "run"])
    capsys.readouterr()

    assert main(["parse", "status"]) == 0
    out = capsys.readouterr().out
    assert "2 page(s)" in out
    assert "aws-en/a" in out
    assert "aws-en/b" in out
    assert "total" in out


def test_parse_status_never_writes_to_the_store(env, capsys):
    seed(env, {URL_A: "Page A"})
    main(["parse", "run"])
    capsys.readouterr()
    with ChunkStore(env / "parsed.sqlite") as store:
        before = store.get_chunks(URL_A)

    main(["parse", "status"])
    with ChunkStore(env / "parsed.sqlite") as store:
        after = store.get_chunks(URL_A)
    assert before == after


# -- parse verify -----------------------------------------------------------


def test_parse_verify_on_a_fully_parsed_corpus_is_ok(env, capsys):
    seed(env, {URL_A: "Page A", URL_B: "Page B"})
    main(["parse", "run"])
    capsys.readouterr()

    assert main(["parse", "verify"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "OK"


def test_parse_verify_on_a_partial_corpus_is_ok_without_complete(env, capsys):
    # A deliberate --sample/--limit run leaves most pages unparsed; that
    # must not look like corruption by default (mirrors `scrape verify`'s
    # treatment of `pending` rows).
    seed(env, {URL_A: "Page A", URL_B: "Page B"})
    main(["parse", "run", "--limit", "1"])
    capsys.readouterr()

    assert main(["parse", "verify"]) == 0
    out = capsys.readouterr().out
    assert "note: 1 ok page(s) have not been parsed yet" in out
    assert out.splitlines()[-1] == "OK"


def test_parse_verify_complete_fails_on_a_partial_corpus(env, capsys):
    seed(env, {URL_A: "Page A", URL_B: "Page B"})
    main(["parse", "run", "--limit", "1"])
    capsys.readouterr()

    assert main(["parse", "verify", "--complete"]) == 1
    out = capsys.readouterr().out
    assert "no-chunks" in out
    assert "FAILED" in out.splitlines()[-1]


def test_parse_verify_without_a_manifest_is_a_clear_error(env, capsys):
    assert main(["parse", "verify"]) == 1
    assert "scrape enumerate" in capsys.readouterr().err


def test_parse_verify_by_section_includes_the_table(env, capsys):
    seed(env, {URL_A: "Page A", URL_B: "Page B"})
    main(["parse", "run"])
    capsys.readouterr()

    main(["parse", "verify", "--by-section"])
    out = capsys.readouterr().out
    assert "per-section" in out
    assert "aws-en/a" in out
    assert "aws-en/b" in out


def test_parse_verify_by_section_omitted_by_default(env, capsys):
    seed(env, {URL_A: "Page A"})
    main(["parse", "run"])
    capsys.readouterr()

    main(["parse", "verify"])
    assert "per-section" not in capsys.readouterr().out
