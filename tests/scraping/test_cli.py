import logging
import os
import signal
import threading
import time
from pathlib import Path

import httpx
import pytest
import respx

from databricks_docs_vectordb.cli import main
from databricks_docs_vectordb.scraping import fetcher
from databricks_docs_vectordb.scraping.config import Settings
from databricks_docs_vectordb.scraping.manifest import Manifest

FIXTURES = Path(__file__).parent.parent / "fixtures"
ROBOTS = "https://docs.databricks.com/robots.txt"
AWS_SITEMAP = "https://docs.databricks.com/aws/en/sitemap.xml"
API_SITEMAP = "https://docs.databricks.com/api/sitemap.xml"
CONTACT = "me@example.com"
AWS = "https://docs.databricks.com/aws/en/"
PAGES = r"https://docs\.databricks\.com/(aws/en|api)/.*"
PAGE = b"<html><head><title>T</title></head><body><article>hi</article></body></html>"
ETAG = '"v1"'


@pytest.fixture
def env(monkeypatch, tmp_path):
    """No real .env, no real sleeping, data in a temp dir."""
    monkeypatch.chdir(tmp_path)  # so find_dotenv() can't see the repo's .env
    monkeypatch.setenv("DATABRICKS_DOCS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATABRICKS_DOCS_CONTACT", CONTACT)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return tmp_path / "data"


@pytest.fixture
def site():
    with respx.mock(assert_all_called=False) as router:
        router.get(ROBOTS).respond(200, content=(FIXTURES / "robots.txt").read_bytes())
        router.get(AWS_SITEMAP).respond(
            200, content=(FIXTURES / "sitemap-aws-en.xml").read_bytes()
        )
        router.get(API_SITEMAP).respond(
            200, content=(FIXTURES / "sitemap-api.xml").read_bytes()
        )
        # registered last, so the specific routes above win; tests re-mock it
        router.get(url__regex=PAGES, name="pages").respond(
            200, content=PAGE, headers={"etag": ETAG}
        )
        yield router


def test_scrape_enumerate_prints_a_summary_and_fills_the_manifest(env, site, capsys):
    assert main(["scrape", "enumerate"]) == 0

    out = capsys.readouterr().out
    assert "aws-en: 26 pending" in out
    assert "api: 6 pending" in out
    assert CONTACT not in out
    assert (env / "robots.txt").read_bytes() == (FIXTURES / "robots.txt").read_bytes()
    assert len(list((env / "sitemaps").glob("*.xml"))) == 2
    with Manifest(env / "manifest.sqlite") as m:
        assert m.counts()["pending"] == 32


def test_running_enumerate_twice_reports_nothing_new(env, site, capsys):
    main(["scrape", "enumerate"])
    capsys.readouterr()
    assert main(["scrape", "enumerate"]) == 0
    out = capsys.readouterr().out
    assert out.count("0 new, 0 revived, 0 removed") == 2


def test_requests_send_the_contact_in_the_user_agent(env, site):
    main(["scrape", "enumerate"])
    agents = {call.request.headers["User-Agent"] for call in site.calls}
    assert len(agents) == 1
    (agent,) = agents
    assert agent.startswith("databricks-docs-vectordb/")
    assert agent.endswith(f"(+{CONTACT})")


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_missing_contact_exits_before_any_request(
    env, site, monkeypatch, capsys, value
):
    if value is None:
        monkeypatch.delenv("DATABRICKS_DOCS_CONTACT")
    else:
        monkeypatch.setenv("DATABRICKS_DOCS_CONTACT", value)

    assert main(["scrape", "enumerate"]) == 1

    assert "DATABRICKS_DOCS_CONTACT" in capsys.readouterr().err
    assert len(site.calls) == 0
    assert not (env / "manifest.sqlite").exists()


def test_a_failed_fetch_exits_nonzero_with_the_reason_on_stderr(env, site, capsys):
    site.get(ROBOTS).respond(503)
    assert main(["scrape", "enumerate"]) == 1
    captured = capsys.readouterr()
    assert "robots.txt returned HTTP 503" in captured.err
    assert CONTACT not in captured.err
    assert captured.out == ""


def test_an_unknown_command_is_an_argparse_error(env, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["scrape", "bogus"])
    assert excinfo.value.code == 2


def test_a_bare_scrape_is_an_argparse_error(env, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["scrape"])
    assert excinfo.value.code == 2


# =========================================================================
# scrape fetch
# =========================================================================


@pytest.fixture
def enumerated(env, site):
    assert main(["scrape", "enumerate"]) == 0
    site["pages"].reset()
    return env


@pytest.fixture
def fast_failures(monkeypatch):
    """One attempt per page and a tiny cool-down, so failure tests are quick."""
    original = Settings.load

    def load(cls, **kwargs):
        return original(**{**kwargs, "max_attempts": 1, "breaker_cooldown": 0.01})

    monkeypatch.setattr(Settings, "load", classmethod(load))


def fetch(*flags: str) -> int:
    return main(["scrape", "fetch", "--rate", "1000", *flags])


def log_text(data_dir: Path) -> str:
    (path,) = (data_dir / "logs").glob("fetch-*.log")
    return path.read_text()


def conditional(request: httpx.Request) -> httpx.Response:
    if request.headers.get("if-none-match") == ETAG:
        return httpx.Response(304)
    return httpx.Response(200, content=PAGE, headers={"etag": ETAG})


def test_fetch_downloads_every_pending_page_and_reports(enumerated, site, capsys):
    capsys.readouterr()
    assert fetch() == 0

    captured = capsys.readouterr()
    assert "aws-en: 26 ok" in captured.out
    assert "api: 6 ok" in captured.out
    assert "all: 32 ok" in captured.out
    assert site["pages"].call_count == 32
    assert (enumerated / "raw/aws/en/delta/index.html").read_bytes() == PAGE
    assert (enumerated / "raw/api/index.html").read_bytes() == PAGE
    with Manifest(enumerated / "manifest.sqlite") as m:
        assert m.get(AWS + "delta/").etag == ETAG


def test_fetch_logs_progress_to_a_file_and_the_console_without_the_contact(
    enumerated, site, capsys
):
    capsys.readouterr()
    fetch()

    captured = capsys.readouterr()
    text = log_text(enumerated)
    assert "starting: 32 pages" in text
    assert "completed: 32/32 pages (32 ok)" in text
    assert "completed: 32/32 pages" in captured.err  # also on the console
    assert "HTTP Request" not in text  # httpx's per-request lines are silenced
    for output in (text, captured.out, captured.err):
        assert CONTACT not in output
    assert f"log: {next((enumerated / 'logs').glob('fetch-*.log'))}" in captured.out
    assert (enumerated / "logs").is_dir()


def test_log_lines_are_timestamped_in_utc(enumerated, site):
    fetch()
    first = log_text(enumerated).splitlines()[0]
    assert first[:4].isdigit() and first[10] == "T" and " INFO " in first
    assert first.split(" ")[0].endswith("Z")


def test_the_log_file_appends_across_runs(enumerated, site):
    fetch("--limit", "3")
    fetch()
    text = log_text(enumerated)
    assert text.count("starting:") == 2


def test_limit_fetches_only_that_many_and_a_rerun_finishes_the_rest(
    enumerated, site, capsys
):
    capsys.readouterr()
    assert fetch("--limit", "5") == 0
    assert "all: 27 pending, 5 ok" in capsys.readouterr().out

    assert fetch() == 0
    assert "all: 32 ok" in capsys.readouterr().out
    # nothing was requested twice
    assert site["pages"].call_count == 32
    requested = [str(call.request.url) for call in site["pages"].calls]
    assert len(set(requested)) == 32


def test_a_second_fetch_with_nothing_pending_makes_no_requests(enumerated, site):
    fetch()
    site["pages"].reset()
    assert fetch() == 0
    assert site["pages"].call_count == 0
    assert "starting: 0 pages" in log_text(enumerated)


def test_rate_and_concurrency_flags_reach_the_run(enumerated, site):
    assert (
        main(["scrape", "fetch", "--rate", "50", "--concurrency", "3", "--limit", "3"])
        == 0
    )
    assert "50.00 requests/s, 3 in flight" in log_text(enumerated)


def test_defaults_are_the_conservative_ones(enumerated, site):
    site["pages"].mock(return_value=httpx.Response(404))  # keep it quick
    main(["scrape", "fetch", "--limit", "2"])
    assert "2.00 requests/s, 4 in flight" in log_text(enumerated)


def test_refresh_rechecks_fetched_pages_with_conditional_requests(
    enumerated, site, capsys
):
    site["pages"].mock(side_effect=conditional)
    fetch()
    site["pages"].reset()
    site["pages"].mock(side_effect=conditional)
    capsys.readouterr()

    assert fetch("--refresh") == 0

    assert site["pages"].call_count == 32
    assert all(c.request.headers["if-none-match"] == ETAG for c in site["pages"].calls)
    assert "completed: 32/32 pages (32 not_modified)" in log_text(enumerated)
    assert "all: 32 ok" in capsys.readouterr().out
    assert "(refresh)" in log_text(enumerated)


def test_retry_failed_puts_failed_pages_back_in_the_queue(enumerated, site, capsys):
    bad = AWS + "delta/"

    def one_forbidden(request: httpx.Request) -> httpx.Response:
        if str(request.url) == bad:
            return httpx.Response(403)
        return httpx.Response(200, content=PAGE)

    site["pages"].mock(side_effect=one_forbidden)
    capsys.readouterr()
    assert fetch() == 0
    out = capsys.readouterr().out
    assert "1 page failed" in out and "--retry-failed" in out

    # without the flag the failed page is left alone
    site["pages"].reset()
    site["pages"].mock(return_value=httpx.Response(200, content=PAGE))
    assert fetch() == 0
    assert site["pages"].call_count == 0

    # with it, it is tried again (and now succeeds)
    assert fetch("--retry-failed") == 0
    assert site["pages"].call_count == 1
    assert "requeued 1 failed page" in log_text(enumerated)
    with Manifest(enumerated / "manifest.sqlite") as m:
        assert m.get(bad).status == "ok"


def test_fetch_before_enumerate_says_so_and_makes_no_requests(env, site, capsys):
    assert fetch() == 1
    assert "scrape enumerate" in capsys.readouterr().err
    assert len(site.calls) == 0
    assert not (env / "logs").exists()


def test_fetch_without_the_saved_robots_file_refuses(enumerated, site, capsys):
    (enumerated / "robots.txt").unlink()
    assert fetch() == 1
    assert "scrape enumerate" in capsys.readouterr().err
    assert site["pages"].call_count == 0


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_fetch_without_a_contact_exits_before_any_request(
    enumerated, site, monkeypatch, capsys, value
):
    if value is None:
        monkeypatch.delenv("DATABRICKS_DOCS_CONTACT")
    else:
        monkeypatch.setenv("DATABRICKS_DOCS_CONTACT", value)
    site.reset()

    assert fetch() == 1

    assert "DATABRICKS_DOCS_CONTACT" in capsys.readouterr().err
    assert len(site.calls) == 0


@pytest.mark.parametrize(
    "flags",
    [
        ["--rate", "0"],
        ["--rate", "-1"],
        ["--rate", "fast"],
        ["--concurrency", "0"],
        ["--concurrency", "2.5"],
        ["--limit", "0"],
        ["--limit", "-3"],
    ],
    ids=lambda f: " ".join(f),
)
def test_nonsense_flag_values_are_usage_errors(env, flags):
    with pytest.raises(SystemExit) as excinfo:
        main(["scrape", "fetch", *flags])
    assert excinfo.value.code == 2


def test_the_circuit_breaker_ending_a_run_exits_nonzero(
    enumerated, site, fast_failures, capsys
):
    site["pages"].mock(return_value=httpx.Response(500))
    capsys.readouterr()

    assert fetch() == 1

    out = capsys.readouterr().out
    assert "circuit breaker" in out
    assert "still pending" in out
    text = log_text(enumerated)
    assert "circuit breaker tripped (10 consecutive failures, trip 1 of 3)" in text
    assert "breaker_tripped:" in text
    with Manifest(enumerated / "manifest.sqlite") as m:
        counts = m.counts()
    assert counts["failed"] >= 20 and counts["pending"] > 0


def test_ctrl_c_stops_cleanly_with_exit_130_and_a_rerun_resumes(
    enumerated, site, capsys
):
    timer = threading.Timer(0.7, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        capsys.readouterr()
        code = main(["scrape", "fetch", "--rate", "5"])
    finally:
        timer.cancel()

    out = capsys.readouterr().out
    assert code == 130
    assert "interrupted:" in out and "run the same command again to resume" in out
    with Manifest(enumerated / "manifest.sqlite") as m:
        counts = m.counts()
    assert 0 < counts["ok"] < 32
    assert counts["failed"] == 0
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    site["pages"].reset()
    assert fetch() == 0
    assert site["pages"].call_count == counts["pending"]  # only what was left
    with Manifest(enumerated / "manifest.sqlite") as m:
        assert m.counts()["ok"] == 32


def test_a_disk_error_stops_the_run_with_a_clear_message(
    enumerated, site, monkeypatch, capsys
):
    def full_disk(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fetcher, "write_atomic", full_disk)
    assert fetch() == 1
    assert "No space left on device" in capsys.readouterr().err


def test_logging_setup_is_undone_after_a_run(enumerated, site):
    package = logging.getLogger("databricks_docs_vectordb")
    original = package.level
    package.setLevel(logging.ERROR)  # a known level, whatever earlier tests left
    handlers = list(package.handlers)
    try:
        fetch()
        assert list(package.handlers) == handlers
        assert package.level == logging.ERROR
    finally:
        package.setLevel(original)


# =========================================================================
# scrape status
# =========================================================================


def add_pages(data_dir: Path, n: int = 3) -> Manifest:
    manifest = Manifest(data_dir / "manifest.sqlite")
    manifest.add_pending("aws-en", [AWS + f"p{i:02d}" for i in range(n)])
    return manifest


def test_status_summarises_the_manifest_without_the_network(enumerated, site, capsys):
    site.reset()
    capsys.readouterr()

    assert main(["scrape", "status"]) == 0

    out = capsys.readouterr().out
    assert "aws-en: 26 pending" in out
    assert "api: 6 pending" in out
    assert "all: 32 pending" in out
    assert "32 pages still to fetch" in out
    assert len(site.calls) == 0


def test_status_needs_no_contact_address(enumerated, monkeypatch, capsys):
    monkeypatch.delenv("DATABRICKS_DOCS_CONTACT")
    assert main(["scrape", "status"]) == 0
    assert "all: 32 pending" in capsys.readouterr().out


def test_status_after_a_crawl(enumerated, site, capsys):
    fetch()
    capsys.readouterr()
    assert main(["scrape", "status"]) == 0
    out = capsys.readouterr().out
    assert "all: 32 ok" in out
    assert "still to fetch" not in out
    assert "failed (" not in out


def test_status_lists_failed_and_gone_pages_with_reasons(env, capsys):
    with add_pages(env, 4) as m:
        m.record_failed(
            AWS + "p00", error="gave up after 5 attempts: HTTP 500", attempts=5
        )
        m.record_gone(AWS + "p01")
        m.record_ok(AWS + "p02", http_status=200, sha256="ab" * 32, size_bytes=1)

    assert main(["scrape", "status"]) == 0

    out = capsys.readouterr().out
    assert "failed (1):" in out
    assert f"{AWS}p00  gave up after 5 attempts: HTTP 500" in out
    assert "gone (1):" in out
    assert f"{AWS}p01  HTTP 404" in out
    assert "aws-en: 1 pending, 1 ok, 1 gone, 1 failed" in out


def test_status_truncates_long_lists(env, capsys):
    with add_pages(env, 15) as m:
        for i in range(15):
            m.record_failed(AWS + f"p{i:02d}", error="boom", attempts=1)

    main(["scrape", "status"])

    out = capsys.readouterr().out
    assert "failed (15):" in out
    assert out.count("  boom") == 10
    assert "... and 5 more" in out


def test_status_on_an_empty_manifest_points_at_enumerate(env, capsys):
    Manifest(env / "manifest.sqlite").close()
    assert main(["scrape", "status"]) == 0
    assert "manifest is empty" in capsys.readouterr().out


def test_status_before_enumerate_is_an_error(env, capsys):
    assert main(["scrape", "status"]) == 1
    assert "scrape enumerate" in capsys.readouterr().err
    assert not (env / "manifest.sqlite").exists()  # it must not create one


# =========================================================================
# scrape verify
# =========================================================================


def test_verify_passes_a_clean_snapshot_without_the_network_or_a_contact(
    enumerated, site, monkeypatch, capsys
):
    fetch()
    monkeypatch.delenv("DATABRICKS_DOCS_CONTACT")
    site.reset()
    capsys.readouterr()

    assert main(["scrape", "verify"]) == 0

    out = capsys.readouterr().out
    assert "checked 32 fetched pages" in out
    assert out.splitlines()[-1] == "OK"
    assert len(site.calls) == 0


def test_verify_reports_a_corrupted_file_and_exits_nonzero(enumerated, site, capsys):
    fetch()
    victim = enumerated / "raw/aws/en/delta/index.html"
    victim.write_bytes(PAGE.replace(b"hi", b"HI"))
    capsys.readouterr()

    assert main(["scrape", "verify"]) == 1

    out = capsys.readouterr().out
    assert f"[hash-mismatch] {AWS}delta/:" in out
    assert out.splitlines()[-1] == "FAILED: 1 violations"


def test_verify_lists_every_problem_in_one_run(enumerated, site, capsys):
    fetch()
    (enumerated / "raw/aws/en/delta/index.html").unlink()
    (enumerated / "raw/stray.html").write_bytes(PAGE)
    capsys.readouterr()

    assert main(["scrape", "verify"]) == 1

    out = capsys.readouterr().out
    assert "[missing-file]" in out and "[orphan-file] stray.html" in out
    assert "violations (2):" in out


def test_verify_by_section_prints_the_feature_table(enumerated, site, capsys):
    fetch()
    capsys.readouterr()

    assert main(["scrape", "verify", "--by-section"]) == 0

    out = capsys.readouterr().out
    assert "per-section (fetched pages" in out
    assert "aws-en/delta" in out and "api/workspace" in out
    total = next(line for line in out.splitlines() if line.strip().startswith("total"))
    assert total.split()[1:3] == ["32", "32"]  # 32 pages, all with an <article


def test_verify_without_by_section_omits_the_table(enumerated, site, capsys):
    fetch()
    capsys.readouterr()
    main(["scrape", "verify"])
    assert "per-section" not in capsys.readouterr().out


def test_verify_treats_pending_pages_as_fine_unless_told_the_crawl_is_complete(
    enumerated, site, capsys
):
    fetch("--limit", "5")
    capsys.readouterr()

    assert main(["scrape", "verify"]) == 0
    assert main(["scrape", "verify", "--complete"]) == 1
    assert "[pending-remaining]" in capsys.readouterr().out

    fetch()
    assert main(["scrape", "verify", "--complete"]) == 0


def test_verify_before_enumerate_is_an_error_and_creates_nothing(env, capsys):
    assert main(["scrape", "verify"]) == 1
    assert "scrape enumerate" in capsys.readouterr().err
    assert not (env / "manifest.sqlite").exists()


def test_verify_changes_nothing_on_disk(enumerated, site):
    fetch()

    def snapshot():
        return {
            p.relative_to(enumerated): (
                p.stat().st_size,
                p.read_bytes() if p.suffix == ".html" else None,
            )
            for p in enumerated.rglob("*")
            if p.is_file() and p.suffix != ".sqlite-wal" and p.suffix != ".sqlite-shm"
        }

    before = snapshot()
    main(["scrape", "verify", "--by-section"])
    assert snapshot() == before
