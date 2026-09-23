import hashlib
import os
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

from databricks_docs_vectordb.scraping import fetcher
from databricks_docs_vectordb.scraping.config import Settings
from databricks_docs_vectordb.scraping.fetcher import fetch_one, record_result
from databricks_docs_vectordb.scraping.manifest import Manifest, Page
from databricks_docs_vectordb.scraping.robots import RobotsRules
from databricks_docs_vectordb.scraping.urls import url_to_rel_path

FIXTURES = Path(__file__).parent.parent / "fixtures"
AWS = "https://docs.databricks.com/aws/en/"
DELTA = AWS + "delta/"
MERGE = AWS + "delta/merge"
CONTACT = "me@example.com"
ETAG = '"abc123"'
LAST_MODIFIED = "Fri, 18 Sep 2026 21:33:20 GMT"

PAGE = (
    b"<!doctype html><html><head><title data-rh=true>Delta Lake</title></head>"
    b"<body><article><h1>Delta Lake</h1><p>ACID tables.</p></article></body></html>"
)
NEW_PAGE = PAGE.replace(b"ACID tables.", b"ACID tables, updated.")
ERROR_PAGE = b"<html><head><title>Oops</title></head><body>Try again</body></html>"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(contact=CONTACT, data_dir=tmp_path / "data")


@pytest.fixture
def robots() -> RobotsRules:
    return RobotsRules((FIXTURES / "robots.txt").read_text())


@pytest.fixture
def site():
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
async def env(settings, robots, site):
    """A fetch helper with instant sleeps and no jitter (rng=1 -> full delay)."""
    sleeps: list[float] = []
    throttled = 0

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def throttle() -> None:
        nonlocal throttled
        throttled += 1

    async with httpx.AsyncClient(headers={"User-Agent": settings.user_agent}) as client:

        async def fetch(page: Page, **kwargs):
            kwargs.setdefault("sleep", sleep)
            kwargs.setdefault("rng", lambda: 1.0)
            return await fetch_one(client, page, settings, robots, **kwargs)

        yield SimpleNamespace(
            fetch=fetch,
            sleeps=sleeps,
            throttle=throttle,
            throttled=lambda: throttled,
            settings=settings,
            site=site,
        )


def page(url: str = DELTA, **overrides) -> Page:
    fields = dict(
        url=url,
        source="aws-en",
        rel_path=url_to_rel_path(url),
        status="pending",
        http_status=None,
        final_url=None,
        etag=None,
        last_modified=None,
        sha256=None,
        size_bytes=None,
        first_seen_at="2026-09-20T00:00:00Z",
        fetched_at=None,
        checked_at=None,
        attempts=0,
        error=None,
    )
    return Page(**{**fields, **overrides})


def stored(
    settings: Settings, url: str = DELTA, body: bytes = PAGE, **overrides
) -> Page:
    """A page that was fetched before: body on disk, validators in the row."""
    fields = dict(
        status="ok",
        etag=ETAG,
        last_modified=LAST_MODIFIED,
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
    )
    result = page(url, **{**fields, **overrides})
    dest = settings.raw_dir / result.rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return result


def ok(body: bytes = PAGE, **headers) -> httpx.Response:
    return httpx.Response(200, content=body, headers=headers)


def on_disk(settings: Settings, p: Page) -> Path:
    return settings.raw_dir / p.rel_path


# -- a plain 200 ---------------------------------------------------------


async def test_200_stores_the_body_and_reports_what_the_manifest_needs(env):
    route = env.site.get(DELTA).mock(
        return_value=ok(etag=ETAG, **{"last-modified": LAST_MODIFIED})
    )
    p = page()

    result = await env.fetch(p)

    assert result.outcome == "ok"
    assert (result.attempts, result.transient_failures) == (1, 0)
    assert (result.http_status, result.final_url) == (200, None)
    assert (result.etag, result.last_modified) == (ETAG, LAST_MODIFIED)
    assert result.sha256 == hashlib.sha256(PAGE).hexdigest()
    assert result.size_bytes == len(PAGE)
    assert on_disk(env.settings, p).read_bytes() == PAGE
    assert on_disk(env.settings, p) == env.settings.raw_dir / "aws/en/delta/index.html"
    assert route.call_count == 1


async def test_a_first_fetch_sends_no_conditional_headers(env):
    route = env.site.get(DELTA).mock(return_value=ok())
    await env.fetch(page())
    headers = route.calls[0].request.headers
    assert "if-none-match" not in headers
    assert "if-modified-since" not in headers


async def test_the_body_is_stored_byte_for_byte(env):
    body = PAGE + bytes(range(256))
    env.site.get(DELTA).mock(return_value=ok(body))
    p = page()
    await env.fetch(p)
    assert on_disk(env.settings, p).read_bytes() == body


async def test_requests_carry_the_user_agent_and_the_configured_timeouts(env):
    route = env.site.get(DELTA).mock(return_value=ok())
    await env.fetch(page())
    request = route.calls[0].request
    assert request.headers["user-agent"] == env.settings.user_agent
    timeout = request.extensions["timeout"]
    assert (timeout["connect"], timeout["read"]) == (10.0, 30.0)


async def test_no_temp_files_are_left_next_to_the_page(env):
    env.site.get(DELTA).mock(return_value=ok())
    p = page()
    await env.fetch(p)
    assert [f.name for f in on_disk(env.settings, p).parent.iterdir()] == ["index.html"]


# -- conditional requests ------------------------------------------------


async def test_validators_are_sent_when_a_stored_body_backs_them(env):
    route = env.site.get(DELTA).mock(return_value=httpx.Response(304))
    await env.fetch(stored(env.settings))
    headers = route.calls[0].request.headers
    assert headers["if-none-match"] == ETAG
    assert headers["if-modified-since"] == LAST_MODIFIED


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"last_modified": None}, {"if-none-match"}),
        ({"etag": None}, {"if-modified-since"}),
    ],
    ids=["etag-only", "last-modified-only"],
)
async def test_only_the_validators_the_row_has_are_sent(env, overrides, expected):
    route = env.site.get(DELTA).mock(return_value=httpx.Response(304))
    await env.fetch(stored(env.settings, **overrides))
    sent = {
        h
        for h in ("if-none-match", "if-modified-since")
        if h in route.calls[0].request.headers
    }
    assert sent == expected


async def test_304_keeps_the_stored_file_untouched(env):
    env.site.get(DELTA).mock(return_value=httpx.Response(304))
    p = stored(env.settings)
    path = on_disk(env.settings, p)
    os.utime(path, (1_000_000_000, 1_000_000_000))

    result = await env.fetch(p)

    assert result.outcome == "not_modified"
    assert result.http_status == 304
    assert result.attempts == 1
    assert path.read_bytes() == PAGE
    assert path.stat().st_mtime == 1_000_000_000  # not rewritten


async def test_a_304_can_carry_fresh_validators(env):
    env.site.get(DELTA).mock(
        return_value=httpx.Response(304, headers={"etag": '"rotated"'})
    )
    result = await env.fetch(stored(env.settings))
    assert result.etag == '"rotated"'


async def test_validators_are_not_sent_when_the_file_is_missing(env):
    # A 304 would otherwise mark the page ok with no body on disk.
    route = env.site.get(DELTA).mock(return_value=ok())
    p = page(etag=ETAG, last_modified=LAST_MODIFIED, sha256="ab" * 32, status="ok")
    result = await env.fetch(p)
    assert "if-none-match" not in route.calls[0].request.headers
    assert result.outcome == "ok"
    assert on_disk(env.settings, p).read_bytes() == PAGE


async def test_validators_are_not_sent_without_a_recorded_body_hash(env):
    route = env.site.get(DELTA).mock(return_value=ok())
    p = page(etag=ETAG, last_modified=LAST_MODIFIED)  # e.g. a stale pending row
    dest = on_disk(env.settings, p)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(
        b"unknown provenance"
    )  # a file exists, but the row can't vouch for it
    await env.fetch(p)
    assert "if-none-match" not in route.calls[0].request.headers


async def test_a_changed_page_replaces_the_stored_body(env):
    env.site.get(DELTA).mock(return_value=ok(NEW_PAGE, etag='"v2"'))
    p = stored(env.settings)

    result = await env.fetch(p)

    assert result.outcome == "ok"
    assert result.etag == '"v2"'
    assert result.sha256 == hashlib.sha256(NEW_PAGE).hexdigest()
    assert on_disk(env.settings, p).read_bytes() == NEW_PAGE


async def test_a_200_with_identical_content_is_not_a_change(env):
    # The server ignored the validators (or re-issued its ETag) but sent the
    # same bytes: keep the file, adopt the new validators.
    env.site.get(DELTA).mock(return_value=ok(PAGE, etag='"reissued"'))
    p = stored(env.settings)
    path = on_disk(env.settings, p)
    os.utime(path, (1_000_000_000, 1_000_000_000))

    result = await env.fetch(p)

    assert result.outcome == "not_modified"
    assert (result.http_status, result.etag) == (200, '"reissued"')
    assert path.stat().st_mtime == 1_000_000_000


async def test_a_304_to_an_unconditional_request_is_a_failure_not_a_success(env):
    env.site.get(DELTA).mock(return_value=httpx.Response(304))
    result = await env.fetch(page())
    assert result.outcome == "failed"
    assert "wasn't conditional" in result.error
    assert result.attempts == 1


# -- 404, 410 and other 4xx ----------------------------------------------


@pytest.mark.parametrize("status", [404, 410])
async def test_404_and_410_are_gone_and_not_retried(env, status):
    route = env.site.get(DELTA).mock(return_value=httpx.Response(status))
    result = await env.fetch(page())
    assert (result.outcome, result.http_status, result.attempts) == ("gone", status, 1)
    assert route.call_count == 1
    assert env.sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403, 405, 418])
async def test_other_4xx_fail_immediately(env, status):
    route = env.site.get(DELTA).mock(return_value=httpx.Response(status))
    result = await env.fetch(page())
    assert result.outcome == "failed"
    assert (result.http_status, result.error) == (status, f"HTTP {status}")
    assert (result.attempts, result.transient_failures) == (1, 0)
    assert route.call_count == 1


async def test_a_page_that_goes_missing_keeps_its_stored_body(env):
    env.site.get(DELTA).mock(return_value=httpx.Response(404))
    p = stored(env.settings)
    await env.fetch(p)
    assert on_disk(env.settings, p).read_bytes() == PAGE


# -- retries -------------------------------------------------------------


async def test_500_then_200_succeeds_after_one_backoff(env):
    route = env.site.get(DELTA).mock(side_effect=[httpx.Response(500), ok()])
    result = await env.fetch(page())
    assert result.outcome == "ok"
    assert (result.attempts, result.transient_failures) == (2, 1)
    assert route.call_count == 2
    assert env.sleeps == [1.0]


async def test_500_forever_gives_up_after_max_attempts_with_growing_pauses(env):
    route = env.site.get(DELTA).mock(return_value=httpx.Response(500))
    p = page()

    result = await env.fetch(p)

    assert result.outcome == "failed"
    assert result.attempts == route.call_count == 5
    assert result.transient_failures == 5
    assert result.http_status == 500
    assert "gave up after 5 attempts" in result.error
    assert "HTTP 500" in result.error
    assert env.sleeps == [1.0, 2.0, 4.0, 8.0]  # no pause after the last attempt
    assert not on_disk(env.settings, p).exists()


async def test_backoff_is_capped_at_backoff_max(env, settings, robots):
    capped = replace(settings, backoff_max=3.0)
    env.site.get(DELTA).mock(return_value=httpx.Response(503))
    sleeps: list[float] = []

    async def sleep(seconds):
        sleeps.append(seconds)

    async with httpx.AsyncClient() as client:
        await fetch_one(client, page(), capped, robots, sleep=sleep, rng=lambda: 1.0)

    assert sleeps == [1.0, 2.0, 3.0, 3.0]


@pytest.mark.parametrize(("unit", "factor"), [(0.0, 0.5), (0.5, 0.75), (1.0, 1.0)])
async def test_jitter_scales_the_delay_between_half_and_full(env, unit, factor):
    env.site.get(DELTA).mock(side_effect=[httpx.Response(500), ok()])
    await env.fetch(page(), rng=lambda: unit)
    assert env.sleeps == [pytest.approx(1.0 * factor)]


async def test_429_honors_retry_after_seconds(env):
    env.site.get(DELTA).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "30"}), ok()]
    )
    result = await env.fetch(page())
    assert result.outcome == "ok"
    assert env.sleeps == [30.0]  # longer than the 1s backoff
    assert result.transient_failures == 1


async def test_503_honors_a_retry_after_date(env):
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=60), usegmt=True)
    env.site.get(DELTA).mock(
        side_effect=[httpx.Response(503, headers={"Retry-After": when}), ok()]
    )
    await env.fetch(page())
    assert 55 <= env.sleeps[0] <= 60


async def test_retry_after_never_shortens_the_backoff(env):
    env.site.get(DELTA).mock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(429, headers={"Retry-After": "0"}),
            ok(),
        ]
    )
    await env.fetch(page())
    assert env.sleeps == [1.0, 2.0]


async def test_retry_after_is_ignored_on_other_5xx(env):
    env.site.get(DELTA).mock(
        side_effect=[httpx.Response(500, headers={"Retry-After": "120"}), ok()]
    )
    await env.fetch(page())
    assert env.sleeps == [1.0]


@pytest.mark.parametrize("value", ["soon", "-5", ""])
async def test_an_unparseable_retry_after_falls_back_to_backoff(env, value):
    env.site.get(DELTA).mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": value}), ok()]
    )
    await env.fetch(page())
    assert env.sleeps == [1.0]


async def test_a_retry_after_beyond_the_limit_fails_without_waiting(env):
    route = env.site.get(DELTA).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "3600"})
    )
    result = await env.fetch(page())
    assert result.outcome == "failed"
    assert "asked us to wait 3600s" in result.error
    assert (result.attempts, result.transient_failures) == (1, 1)
    assert route.call_count == 1
    assert env.sleeps == []


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("slow"),
        httpx.ReadTimeout("slow"),
        httpx.RemoteProtocolError("dropped"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_connection_problems_are_retried_and_then_succeed(env, error):
    route = env.site.get(DELTA).mock(side_effect=[error, ok()])
    result = await env.fetch(page())
    assert result.outcome == "ok"
    assert (result.attempts, result.transient_failures) == (2, 1)
    assert route.call_count == 2


async def test_connection_problems_forever_fail_with_the_error_type(env):
    env.site.get(DELTA).mock(side_effect=httpx.ReadTimeout("slow"))
    result = await env.fetch(page())
    assert result.outcome == "failed"
    assert "ReadTimeout" in result.error
    assert (result.attempts, result.transient_failures, result.http_status) == (
        5,
        5,
        None,
    )


async def test_a_different_max_attempts_is_respected(env, settings, robots):
    route = env.site.get(DELTA).mock(return_value=httpx.Response(500))

    async def sleep(_):
        pass

    async with httpx.AsyncClient() as client:
        result = await fetch_one(
            client, page(), replace(settings, max_attempts=2), robots, sleep=sleep
        )

    assert result.attempts == route.call_count == 2


# -- pages that don't look like docs ------------------------------------


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (ERROR_PAGE, "no <article>"),
        (b"", "no <article>"),
        (b"<html><body><article>hi</article></body></html>", "no <title>"),
    ],
    ids=["no-article", "empty", "no-title"],
)
async def test_a_200_that_isnt_a_docs_page_is_retried_then_failed_and_never_stored(
    env, body, message
):
    route = env.site.get(DELTA).mock(return_value=ok(body))
    p = page()

    result = await env.fetch(p)

    assert result.outcome == "failed"
    assert message in result.error
    assert result.attempts == route.call_count == 5
    assert result.transient_failures == 0  # soft errors don't trip the breaker
    assert not on_disk(env.settings, p).exists()


async def test_a_bad_page_followed_by_a_good_one_succeeds(env):
    env.site.get(DELTA).mock(side_effect=[ok(ERROR_PAGE), ok()])
    result = await env.fetch(page())
    assert (result.outcome, result.attempts) == ("ok", 2)


async def test_an_error_page_never_overwrites_a_good_stored_body(env):
    env.site.get(DELTA).mock(return_value=ok(ERROR_PAGE))
    p = stored(env.settings)
    result = await env.fetch(p)
    assert result.outcome == "failed"
    assert on_disk(env.settings, p).read_bytes() == PAGE


async def test_article_and_title_are_matched_case_insensitively_with_attributes(env):
    body = b"<HTML><TITLE lang=en>T</TITLE><ARTICLE class=x>hi</ARTICLE></HTML>"
    env.site.get(DELTA).mock(return_value=ok(body))
    assert (await env.fetch(page())).outcome == "ok"


# -- redirects -----------------------------------------------------------


def redirect(target: str, status: int = 301) -> httpx.Response:
    return httpx.Response(status, headers={"Location": target})


async def test_a_redirect_within_the_source_is_followed_and_recorded(env):
    old, new = AWS + "api/", AWS + "reference/api"
    env.site.get(old).mock(return_value=redirect(new))
    target = env.site.get(new).mock(return_value=ok())
    p = page(old)

    result = await env.fetch(p)

    assert result.outcome == "ok"
    assert result.final_url == new
    assert target.call_count == 1
    # stored under the row's own path, keyed by the sitemap URL
    assert on_disk(env.settings, p).read_bytes() == PAGE
    assert on_disk(env.settings, p).name == "index.html"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_all_redirect_statuses_are_followed(env, status):
    env.site.get(DELTA).mock(return_value=redirect(AWS + "moved", status))
    env.site.get(AWS + "moved").mock(return_value=ok())
    assert (await env.fetch(page())).outcome == "ok"


async def test_relative_redirect_targets_are_resolved(env):
    env.site.get(DELTA).mock(return_value=redirect("/aws/en/moved#section"))
    target = env.site.get(AWS + "moved").mock(return_value=ok())
    result = await env.fetch(page())
    assert result.final_url == AWS + "moved"  # fragment dropped
    assert target.call_count == 1


async def test_a_redirect_into_a_disallowed_path_is_refused_before_it_is_requested(env):
    archived = AWS + "archive/old-page"
    env.site.get(DELTA).mock(return_value=redirect(archived))
    target = env.site.get(archived).mock(return_value=ok())

    result = await env.fetch(page())

    assert result.outcome == "failed"
    assert "robots.txt disallows" in result.error
    assert target.call_count == 0
    assert result.attempts == 1  # permanent: not retried


@pytest.mark.parametrize(
    "target",
    [
        "https://example.com/aws/en/delta/",
        "https://docs.databricks.com/gcp/en/delta/",
        "http://docs.databricks.com/aws/en/delta/",
    ],
    ids=["other-host", "other-cloud", "plain-http"],
)
async def test_a_redirect_outside_the_configured_sources_is_refused(env, target):
    env.site.get(DELTA).mock(return_value=redirect(target))
    other = env.site.get(target).mock(return_value=ok())
    result = await env.fetch(page())
    assert result.outcome == "failed"
    assert "outside the configured sources" in result.error
    assert other.call_count == 0


async def test_redirect_loops_stop_at_max_redirects(env):
    route = env.site.get(DELTA).mock(return_value=redirect(DELTA))
    result = await env.fetch(page())
    assert result.outcome == "failed"
    assert "more than 5 redirects" in result.error
    assert route.call_count == 6  # the original request plus five hops
    assert result.attempts == 1


async def test_a_redirect_status_without_a_location_is_a_failure(env):
    env.site.get(DELTA).mock(return_value=httpx.Response(301))
    result = await env.fetch(page())
    assert (result.outcome, result.http_status, result.error) == (
        "failed",
        301,
        "HTTP 301",
    )


async def test_validators_go_to_the_url_that_produced_the_body(env):
    old, new = AWS + "api/", AWS + "reference/api"
    first = env.site.get(old).mock(return_value=redirect(new))
    second = env.site.get(new).mock(return_value=httpx.Response(304))
    p = stored(env.settings, old, final_url=new)

    result = await env.fetch(p)

    assert result.outcome == "not_modified"
    assert "if-none-match" not in first.calls[0].request.headers
    assert second.calls[0].request.headers["if-none-match"] == ETAG


async def test_a_redirect_to_a_new_place_does_not_reuse_old_validators(env):
    old = AWS + "api/"
    env.site.get(old).mock(return_value=redirect(AWS + "somewhere-else"))
    other = env.site.get(AWS + "somewhere-else").mock(return_value=ok(NEW_PAGE))
    p = stored(env.settings, old, final_url=AWS + "reference/api")

    result = await env.fetch(p)

    assert result.outcome == "ok"
    assert "if-none-match" not in other.calls[0].request.headers


# -- throttling ----------------------------------------------------------


async def test_the_throttle_is_awaited_before_every_request_including_retries_and_hops(
    env,
):
    env.site.get(DELTA).mock(side_effect=[httpx.Response(500), redirect(AWS + "moved")])
    env.site.get(AWS + "moved").mock(return_value=ok())

    await env.fetch(page(), throttle=env.throttle)

    assert env.throttled() == 3  # 500, then the redirect, then the page


# -- safety --------------------------------------------------------------


async def test_a_path_that_escapes_the_raw_directory_is_refused_without_a_request(env):
    route = env.site.get(DELTA).mock(return_value=ok())
    result = await env.fetch(page(rel_path="../../outside.html"))
    assert result.outcome == "failed"
    assert "unsafe file path" in result.error
    assert route.call_count == 0
    assert not (env.settings.data_dir.parent / "outside.html").exists()


async def test_a_failure_to_write_raises_instead_of_being_recorded_as_a_page_failure(
    env, monkeypatch
):
    def full_disk(path, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(fetcher, "write_atomic", full_disk)
    env.site.get(DELTA).mock(return_value=ok())
    with pytest.raises(OSError, match="No space left"):
        await env.fetch(page())


async def test_the_contact_address_never_appears_in_a_result(env):
    env.site.get(DELTA).mock(return_value=httpx.Response(500))
    result = await env.fetch(page())
    assert CONTACT not in repr(result)


# -- record_result: results become manifest updates ----------------------


@pytest.fixture
def manifest(settings):
    with Manifest(settings.manifest_path) as m:
        yield m


async def test_recording_a_fetched_page(env, manifest):
    env.site.get(DELTA).mock(
        return_value=ok(etag=ETAG, **{"last-modified": LAST_MODIFIED})
    )
    manifest.add_pending("aws-en", [DELTA])
    row = manifest.get(DELTA)

    record_result(manifest, await env.fetch(row), now="2026-09-21T01:00:00Z")

    saved = manifest.get(DELTA)
    assert (saved.status, saved.http_status) == ("ok", 200)
    assert saved.sha256 == hashlib.sha256(PAGE).hexdigest()
    assert (saved.etag, saved.last_modified) == (ETAG, LAST_MODIFIED)
    assert (saved.fetched_at, saved.checked_at) == ("2026-09-21T01:00:00Z",) * 2
    assert manifest.next_batch() == []


async def test_a_refresh_that_gets_a_304_only_advances_checked_at(env, manifest):
    env.site.get(DELTA).mock(return_value=ok(etag=ETAG))
    manifest.add_pending("aws-en", [DELTA])
    record_result(
        manifest, await env.fetch(manifest.get(DELTA)), now="2026-09-21T01:00:00Z"
    )

    env.site.get(DELTA).mock(return_value=httpx.Response(304))
    (row,) = manifest.next_batch(refresh=True)
    record_result(manifest, await env.fetch(row), now="2026-09-21T02:00:00Z")

    saved = manifest.get(DELTA)
    assert saved.status == "ok"
    assert saved.http_status == 304
    assert (saved.fetched_at, saved.checked_at) == (
        "2026-09-21T01:00:00Z",
        "2026-09-21T02:00:00Z",
    )


async def test_recording_gone_and_failed_pages(env, manifest):
    env.site.get(DELTA).mock(return_value=httpx.Response(404))
    env.site.get(MERGE).mock(return_value=httpx.Response(500))
    manifest.add_pending("aws-en", [DELTA, MERGE])
    for row in manifest.next_batch():
        record_result(manifest, await env.fetch(row))

    assert manifest.counts() == Counter(gone=1, failed=1)
    failed = manifest.get(MERGE)
    assert (failed.attempts, failed.http_status) == (5, 500)
    assert "gave up after 5 attempts" in failed.error


async def test_a_redirected_page_records_its_final_url(env, manifest):
    old, new = AWS + "api/", AWS + "reference/api"
    env.site.get(old).mock(return_value=redirect(new))
    env.site.get(new).mock(return_value=ok())
    manifest.add_pending("aws-en", [old])
    record_result(manifest, await env.fetch(manifest.get(old)))
    assert manifest.get(old).final_url == new
