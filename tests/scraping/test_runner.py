import asyncio
import hashlib
import logging
import os
import signal
import time
from collections import Counter
from pathlib import Path

import httpx
import pytest

from databricks_docs_vectordb.scraping.config import Settings
from databricks_docs_vectordb.scraping.fetcher import Result
from databricks_docs_vectordb.scraping.manifest import Manifest
from databricks_docs_vectordb.scraping.robots import RobotsRules
from databricks_docs_vectordb.scraping.runner import (
    Runner,
    RunStopped,
    TokenBucket,
    make_client,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"
AWS = "https://docs.databricks.com/aws/en/"
CONTACT = "me@example.com"
PAGE = (
    b"<html><head><title>T</title></head><body><article>hello</article></body></html>"
)


def urls(n: int) -> list[str]:
    return [AWS + f"p{i:03d}" for i in range(n)]


def make_settings(tmp_path: Path, **overrides) -> Settings:
    fields = dict(
        contact=CONTACT,
        data_dir=tmp_path / "data",
        rate=1000.0,
        concurrency=2,
        max_attempts=1,
        breaker_cooldown=0.05,
        connect_timeout=5.0,
        read_timeout=5.0,
    )
    return Settings(**{**fields, **overrides})


@pytest.fixture
def robots() -> RobotsRules:
    return RobotsRules((FIXTURES / "robots.txt").read_text())


@pytest.fixture
def make_manifest(tmp_path):
    opened = []

    def make(n: int, settings: Settings | None = None) -> Manifest:
        manifest = Manifest((settings or make_settings(tmp_path)).manifest_path)
        manifest.add_pending("aws-en", urls(n))
        opened.append(manifest)
        return manifest

    yield make
    for manifest in opened:
        manifest.close()


# -- a real local server, reached through the real client ---------------


class FakeSite:
    """Serves HTTP on localhost and records what it saw."""

    def __init__(self, respond=None, delay: float = 0.0) -> None:
        self.respond = respond or (lambda n, path: 200)  # -> status or "hang"
        self.delay = delay
        self.times: list[float] = []
        self.paths: list[str] = []
        self.active = 0
        self.peak = 0
        self._tasks: set[asyncio.Task] = set()

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._server.wait_closed()

    async def _handle(self, reader, writer):
        self._tasks.add(asyncio.current_task())
        try:
            line = await reader.readline()
            while await reader.readline() not in (b"\r\n", b"\n", b""):
                pass
            path = line.split()[1].decode()
            self.times.append(time.monotonic())
            self.paths.append(path)
            number = len(self.paths)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                if self.delay:
                    await asyncio.sleep(self.delay)
                verdict = self.respond(number, path)
                if verdict == "hang":
                    await asyncio.sleep(60)
                body = PAGE if verdict == 200 else b"error"
                head = (
                    f"HTTP/1.1 {verdict} X\r\nContent-Length: {len(body)}\r\n"
                    "Content-Type: text/html\r\nConnection: close\r\n\r\n"
                )
                writer.write(head.encode() + body)
                await writer.drain()
            finally:
                self.active -= 1
        finally:
            writer.close()

    def client(self, settings: Settings) -> httpx.AsyncClient:
        return make_client(settings, transport=_RewriteTransport(self.port))

    def gaps(self, at_least: float) -> int:
        times = sorted(self.times)
        return sum(b - a >= at_least for a, b in zip(times, times[1:]))


class _RewriteTransport(httpx.AsyncBaseTransport):
    """Send requests for docs.databricks.com to the local server instead."""

    def __init__(self, port: int) -> None:
        self._port = port
        self._inner = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request):
        request.url = request.url.copy_with(
            scheme="http", host="127.0.0.1", port=self._port
        )
        return await self._inner.handle_async_request(request)

    async def aclose(self):
        await self._inner.aclose()


# -- scripted fetches: no sockets, for the breaker and shutdown logic ----


def result(
    url: str, outcome: str = "ok", *, transient: int = 0, attempts: int = 1
) -> Result:
    if outcome == "ok":
        return Result(
            url, "ok", attempts, http_status=200, sha256="ab" * 32, size_bytes=1
        )
    if outcome == "not_modified":
        return Result(url, outcome, attempts, http_status=304)
    if outcome == "gone":
        return Result(url, outcome, attempts, http_status=404)
    return Result(
        url,
        "failed",
        attempts,
        http_status=500 if transient else 403,
        error="boom",
        transient_failures=transient,
    )


def scripted(outcomes, *, request: bool = True):
    """A fetch function: outcomes maps url -> Result (or a callable returning one)."""
    calls: list[str] = []
    passed: list[float] = []

    async def fetch(client, page, settings, robots, *, throttle, sleep, rng=None):
        calls.append(page.url)
        if request:
            await throttle()
            passed.append(time.monotonic())
        outcome = (
            outcomes[page.url] if isinstance(outcomes, dict) else outcomes(page.url)
        )
        return outcome() if callable(outcome) else outcome

    fetch.calls = calls
    fetch.passed = passed
    return fetch


def runner_with(manifest, settings, robots, fetch, **kwargs) -> Runner:
    return Runner(object(), settings, manifest, robots, fetch=fetch, **kwargs)


# =========================================================================
# token bucket
# =========================================================================


class FakeTime:
    def __init__(self):
        self.now = 100.0
        self.slept: list[float] = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


async def test_bucket_spaces_requests_at_the_rate():
    t = FakeTime()
    bucket = TokenBucket(2.0, clock=t.clock, sleep=t.sleep)
    for _ in range(4):
        await bucket.acquire()
    assert t.slept == [pytest.approx(0.5)] * 3  # first is free, then one per 1/rate


async def test_bucket_does_not_bank_idle_time_beyond_its_capacity():
    t = FakeTime()
    bucket = TokenBucket(2.0, capacity=1, clock=t.clock, sleep=t.sleep)
    await bucket.acquire()
    t.now += 1000  # a long idle spell
    await bucket.acquire()
    await bucket.acquire()
    assert t.slept == [pytest.approx(0.5)]  # only one token was banked


async def test_bucket_allows_a_burst_up_to_its_capacity():
    t = FakeTime()
    bucket = TokenBucket(1.0, capacity=3, clock=t.clock, sleep=t.sleep)
    for _ in range(3):
        await bucket.acquire()
    assert t.slept == []
    await bucket.acquire()
    assert t.slept == [pytest.approx(1.0)]


@pytest.mark.parametrize(
    "kwargs", [{"rate": 0}, {"rate": -1}, {"rate": 1, "capacity": 0.5}]
)
def test_bucket_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        TokenBucket(**kwargs)


async def test_bucket_serves_waiters_in_order():
    bucket = TokenBucket(200.0)
    order: list[int] = []

    async def take(i):
        await bucket.acquire()
        order.append(i)

    await asyncio.gather(*(take(i) for i in range(8)))
    assert order == list(range(8))


# =========================================================================
# against a real local server
# =========================================================================


async def test_the_rate_limit_is_actually_enforced(tmp_path, robots, make_manifest):
    settings = make_settings(tmp_path, rate=5.0, concurrency=4)
    manifest = make_manifest(20, settings)
    async with FakeSite() as site:
        async with site.client(settings) as client:
            summary = await Runner(client, settings, manifest, robots).run(
                handle_signals=False
            )

    assert summary.outcomes == Counter(ok=20)
    assert summary.status == "completed"
    # 20 requests at 5/s: the first is immediate, then 19 gaps of 0.2s
    assert summary.elapsed >= 3.7
    assert max(site.times) - min(site.times) >= 3.6
    # and never a burst: no 1-second window holds more than rate + 1 requests
    times = sorted(site.times)
    busiest = max(sum(t <= start < t + 1.0 for start in times) for t in times)
    assert busiest <= 6


async def test_concurrency_never_exceeds_the_limit_but_is_used(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, concurrency=3)
    manifest = make_manifest(15, settings)
    async with FakeSite(delay=0.1) as site:
        async with site.client(settings) as client:
            summary = await Runner(client, settings, manifest, robots).run(
                handle_signals=False
            )

    assert summary.outcomes == Counter(ok=15)
    assert site.peak == 3


async def test_pages_are_written_and_recorded(tmp_path, robots, make_manifest):
    settings = make_settings(tmp_path)
    manifest = make_manifest(6, settings)
    async with FakeSite() as site:
        async with site.client(settings) as client:
            await Runner(client, settings, manifest, robots).run(handle_signals=False)

    assert manifest.counts() == Counter(ok=6)
    for url in urls(6):
        row = manifest.get(url)
        body = (settings.raw_dir / row.rel_path).read_bytes()
        assert body == PAGE
        assert hashlib.sha256(body).hexdigest() == row.sha256


async def test_a_server_that_never_answers_is_cut_off_by_the_read_timeout(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, read_timeout=0.3)
    manifest = make_manifest(3, settings)
    hang_on = urls(3)[0]
    async with FakeSite(
        respond=lambda n, path: "hang" if path.endswith(hang_on[-4:]) else 200
    ) as site:
        async with site.client(settings) as client:
            started = time.monotonic()
            summary = await Runner(client, settings, manifest, robots).run(
                handle_signals=False
            )
            elapsed = time.monotonic() - started

    assert elapsed < 3  # not the 60s the server would have taken
    assert summary.status == "completed"
    assert summary.outcomes == Counter(ok=2, failed=1)
    stuck = manifest.get(hang_on)
    assert stuck.status == "failed"
    assert "ReadTimeout" in stuck.error


async def test_ten_500s_trip_the_breaker_and_the_third_trip_ends_the_run(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, breaker_cooldown=0.4, breaker_threshold=10)
    manifest = make_manifest(100, settings)
    async with FakeSite(respond=lambda n, path: 500) as site:
        async with site.client(settings) as client:
            summary = await Runner(client, settings, manifest, robots).run(
                handle_signals=False
            )

    assert summary.status == "breaker_tripped"
    assert summary.breaker_trips == 3
    counts = manifest.counts()
    assert counts["failed"] >= 28
    assert counts["pending"] >= 60  # the rest were never tried
    assert counts["failed"] + counts["pending"] == 100
    assert summary.pending_left == counts["pending"]
    # trips 1 and 2 each paused every worker for the cool-down; trip 3 just stopped
    assert site.gaps(at_least=0.35) == 2


async def test_the_run_finishes_normally_if_the_server_recovers_after_a_trip(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, concurrency=1, breaker_cooldown=0.3)
    manifest = make_manifest(40, settings)
    async with FakeSite(respond=lambda n, path: 500 if n <= 10 else 200) as site:
        async with site.client(settings) as client:
            summary = await Runner(client, settings, manifest, robots).run(
                handle_signals=False
            )

    assert summary.status == "completed"
    assert summary.breaker_trips == 1
    assert summary.outcomes == Counter(ok=30, failed=10)
    assert manifest.counts() == Counter(ok=30, failed=10)
    assert site.gaps(at_least=0.25) == 1  # the one cool-down pause


async def test_sigint_stops_cleanly_and_a_rerun_resumes_without_refetching(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, concurrency=4)
    manifest = make_manifest(120, settings)
    async with FakeSite(delay=0.03) as site:
        async with site.client(settings) as client:

            async def interrupt():
                while len(site.paths) < 25:
                    await asyncio.sleep(0.005)
                os.kill(os.getpid(), signal.SIGINT)

            interrupter = asyncio.create_task(interrupt())
            first = await Runner(client, settings, manifest, robots).run()
            await interrupter

            assert first.status == "interrupted"
            assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
            counts = manifest.counts()
            assert 0 < counts["ok"] < 120
            assert counts["failed"] == 0
            assert counts["ok"] + counts["pending"] == 120
            # everything recorded as ok really is on disk, intact, with no temp files
            for row in manifest.next_batch(refresh=True):
                if row.status == "ok":
                    body = (settings.raw_dir / row.rel_path).read_bytes()
                    assert hashlib.sha256(body).hexdigest() == row.sha256
            assert not [p for p in settings.raw_dir.rglob(".*") if p.is_file()]

            second = await Runner(client, settings, manifest, robots).run(
                handle_signals=False
            )

    assert second.status == "completed"
    assert second.selected == counts["pending"]
    assert manifest.counts() == Counter(ok=120)
    # no page was ever requested twice: finished pages were not redone
    assert max(Counter(site.paths).values()) == 1
    assert len(site.paths) == 120


async def test_a_second_sigint_aborts_immediately(tmp_path, robots, make_manifest):
    settings = make_settings(tmp_path, read_timeout=30.0)
    manifest = make_manifest(10, settings)
    async with FakeSite(respond=lambda n, path: "hang") as site:
        async with site.client(settings) as client:

            async def interrupt_twice():
                while len(site.paths) < 2:
                    await asyncio.sleep(0.005)
                os.kill(os.getpid(), signal.SIGINT)
                await asyncio.sleep(0.1)
                os.kill(os.getpid(), signal.SIGINT)

            interrupter = asyncio.create_task(interrupt_twice())
            started = time.monotonic()
            summary = await Runner(client, settings, manifest, robots).run()
            await interrupter
            elapsed = time.monotonic() - started

    assert elapsed < 5  # not the 30s read timeout
    assert summary.status == "interrupted"
    assert manifest.counts() == Counter(pending=10)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# =========================================================================
# scripted fetches: circuit breaker logic
# =========================================================================


async def test_results_are_recorded_and_summarised(tmp_path, robots, make_manifest):
    settings = make_settings(tmp_path)
    manifest = make_manifest(4, settings)
    u = urls(4)
    fetch = scripted(
        {
            u[0]: result(u[0], "ok"),
            u[1]: result(u[1], "gone"),
            u[2]: result(u[2], "failed", attempts=5),
            u[3]: result(u[3], "ok", attempts=2),
        }
    )

    summary = await runner_with(manifest, settings, robots, fetch).run(
        handle_signals=False
    )

    assert summary.status == "completed"
    assert (summary.selected, summary.processed) == (4, 4)
    assert summary.outcomes == Counter(ok=2, gone=1, failed=1)
    assert summary.requests == 1 + 1 + 5 + 2  # attempts, retries included
    assert manifest.counts() == Counter(ok=2, gone=1, failed=1)
    assert summary.pending_left == 0


async def test_an_empty_batch_completes_at_once(tmp_path, robots, make_manifest):
    settings = make_settings(tmp_path)
    manifest = make_manifest(0, settings)
    fetch = scripted({})
    summary = await runner_with(manifest, settings, robots, fetch).run(
        handle_signals=False
    )
    assert (summary.status, summary.selected, summary.processed) == ("completed", 0, 0)
    assert fetch.calls == []


async def test_limit_and_refresh_choose_the_pages(tmp_path, robots, make_manifest):
    settings = make_settings(tmp_path)
    manifest = make_manifest(6, settings)
    u = urls(6)
    manifest.record_ok(u[0], http_status=200, sha256="ab" * 32, size_bytes=1)
    manifest.record_ok(u[1], http_status=200, sha256="ab" * 32, size_bytes=1)

    fetch = scripted(lambda url: result(url))
    await runner_with(manifest, settings, robots, fetch).run(
        limit=3, handle_signals=False
    )
    assert sorted(fetch.calls) == u[2:5]  # pending only, and only 3

    fetch = scripted(lambda url: result(url, "not_modified"))
    await runner_with(manifest, settings, robots, fetch).run(
        refresh=True, handle_signals=False
    )
    assert sorted(fetch.calls) == u  # everything ok or still pending


async def test_explicit_pages_replace_the_manifests_own_selection(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path)
    manifest = make_manifest(6, settings)
    u = urls(6)
    chosen = [manifest.get(u[4]), manifest.get(u[1])]

    fetch = scripted(lambda url: result(url))
    summary = await runner_with(manifest, settings, robots, fetch).run(
        pages=chosen,
        limit=1,
        handle_signals=False,  # limit is ignored when pages are given
    )

    assert sorted(fetch.calls) == sorted([u[1], u[4]])
    assert (summary.selected, summary.processed) == (2, 2)
    assert manifest.counts() == Counter(ok=2, pending=4)


async def test_an_empty_explicit_list_fetches_nothing_even_with_pages_pending(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path)
    manifest = make_manifest(3, settings)
    fetch = scripted(lambda url: result(url))
    summary = await runner_with(manifest, settings, robots, fetch).run(
        pages=[], handle_signals=False
    )
    assert fetch.calls == [] and summary.selected == 0
    assert manifest.counts() == Counter(pending=3)


async def test_stale_before_is_passed_through_to_the_selection(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path)
    manifest = make_manifest(2, settings)
    u = urls(2)
    manifest.record_ok(
        u[0],
        http_status=200,
        sha256="ab" * 32,
        size_bytes=1,
        now="2026-09-21T01:00:00Z",
    )
    manifest.record_ok(
        u[1],
        http_status=200,
        sha256="ab" * 32,
        size_bytes=1,
        now="2026-09-21T09:00:00Z",
    )
    fetch = scripted(lambda url: result(url, "not_modified"))
    await runner_with(manifest, settings, robots, fetch).run(
        refresh=True, stale_before="2026-09-21T05:00:00Z", handle_signals=False
    )
    assert fetch.calls == [u[0]]


async def test_failures_from_different_pages_add_up_to_a_trip(
    tmp_path, robots, make_manifest
):
    settings = make_settings(
        tmp_path, concurrency=1, breaker_threshold=10, max_attempts=5
    )
    manifest = make_manifest(10, settings)
    # each page exhausts 5 attempts (5 transient failures); two pages = 10
    fetch = scripted(lambda url: result(url, "failed", transient=5, attempts=5))
    summary = await runner_with(manifest, settings, robots, fetch).run(
        handle_signals=False
    )
    assert summary.breaker_trips == 3
    assert len(fetch.calls) == 6  # trips after pages 2, 4 and 6
    assert summary.status == "breaker_tripped"


async def test_a_success_resets_the_failure_count(tmp_path, robots, make_manifest):
    settings = make_settings(tmp_path, concurrency=1, breaker_threshold=10)
    manifest = make_manifest(8, settings)
    u = urls(8)
    pattern = ["failed", "ok"] * 4  # 5 transient failures, a success, and so on
    fetch = scripted(
        {
            url: result(url, kind, transient=5 if kind == "failed" else 0)
            for url, kind in zip(u, pattern)
        }
    )
    summary = await runner_with(manifest, settings, robots, fetch).run(
        handle_signals=False
    )
    assert summary.breaker_trips == 0
    assert summary.status == "completed"


async def test_a_page_that_succeeded_after_retries_also_resets_the_count(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, concurrency=1, breaker_threshold=10)
    manifest = make_manifest(6, settings)
    # 3 transient failures then success, over and over: never 10 in a row
    fetch = scripted(lambda url: result(url, "ok", transient=3, attempts=4))
    summary = await runner_with(manifest, settings, robots, fetch).run(
        handle_signals=False
    )
    assert summary.breaker_trips == 0


async def test_a_permanent_failure_means_the_server_is_answering(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, concurrency=1, breaker_threshold=10)
    manifest = make_manifest(6, settings)
    u = urls(6)
    kinds = [
        ("failed", 5),
        ("failed", 0),
        ("failed", 5),
        ("failed", 0),
        ("failed", 5),
        ("failed", 0),
    ]
    fetch = scripted(
        {url: result(url, k, transient=t) for url, (k, t) in zip(u, kinds)}
    )
    summary = await runner_with(manifest, settings, robots, fetch).run(
        handle_signals=False
    )
    assert summary.breaker_trips == 0  # a 403 between the 500s resets the run


async def test_a_trip_pauses_every_request_for_the_cooldown(
    tmp_path, robots, make_manifest
):
    settings = make_settings(
        tmp_path,
        concurrency=1,
        breaker_threshold=2,
        breaker_cooldown=0.3,
        breaker_max_trips=3,
    )
    manifest = make_manifest(5, settings)
    fetch = scripted(lambda url: result(url, "failed", transient=2))
    summary = await runner_with(manifest, settings, robots, fetch).run(
        handle_signals=False
    )

    assert summary.breaker_trips == 3
    assert len(fetch.passed) == 3
    gaps = [b - a for a, b in zip(fetch.passed, fetch.passed[1:])]
    assert all(gap >= 0.25 for gap in gaps)  # each trip paused the next request


async def test_the_pause_applies_to_retries_inside_a_page_not_just_new_pages(
    tmp_path, robots, make_manifest
):
    settings = make_settings(
        tmp_path,
        concurrency=2,
        breaker_threshold=2,
        breaker_cooldown=0.3,
        breaker_max_trips=5,
    )
    manifest = make_manifest(2, settings)
    u = urls(2)
    stamps: list[float] = []

    async def fetch(client, page, settings, robots, *, throttle, sleep, rng=None):
        if page.url == u[0]:
            await throttle()
            return result(u[0], "failed", transient=2)  # trips the breaker
        await asyncio.sleep(0.1)  # page 1 is mid-fetch when the breaker trips
        await throttle()  # ...and its retry must wait out the cool-down
        stamps.append(time.monotonic())
        return result(u[1])

    started = time.monotonic()
    await runner_with(manifest, settings, robots, fetch).run(handle_signals=False)
    assert stamps[0] - started >= 0.28  # tripped at ~0s, cool-down 0.3s


# =========================================================================
# scripted fetches: shutdown
# =========================================================================


async def test_stop_cuts_a_backoff_sleep_short_and_leaves_the_page_pending(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, concurrency=1)
    manifest = make_manifest(3, settings)

    async def fetch(client, page, settings, robots, *, throttle, sleep, rng=None):
        await throttle()
        await sleep(60)  # a long backoff after a failed request
        await throttle()  # the retry
        return result(page.url)

    runner = runner_with(manifest, settings, robots, fetch)
    asyncio.get_running_loop().call_later(0.1, runner.request_stop)
    started = time.monotonic()
    summary = await runner.run(handle_signals=False)

    assert time.monotonic() - started < 3
    assert summary.status == "interrupted"
    assert summary.processed == 0
    assert manifest.counts() == Counter(pending=3)  # untouched, not failed


async def test_stop_during_the_cooldown_does_not_wait_it_out(
    tmp_path, robots, make_manifest
):
    settings = make_settings(
        tmp_path,
        concurrency=1,
        breaker_threshold=2,
        breaker_cooldown=600,
        breaker_max_trips=5,
    )
    manifest = make_manifest(4, settings)
    fetch = scripted(lambda url: result(url, "failed", transient=2))
    runner = runner_with(manifest, settings, robots, fetch)
    asyncio.get_running_loop().call_later(0.15, runner.request_stop)

    started = time.monotonic()
    summary = await runner.run(handle_signals=False)

    assert time.monotonic() - started < 3  # not 600s
    assert summary.status == "interrupted"
    assert summary.breaker_trips == 1
    assert manifest.counts() == Counter(failed=1, pending=3)


async def test_the_first_stop_reason_wins(tmp_path, robots, make_manifest):
    settings = make_settings(
        tmp_path, concurrency=1, breaker_threshold=2, breaker_max_trips=1
    )
    manifest = make_manifest(4, settings)
    fetch = scripted(lambda url: result(url, "failed", transient=2))
    runner = runner_with(manifest, settings, robots, fetch)
    summary = await runner.run(handle_signals=False)
    runner.request_stop("interrupted")  # too late to change the outcome
    assert summary.status == "breaker_tripped"


async def test_stop_ends_the_wait_for_a_rate_limit_token_at_a_slow_rate(
    tmp_path, robots, make_manifest
):
    # At 0.2 requests/s a token takes 5s. A Ctrl-C must not have to wait for it.
    settings = make_settings(tmp_path, rate=0.2, concurrency=2)
    manifest = make_manifest(6, settings)
    fetch = scripted(lambda url: result(url))
    runner = runner_with(manifest, settings, robots, fetch)
    asyncio.get_running_loop().call_later(0.2, runner.request_stop)

    started = time.monotonic()
    summary = await runner.run(handle_signals=False)

    assert time.monotonic() - started < 2
    assert summary.status == "interrupted"
    assert summary.processed == 1  # only the first page got a token
    assert manifest.counts() == Counter(ok=1, pending=5)


async def test_a_later_interrupt_does_not_change_why_the_breaker_stopped_the_run(
    tmp_path, robots, make_manifest
):
    settings = make_settings(
        tmp_path, concurrency=2, breaker_threshold=1, breaker_max_trips=1
    )
    manifest = make_manifest(2, settings)
    u = urls(2)
    box: list[Runner] = []

    async def fetch(client, page, settings, robots, *, throttle, sleep, rng=None):
        await throttle()
        if page.url == u[0]:
            await asyncio.sleep(0.05)  # let the other worker get through the throttle
            return result(u[0], "failed", transient=1)  # trips the breaker
        await asyncio.sleep(0.15)  # still in flight when that happens
        box[0].request_stop("interrupted")  # a Ctrl-C arrives afterwards
        return result(u[1])

    runner = runner_with(manifest, settings, robots, fetch)
    box.append(runner)
    summary = await runner.run(handle_signals=False)

    assert summary.status == "breaker_tripped"
    assert summary.outcomes == Counter(failed=1, ok=1)


async def test_stop_between_pages_finishes_the_one_in_flight(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, concurrency=1)
    manifest = make_manifest(5, settings)
    runner_box = []

    async def fetch(client, page, settings, robots, *, throttle, sleep, rng=None):
        await throttle()
        if page.url == urls(5)[1]:
            runner_box[0].request_stop()  # stop requested mid-page
        return result(page.url)

    runner = runner_with(manifest, settings, robots, fetch)
    runner_box.append(runner)
    summary = await runner.run(handle_signals=False)

    assert summary.status == "interrupted"
    assert manifest.counts() == Counter(ok=2, pending=3)  # page 2 was still recorded


async def test_an_unexpected_error_stops_the_run_and_cancels_the_other_workers(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path, concurrency=3)
    manifest = make_manifest(9, settings)
    started: list[str] = []
    cancelled: list[str] = []

    async def fetch(client, page, settings, robots, *, throttle, sleep, rng=None):
        started.append(page.url)
        if page.url == urls(9)[0]:
            await asyncio.sleep(0.05)
            raise OSError(28, "No space left on device")
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(page.url)
            raise
        return result(page.url)

    with pytest.raises(OSError, match="No space left"):
        await runner_with(manifest, settings, robots, fetch).run(handle_signals=False)

    assert len(cancelled) == 2  # the two workers that were mid-fetch
    assert manifest.counts()["pending"] == 9  # nothing half-recorded
    assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]


async def test_throttle_raises_runstopped_only_after_a_stop(
    tmp_path, robots, make_manifest
):
    settings = make_settings(tmp_path)
    manifest = make_manifest(0, settings)
    runner = runner_with(manifest, settings, robots, scripted({}))
    await runner._throttle()  # fine while running
    runner.request_stop()
    with pytest.raises(RunStopped):
        await runner._throttle()


# =========================================================================
# logging
# =========================================================================


async def test_progress_lines_and_a_final_summary_are_logged(
    tmp_path, robots, make_manifest, caplog
):
    settings = make_settings(tmp_path, progress_every=5, concurrency=1)
    manifest = make_manifest(12, settings)
    fetch = scripted(lambda url: result(url))
    with caplog.at_level(
        logging.INFO, logger="databricks_docs_vectordb.scraping.runner"
    ):
        await runner_with(manifest, settings, robots, fetch).run(handle_signals=False)

    messages = [r.getMessage() for r in caplog.records]
    progress = [m for m in messages if m.startswith("progress:")]
    assert len(progress) == 2
    assert "5/12 done (5 ok)" in progress[0]
    assert "10/12 done (10 ok)" in progress[1]
    assert "pages/s" in progress[0]
    assert any(m.startswith("completed: 12/12 pages") for m in messages)
    assert not any(CONTACT in m for m in messages)


async def test_the_breaker_logs_what_it_did(tmp_path, robots, make_manifest, caplog):
    settings = make_settings(
        tmp_path, concurrency=1, breaker_threshold=2, breaker_max_trips=2
    )
    manifest = make_manifest(4, settings)
    fetch = scripted(lambda url: result(url, "failed", transient=2))
    with caplog.at_level(
        logging.WARNING, logger="databricks_docs_vectordb.scraping.runner"
    ):
        await runner_with(manifest, settings, robots, fetch).run(handle_signals=False)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "circuit breaker tripped (2 consecutive failures, trip 1 of 2)" in text
    assert "tripped 2 times: stopping the run" in text
