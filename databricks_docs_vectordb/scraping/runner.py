import asyncio
import logging
import signal
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

import httpx

from databricks_docs_vectordb.scraping.config import Settings
from databricks_docs_vectordb.scraping.fetcher import Result, fetch_one, record_result
from databricks_docs_vectordb.scraping.manifest import Manifest, Page
from databricks_docs_vectordb.scraping.robots import RobotsRules

logger = logging.getLogger(__name__)

RunStatus = Literal["completed", "interrupted", "breaker_tripped"]
FetchFn = Callable[..., Awaitable[Result]]


class RunStopped(Exception):
    """Raised inside a fetch when the run is stopping; the page is left as it was."""


class TokenBucket:
    """Shared rate limiter: at most `rate` request starts per second.

    With the default capacity of one there are no bursts, so requests are
    evenly spaced. Waiters are served in order. `clock` and `sleep` are
    injectable so tests don't need real time.

    `acquire(amount)` takes more than one token at a time, for limiting
    something other than request count (embedding/runner.py's
    tokens-per-minute budget); `amount` can't exceed `capacity`, or it
    could never be satisfied.
    """

    def __init__(
        self,
        rate: float,
        capacity: float = 1.0,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if rate <= 0 or capacity < 1:
            raise ValueError("rate must be > 0 and capacity >= 1")
        self._rate = rate
        self._capacity = capacity
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._tokens = capacity
        self._updated = self._clock()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> None:
        if not 0 < amount <= self._capacity:
            raise ValueError(
                f"amount must be > 0 and <= capacity ({self._capacity}), got {amount}"
            )
        async with self._lock:  # holding it while we wait keeps waiters in order
            while True:
                now = self._clock()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                await self._sleep((amount - self._tokens) / self._rate)


@dataclass(frozen=True)
class RunSummary:
    status: RunStatus
    selected: int  # pages the run set out to fetch
    outcomes: Counter[str]  # ok / not_modified / gone / failed
    requests: int  # attempts across all pages, retries included
    breaker_trips: int
    elapsed: float
    pending_left: int  # pending rows in the manifest afterwards

    @property
    def processed(self) -> int:
        return sum(self.outcomes.values())


def make_client(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """An HTTP client carrying our User-Agent. Timeouts and redirect handling
    are applied per request by the fetcher."""
    return httpx.AsyncClient(
        headers={"User-Agent": settings.user_agent},
        transport=transport or httpx.AsyncHTTPTransport(retries=0),
        limits=httpx.Limits(max_connections=settings.concurrency),
    )


class Runner:
    """Fetch a batch of pages with a pool of workers.

    - One shared token bucket limits request starts across all workers,
      retries and redirect hops included (it is the `throttle` the fetcher
      awaits before every request).
    - A circuit breaker counts consecutive 429/5xx/connection failures. When
      it trips, every worker pauses for `breaker_cooldown`; the
      `breaker_max_trips`-th trip ends the run instead.
    - SIGINT/SIGTERM once: stop starting requests, let those on the wire
      finish and be recorded, cut backoff sleeps short, leave unfinished pages
      as they were. Twice: cancel everything now. Either way the manifest is
      consistent, so re-running resumes.

    A Runner does one run.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: Settings,
        manifest: Manifest,
        robots: RobotsRules,
        *,
        fetch: FetchFn = fetch_one,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self._manifest = manifest
        self._robots = robots
        self._fetch = fetch
        self._clock = clock or time.monotonic
        self._bucket = TokenBucket(
            settings.rate, clock=self._clock, sleep=self._bucket_sleep
        )

        self._gate = asyncio.Event()  # set while requests may go out
        self._gate.set()
        self._stop = asyncio.Event()
        self._stop_reason: RunStatus | None = None
        self._signals_received = 0
        self._hard_stop = False
        self._workers: list[asyncio.Task[None]] = []
        self._cooldown_task: asyncio.Task[None] | None = None

        self._outcomes: Counter[str] = Counter()
        self._requests = 0
        self._consecutive_failures = 0
        self._trips = 0
        self._selected = 0
        self._started = 0.0

    # -- public ----------------------------------------------------------

    def request_stop(self, reason: RunStatus = "interrupted") -> None:
        """Ask the run to wind down. The first reason given is the one kept."""
        if self._stop_reason is None:
            self._stop_reason = reason
            self._stop.set()

    async def run(
        self,
        *,
        refresh: bool = False,
        limit: int | None = None,
        stale_before: str | None = None,
        pages: list[Page] | None = None,
        handle_signals: bool = True,
    ) -> RunSummary:
        """Fetch a batch. `pages` replaces the manifest's own selection (and so
        overrides `limit` and `stale_before`); `refresh` then only labels the log."""
        if pages is None:
            pages = self._manifest.next_batch(
                refresh=refresh, limit=limit, stale_before=stale_before
            )
        self._selected = len(pages)
        self._started = self._clock()
        logger.info(
            "starting: %d pages%s, %.2f requests/s, %d in flight",
            len(pages),
            " (refresh)" if refresh else "",
            self._settings.rate,
            self._settings.concurrency,
        )

        queue = deque(pages)
        self._workers = [
            asyncio.create_task(self._worker(queue))
            for _ in range(min(self._settings.concurrency, len(pages)))
        ]
        installed = self._install_signal_handlers() if handle_signals else []
        try:
            await self._wait_for(self._workers)
        finally:
            self._remove_signal_handlers(installed)
            if self._cooldown_task is not None:
                self._cooldown_task.cancel()
                await asyncio.gather(self._cooldown_task, return_exceptions=True)

        summary = RunSummary(
            status=self._stop_reason or "completed",
            selected=self._selected,
            outcomes=self._outcomes,
            requests=self._requests,
            breaker_trips=self._trips,
            elapsed=self._clock() - self._started,
            pending_left=self._manifest.counts()["pending"],
        )
        logger.info(
            "%s: %d/%d pages (%s) in %.0fs, %d requests, %d breaker trips, %d still pending",
            summary.status,
            summary.processed,
            summary.selected,
            _format_counts(summary.outcomes),
            summary.elapsed,
            summary.requests,
            summary.breaker_trips,
            summary.pending_left,
        )
        return summary

    # -- workers ---------------------------------------------------------

    async def _worker(self, queue: deque[Page]) -> None:
        while not self._stop.is_set():
            try:
                page = queue.popleft()
            except IndexError:
                return
            try:
                result = await self._fetch(
                    self._client,
                    page,
                    self._settings,
                    self._robots,
                    throttle=self._throttle,
                    sleep=self._sleep,
                )
            except RunStopped:
                return  # page untouched: still pending (or still ok on refresh)
            record_result(self._manifest, result)
            self._record(result)

    async def _throttle(self) -> None:
        """Awaited by the fetcher before every request."""
        await self._gate.wait()  # closed while the breaker cools down
        if self._stop.is_set():
            raise RunStopped
        await self._bucket.acquire()
        if self._stop.is_set():
            raise RunStopped

    async def _sleep(self, seconds: float) -> None:
        """Sleep that a stop request cuts short."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _bucket_sleep(self, seconds: float) -> None:
        """The rate limiter's wait: at a slow rate it can be many seconds, so a
        stop request must end it (by raising) rather than be waited out."""
        await self._sleep(seconds)
        if self._stop.is_set():
            raise RunStopped

    # -- bookkeeping and the circuit breaker ----------------------------

    def _record(self, result: Result) -> None:
        self._outcomes[result.outcome] += 1
        self._requests += result.attempts

        # A result that ended on a non-transient response means the server is
        # answering; only unbroken runs of transient failures trip the breaker.
        if result.outcome != "failed" or result.transient_failures == 0:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += result.transient_failures
            if self._consecutive_failures >= self._settings.breaker_threshold:
                self._trip()

        done = self._selected and sum(self._outcomes.values())
        if done and done % self._settings.progress_every == 0:
            elapsed = self._clock() - self._started
            logger.info(
                "progress: %d/%d done (%s) | %.0fs elapsed | %.2f pages/s",
                done,
                self._selected,
                _format_counts(self._outcomes),
                elapsed,
                done / elapsed if elapsed else 0.0,
            )

    def _trip(self) -> None:
        self._consecutive_failures = 0
        self._trips += 1
        if self._trips >= self._settings.breaker_max_trips:
            logger.error(
                "circuit breaker tripped %d times: stopping the run", self._trips
            )
            self.request_stop("breaker_tripped")
            return
        logger.warning(
            "circuit breaker tripped (%d consecutive failures, trip %d of %d): "
            "pausing all requests for %.0fs",
            self._settings.breaker_threshold,
            self._trips,
            self._settings.breaker_max_trips,
            self._settings.breaker_cooldown,
        )
        self._gate.clear()
        self._cooldown_task = asyncio.create_task(self._cooldown())

    async def _cooldown(self) -> None:
        try:
            await self._sleep(self._settings.breaker_cooldown)
        finally:
            self._gate.set()

    # -- shutdown --------------------------------------------------------

    async def _wait_for(self, workers: list[asyncio.Task[None]]) -> None:
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            await self._cancel(workers)
            if not self._hard_stop:  # someone else cancelled us: let it propagate
                raise
        except BaseException:
            await self._cancel(workers)
            raise

    @staticmethod
    async def _cancel(workers: list[asyncio.Task[None]]) -> None:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    def _install_signal_handlers(self) -> list[signal.Signals]:
        loop = asyncio.get_running_loop()
        installed = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._on_signal, sig)
            except (NotImplementedError, ValueError, RuntimeError):
                continue  # not the main thread, or not supported here
            installed.append(sig)
        return installed

    @staticmethod
    def _remove_signal_handlers(installed: list[signal.Signals]) -> None:
        loop = asyncio.get_running_loop()
        for sig in installed:
            loop.remove_signal_handler(sig)

    def _on_signal(self, sig: signal.Signals) -> None:
        self._signals_received += 1
        if self._signals_received == 1:
            logger.warning(
                "%s received: finishing requests already in flight, then stopping "
                "(send it again to abort immediately)",
                sig.name,
            )
            self.request_stop("interrupted")
        else:
            logger.warning("%s received again: aborting now", sig.name)
            self.request_stop("interrupted")
            self._hard_stop = True
            for worker in self._workers:
                worker.cancel()


def _format_counts(counts: Counter[str]) -> str:
    order = ("ok", "not_modified", "gone", "failed")
    return ", ".join(f"{counts[k]} {k}" for k in order if counts[k]) or "nothing yet"
