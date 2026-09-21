import asyncio
import hashlib
import logging
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urldefrag, urljoin

import httpx

from databricks_docs_vectordb.scraping.config import Settings
from databricks_docs_vectordb.scraping.files import write_atomic
from databricks_docs_vectordb.scraping.manifest import Manifest, Page
from databricks_docs_vectordb.scraping.robots import RobotsRules

logger = logging.getLogger(__name__)

Outcome = Literal["ok", "not_modified", "gone", "failed"]

_ARTICLE = re.compile(rb"<article[\s>]", re.IGNORECASE)
_TITLE = re.compile(rb"<title[\s>]", re.IGNORECASE)


@dataclass(frozen=True)
class Result:
    """What happened to one page; `record_result` turns it into a manifest update.

    `ok`: a new or changed body was written to disk.
    `not_modified`: the stored body is still current (a 304, or a 200 with
    identical content).
    `gone`: the page is a 404/410.
    `failed`: retries were exhausted or the failure can't be retried.
    """

    url: str
    outcome: Outcome
    attempts: int
    http_status: int | None = None
    final_url: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    error: str | None = None
    # 429s, 5xx and connection failures seen along the way. The runner feeds
    # this to its circuit breaker; it is not the same as `attempts`.
    transient_failures: int = 0


class _Permanent(Exception):
    """Retrying won't help."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class _Transient(Exception):
    """Worth retrying after a pause."""

    def __init__(
        self,
        message: str,
        http_status: int | None = None,
        retry_after: float | None = None,
        counts_for_breaker: bool = True,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.retry_after = retry_after
        self.counts_for_breaker = counts_for_breaker


async def fetch_one(
    client: httpx.AsyncClient,
    page: Page,
    settings: Settings,
    robots: RobotsRules,
    *,
    throttle: Callable[[], Awaitable[None]] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    rng: Callable[[], float] | None = None,
) -> Result:
    """Fetch one page and store its body; never raises for HTTP-level problems.

    - Sends `If-None-Match` / `If-Modified-Since` only when a body is already
      stored on disk for the page, and only to the URL that produced it.
    - Follows redirects one hop at a time, checking each target against the
      source prefixes and robots.txt *before* requesting it.
    - Retries timeouts, connection errors, 429, 5xx and pages that don't look
      like docs (no `<article>` / `<title>`), with exponential backoff and
      jitter; `Retry-After` on 429/503 is honored. 404/410 are `gone`; other
      4xx and redirect problems fail immediately.
    - Writes the body atomically. A failure to write (disk full, permissions)
      raises OSError rather than being recorded as a page failure, so the
      run stops instead of failing every remaining page.

    `throttle` is awaited before every request, retries and redirect hops
    included, so a shared rate limiter sees all of them. `sleep` and `rng` are
    injection points for tests.
    """
    sleep = sleep or asyncio.sleep
    rng = rng or random.random

    dest = settings.raw_dir / page.rel_path
    if not dest.resolve().is_relative_to(settings.raw_dir.resolve()):
        return Result(
            page.url, "failed", 0, error=f"unsafe file path: {page.rel_path!r}"
        )
    validators = _validators(page, dest)

    transient_failures = 0
    last: _Transient | None = None
    for attempt in range(1, settings.max_attempts + 1):
        try:
            response, fetched_url, conditional = await _follow(
                client, page, validators, settings, robots, throttle
            )
            result = await _interpret(response, fetched_url, conditional, page, dest)
            logger.debug(
                "%s -> %s (HTTP %s)", page.url, result.outcome, result.http_status
            )
            return replace(
                result, attempts=attempt, transient_failures=transient_failures
            )
        except _Permanent as e:
            logger.warning("%s failed: %s", page.url, e)
            return Result(
                page.url,
                "failed",
                attempt,
                http_status=e.http_status,
                error=str(e),
                transient_failures=transient_failures,
            )
        except _Transient as e:
            last = e
            if e.counts_for_breaker:
                transient_failures += 1
            if e.retry_after is not None and e.retry_after > settings.retry_after_max:
                return Result(
                    page.url,
                    "failed",
                    attempt,
                    http_status=e.http_status,
                    error=(
                        f"{e}; server asked us to wait {e.retry_after:.0f}s, "
                        f"more than the {settings.retry_after_max:.0f}s limit"
                    ),
                    transient_failures=transient_failures,
                )
            if attempt < settings.max_attempts:
                delay = _backoff(attempt, settings, rng(), e.retry_after)
                logger.warning(
                    "%s: %s; retrying in %.1fs (attempt %d/%d)",
                    page.url,
                    e,
                    delay,
                    attempt,
                    settings.max_attempts,
                )
                await sleep(delay)

    assert last is not None
    return Result(
        page.url,
        "failed",
        settings.max_attempts,
        http_status=last.http_status,
        error=f"gave up after {settings.max_attempts} attempts: {last}",
        transient_failures=transient_failures,
    )


def record_result(manifest: Manifest, result: Result, now: str | None = None) -> None:
    """Apply a fetch result to the manifest."""
    match result.outcome:
        case "ok":
            assert result.sha256 is not None and result.size_bytes is not None
            manifest.record_ok(
                result.url,
                http_status=result.http_status or 200,
                sha256=result.sha256,
                size_bytes=result.size_bytes,
                etag=result.etag,
                last_modified=result.last_modified,
                final_url=result.final_url,
                now=now,
            )
        case "not_modified":
            manifest.record_not_modified(
                result.url,
                http_status=result.http_status or 304,
                etag=result.etag,
                last_modified=result.last_modified,
                now=now,
            )
        case "gone":
            manifest.record_gone(
                result.url, http_status=result.http_status or 404, now=now
            )
        case "failed":
            manifest.record_failed(
                result.url,
                error=result.error or "unknown error",
                attempts=result.attempts,
                http_status=result.http_status,
            )


def _validators(page: Page, dest: Path) -> dict[str, str]:
    """Conditional-request headers, but only if a stored body backs them.

    Without the file, a 304 would mark the page `ok` with nothing to read.
    """
    if not (page.sha256 and dest.is_file()):
        return {}
    headers = {}
    if page.etag:
        headers["If-None-Match"] = page.etag
    if page.last_modified:
        headers["If-Modified-Since"] = page.last_modified
    return headers


async def _follow(
    client: httpx.AsyncClient,
    page: Page,
    validators: dict[str, str],
    settings: Settings,
    robots: RobotsRules,
    throttle: Callable[[], Awaitable[None]] | None,
) -> tuple[httpx.Response, str, bool]:
    """One request chain: the page, plus any redirects. Returns the final
    response, the URL it came from, and whether conditional headers were sent."""
    url = page.url
    validated_url = page.final_url or page.url  # the URL that produced the stored body
    timeout = httpx.Timeout(settings.read_timeout, connect=settings.connect_timeout)

    for hop in range(settings.max_redirects + 1):
        headers = validators if url == validated_url else {}
        if throttle is not None:
            await throttle()
        try:
            response = await client.get(
                url, headers=headers, timeout=timeout, follow_redirects=False
            )
        except httpx.InvalidURL as e:
            raise _Permanent(f"invalid URL {url!r}: {e}") from e
        except httpx.TransportError as e:
            raise _Transient(f"{type(e).__name__}: {e}".rstrip(": ")) from e

        # Not `is_redirect`, which is true for any 3xx (304 included).
        if not response.has_redirect_location:
            return response, url, bool(headers)
        if hop == settings.max_redirects:
            raise _Permanent(f"more than {settings.max_redirects} redirects")

        target = urldefrag(urljoin(str(response.url), response.headers["location"])).url
        if not any(target.startswith(s.url_prefix) for s in settings.sources):
            raise _Permanent(f"redirected outside the configured sources: {target}")
        if not robots.is_allowed(target):
            raise _Permanent(f"redirected to a page robots.txt disallows: {target}")
        url = target

    raise AssertionError("unreachable")  # pragma: no cover


async def _interpret(
    response: httpx.Response,
    fetched_url: str,
    conditional: bool,
    page: Page,
    dest: Path,
) -> Result:
    status = response.status_code
    final_url = fetched_url if fetched_url != page.url else None
    etag = response.headers.get("etag")
    last_modified = response.headers.get("last-modified")

    if status == 304:
        if not conditional:
            raise _Permanent(
                "HTTP 304 in reply to a request that wasn't conditional", 304
            )
        return Result(
            page.url,
            "not_modified",
            0,
            http_status=304,
            final_url=final_url,
            etag=etag,
            last_modified=last_modified,
        )

    if status == 200:
        body = response.content
        if problem := invalid_body(body):
            raise _Transient(problem, 200, counts_for_breaker=False)
        digest = hashlib.sha256(body).hexdigest()
        common = dict(
            http_status=200,
            final_url=final_url,
            etag=etag,
            last_modified=last_modified,
            sha256=digest,
            size_bytes=len(body),
        )
        if digest == page.sha256 and dest.is_file():
            # The server ignored (or changed) the validators but the content is
            # identical: leave the file alone, and adopt the new validators.
            return Result(page.url, "not_modified", 0, **common)
        await asyncio.to_thread(write_atomic, dest, body)
        return Result(page.url, "ok", 0, **common)

    if status in (404, 410):
        return Result(page.url, "gone", 0, http_status=status, final_url=final_url)

    if status == 429 or status >= 500:
        raise _Transient(f"HTTP {status}", status, retry_after=_retry_after(response))

    raise _Permanent(f"HTTP {status}", status)


def invalid_body(body: bytes) -> str | None:
    """Why a 200 response doesn't look like a docs page, or None if it does."""
    if not _ARTICLE.search(body):
        return "HTTP 200 but the page has no <article> element"
    if not _TITLE.search(body):
        return "HTTP 200 but the page has no <title>"
    return None


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds from a `Retry-After` header; only honored on 429 and 503."""
    if response.status_code not in (429, 503):
        return None
    value = response.headers.get("retry-after", "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _backoff(
    attempt: int, settings: Settings, unit: float, retry_after: float | None
) -> float:
    """Exponential backoff with jitter (50-100% of the nominal delay), never
    shorter than what the server asked for."""
    nominal = min(settings.backoff_max, settings.backoff_base * 2 ** (attempt - 1))
    delay = nominal * (0.5 + unit / 2)
    return max(delay, retry_after) if retry_after is not None else delay
