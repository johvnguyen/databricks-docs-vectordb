import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from databricks_docs_vectordb.scraping.config import Settings, Source
from databricks_docs_vectordb.scraping.manifest import STATUSES, Manifest
from databricks_docs_vectordb.scraping.robots import RobotsRules
from databricks_docs_vectordb.scraping.urls import parse_sitemap, url_to_rel_path

_ROBOTS_DIRECTIVE = re.compile(r"^\s*user-agent\s*:", re.IGNORECASE | re.MULTILINE)


class EnumerateError(RuntimeError):
    """Enumeration can't continue safely; nothing in the manifest was changed."""


@dataclass(frozen=True)
class SourceResult:
    source: str
    listed: int  # URLs in the sitemap
    disallowed: int  # excluded by robots.txt
    outside_source: list[str]  # not under the source's URL prefix
    invalid: list[str]  # rejected by url_to_rel_path
    allowed: int
    added: int
    revived: int
    removed: int
    sitemap_copy: Path


@dataclass(frozen=True)
class EnumerateResult:
    robots_copy: Path
    robots_bytes: int
    sources: list[SourceResult]
    totals: dict[str, Counter[str]] = field(default_factory=dict)


@dataclass
class _Fetched:
    source: Source
    listed: int
    disallowed: int
    outside_source: list[str]
    invalid: list[str]
    allowed: list[str]
    sitemap_copy: Path
    sitemap_content: bytes


def enumerate_sources(
    settings: Settings,
    client: httpx.Client,
    manifest: Manifest,
    *,
    sleep: Callable[[float], None] | None = None,
) -> EnumerateResult:
    """Fetch robots.txt and every sitemap, then sync the manifest.

    Everything is fetched and validated before anything is written, so if any
    request or check fails, no file or manifest row has changed. The client
    should already carry the User-Agent, timeouts and redirect limit.
    """
    fetch = _Fetcher(client, sleep or time.sleep, delay=1 / settings.rate)

    origins = {_origin(source.sitemap_url) for source in settings.sources}
    if len(origins) != 1:
        raise EnumerateError(
            f"sources span several hosts ({sorted(origins)}); per-host robots.txt "
            "handling is not implemented"
        )
    (origin,) = origins

    robots_response = fetch(f"{origin}/robots.txt")
    robots_text = robots_response.text
    if not _ROBOTS_DIRECTIVE.search(robots_text):
        raise EnumerateError(
            f"{origin}/robots.txt has no User-agent line; refusing to treat an "
            "unrecognized response as 'everything allowed'"
        )
    rules = RobotsRules(robots_text)

    fetched = [
        _fetch_source(source, fetch, rules, settings) for source in settings.sources
    ]

    # Every fetch succeeded and validated: now save the raw copies, then sync.
    _write_atomic(settings.robots_path, robots_response.content)
    for item in fetched:
        _write_atomic(item.sitemap_copy, item.sitemap_content)

    results = []
    for item in fetched:
        name = item.source.name
        added = manifest.add_pending(name, item.allowed)
        removed = manifest.mark_removed(name, item.allowed)
        results.append(
            SourceResult(
                source=name,
                listed=item.listed,
                disallowed=item.disallowed,
                outside_source=item.outside_source,
                invalid=item.invalid,
                allowed=len(item.allowed),
                added=added.added,
                revived=added.revived,
                removed=removed,
                sitemap_copy=item.sitemap_copy,
            )
        )
    return EnumerateResult(
        robots_copy=settings.robots_path,
        robots_bytes=len(robots_response.content),
        sources=results,
        totals=manifest.counts_by_source(),
    )


def format_report(result: EnumerateResult) -> str:
    lines = [f"robots.txt: {result.robots_bytes} bytes saved to {result.robots_copy}"]
    for r in result.sources:
        lines.append(
            f"{r.source}: {r.listed} URLs in sitemap; skipped {r.disallowed} by "
            f"robots.txt, {len(r.outside_source)} outside the source, "
            f"{len(r.invalid)} invalid; {r.allowed} allowed"
        )
        lines.append(f"  {r.added} new, {r.revived} revived, {r.removed} removed")
        lines.append(f"  sitemap copy: {r.sitemap_copy}")
        for label, urls in (
            ("outside the source", r.outside_source),
            ("invalid", r.invalid),
        ):
            for url in urls[:5]:
                lines.append(f"  skipped ({label}): {url}")
            if len(urls) > 5:
                lines.append(f"  ... and {len(urls) - 5} more ({label})")
    lines.append("manifest:")
    for source, counts in result.totals.items():
        summary = ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s])
        lines.append(f"  {source}: {summary}")
    return "\n".join(lines)


class _Fetcher:
    """GET with a polite pause between requests; anything but a 200 is an error."""

    def __init__(
        self, client: httpx.Client, sleep: Callable[[float], None], delay: float
    ) -> None:
        self._client = client
        self._sleep = sleep
        self._delay = delay
        self._requests = 0

    def __call__(self, url: str) -> httpx.Response:
        if self._requests:
            self._sleep(self._delay)
        self._requests += 1
        try:
            response = self._client.get(url)
        except httpx.HTTPError as e:
            raise EnumerateError(f"could not fetch {url}: {e!r}") from e
        if response.status_code != 200:
            raise EnumerateError(f"{url} returned HTTP {response.status_code}")
        return response


def _fetch_source(
    source: Source, fetch: _Fetcher, rules: RobotsRules, settings: Settings
) -> _Fetched:
    response = fetch(source.sitemap_url)
    try:
        urls = parse_sitemap(response.content)
    except ValueError as e:
        raise EnumerateError(
            f"{source.name}: bad sitemap at {source.sitemap_url}: {e}"
        ) from e
    if not urls:
        raise EnumerateError(
            f"{source.name}: sitemap at {source.sitemap_url} lists no URLs"
        )

    outside: list[str] = []
    invalid: list[str] = []
    disallowed = 0
    allowed: list[str] = []
    for url in urls:
        if not url.startswith(source.url_prefix):
            outside.append(url)
            continue
        try:
            url_to_rel_path(url, settings.sources)
        except ValueError:
            invalid.append(url)
            continue
        if rules.is_allowed(url):
            allowed.append(url)
        else:
            disallowed += 1
    if not allowed:
        raise EnumerateError(
            f"{source.name}: none of the {len(urls)} sitemap URLs are usable "
            f"({disallowed} disallowed, {len(outside)} outside the source, "
            f"{len(invalid)} invalid)"
        )

    today = datetime.now(UTC).date().isoformat()
    copy = settings.sitemaps_dir / f"{source.name}-{today}.xml"
    return _Fetched(
        source, len(urls), disallowed, outside, invalid, allowed, copy, response.content
    )


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a temp file and rename, so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)  # mkstemp creates private (0600) files
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
