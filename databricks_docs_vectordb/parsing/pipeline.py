"""Per-page orchestration: read one page from the Phase 1 snapshot, extract
its metadata and content, chunk it, and store the result -- skipping pages
whose chunks are already current.

Idempotent, incremental re-parse: a page is only walked/extracted/chunked
again if the store's chunks for it don't already carry this manifest row's
`sha256` (or there are none yet). This is what makes re-running parsing
after a `scrape fetch --refresh` cheap, and safe to interrupt.

Callers choose *which* pages to process (every `ok` manifest row, a
`--limit`'d batch, a stratified sample, ...) and pass them in; this module
only does the per-page work, mirroring how `scraping/runner.py` takes an
explicit `pages` sequence rather than querying the manifest itself.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from databricks_docs_vectordb.parsing.chunker import chunk_page
from databricks_docs_vectordb.parsing.config import Settings
from databricks_docs_vectordb.parsing.html_walker import walk
from databricks_docs_vectordb.parsing.metadata import extract as extract_metadata
from databricks_docs_vectordb.parsing.store import ChunkStore
from databricks_docs_vectordb.scraping.manifest import Page

Outcome = Literal["parsed", "skipped"]


@dataclass(frozen=True)
class PageResult:
    url: str
    outcome: Outcome
    chunk_count: int


def parse_page(
    page: Page, *, raw_dir: Path, store: ChunkStore, settings: Settings
) -> PageResult:
    """Parse and store one manifest row's chunks, unless already current.

    "Already current" means the store already has chunks for this `url`
    and every one of them carries this row's `sha256` -- nothing changed
    since the last time this page was parsed. A page that has never
    produced any chunks (an edge case: no extractable content at all) is
    always reparsed, since an empty result is indistinguishable from
    "never parsed"; this is cheap because there's nothing to chunk.
    """
    existing = store.get_chunks(page.url)
    if existing and all(c.page_sha256 == page.sha256 for c in existing):
        return PageResult(page.url, "skipped", len(existing))

    html = (raw_dir / page.rel_path).read_text(encoding="utf-8")
    nodes = walk(html)
    metadata = extract_metadata(
        html,
        url=page.url,
        source=page.source,
        sha256=page.sha256 or "",
        fetched_at=page.fetched_at,
    )
    chunks = chunk_page(nodes, metadata, settings)
    store.replace_page_chunks(page.url, chunks)
    return PageResult(page.url, "parsed", len(chunks))


def parse_pages(
    pages: Iterable[Page], *, raw_dir: Path, store: ChunkStore, settings: Settings
) -> Iterator[PageResult]:
    """Parse each page in turn, yielding one result as each one completes."""
    for page in pages:
        yield parse_page(page, raw_dir=raw_dir, store=store, settings=settings)
