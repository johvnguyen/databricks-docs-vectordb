import hashlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from databricks_docs_vectordb.scraping.config import Source
from databricks_docs_vectordb.scraping.manifest import Page
from databricks_docs_vectordb.scraping.verify import section_of

Section = tuple[str, str]  # (source, top-level section)


class SampleError(RuntimeError):
    """The requested sample can't be drawn as asked."""


@dataclass(frozen=True)
class Sample:
    pages: list[Page]  # in the order they should be fetched
    allocation: dict[Section, int]  # pages taken from each section
    candidates: int  # pages the sample was drawn from
    minimum: int  # smallest N that covers every section with `floor` pages

    @property
    def sections(self) -> int:
        return len(self.allocation)


def _url(page: Page) -> str:
    return page.url


def sample_pages(
    pages: Sequence[Page],
    sources: tuple[Source, ...],
    n: int,
    *,
    floor: int = 2,
    identity: Callable[[Page], str] = _url,
    noun: str = "pages",
) -> Sample:
    """Pick about `n` pages, spread across every section of every source.

    Every section contributes at least `floor` pages (or all of them, if it has
    fewer), which is what makes a small crawl evidence about *all* the page
    templates, not just the big sections. The rest of the budget goes to the
    sections in proportion to how many pages they have left, so the largest
    ones get a few more.

    Deterministic: within a section, pages are ranked by a hash of their URL,
    so the same candidates always give the same sample, whatever order they
    arrive in, and the sample isn't biased toward whatever sorts first.

    Raises SampleError if `n` is too small to give every section its `floor`.
    If `n` is at least the number of candidates, every candidate is returned.

    Anything with a `.url` and `.source` can be sampled, not just manifest
    `Page`s: `embedding/sampling.py` samples chunks, passing the chunk_id as
    `identity` (what the ranking hashes -- every chunk of a page shares its
    URL, which would otherwise rank a page's chunks together) and "chunks"
    as the `noun` in the error message.
    """

    def rank(page: Page) -> str:
        return hashlib.sha256(identity(page).encode()).hexdigest()

    by_section: dict[Section, list[Page]] = defaultdict(list)
    for page in pages:
        by_section[section_of(page, sources)].append(page)
    for members in by_section.values():
        members.sort(key=rank)

    sizes = {section: len(members) for section, members in by_section.items()}
    minimum = sum(min(floor, size) for size in sizes.values())
    if n < minimum:
        raise SampleError(
            f"--sample {n} is too small: covering all {len(sizes)} sections with at "
            f"least {floor} {noun} each takes {minimum} {noun}. Use --sample "
            f"{minimum} or more (or --limit for a plain cap)"
        )

    allocation = _allocate(sizes, min(n, len(pages)), floor)
    chosen = [
        page
        for section, members in by_section.items()
        for page in members[: allocation[section]]
    ]
    chosen.sort(
        key=rank
    )  # fetch order: a pseudo-random mix, so an early stop still spans sections
    return Sample(chosen, dict(sorted(allocation.items())), len(pages), minimum)


def _allocate(sizes: dict[Section, int], budget: int, floor: int) -> dict[Section, int]:
    allocation = {section: min(floor, size) for section, size in sizes.items()}
    extra = budget - sum(allocation.values())
    while extra > 0:
        room = {s: sizes[s] - allocation[s] for s in sizes if sizes[s] > allocation[s]}
        if not room:
            break
        total = sum(room.values())
        # a whole-number share of what's left, proportional to each section's room
        shares = {s: min(r, extra * r // total) for s, r in room.items()}
        if not any(shares.values()):
            # fewer extras than sections: the sections with the most room get one each
            biggest = sorted(room, key=lambda s: (-room[s], s))[:extra]
            shares = {s: 1 for s in biggest}
        for section, share in shares.items():
            allocation[section] += share
        extra -= sum(shares.values())
    return allocation
