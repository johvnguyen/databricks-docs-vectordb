"""Formatting for `parse status`: counts by source/section, the chunk-size
distribution, and how many chunks are oversized. Mirrors the shape of
`scraping/report.py` and `scrape verify --by-section`'s table.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from databricks_docs_vectordb.parsing.store import ChunkSummary
from databricks_docs_vectordb.scraping.config import Source

TOP_LEVEL = "(top-level)"


@dataclass(frozen=True)
class SectionStats:
    pages: int  # distinct urls
    chunks: int
    avg_chars: int
    median_chars: int
    oversized: int


def section_of(url: str, source: str, sources: Sequence[Source]) -> str:
    """The top-level section of a chunk's page, under `source`'s URL prefix.

    Mirrors `scraping.verify.section_of`'s rule (`.../sql/` and
    `.../sql/x` are both `sql`; a lone `.../introduction` is top-level) --
    duplicated rather than shared because that one takes a full manifest
    `Page`, and a chunk row only has the `(url, source)` pair.
    """
    prefix = next((s.url_prefix for s in sources if s.name == source), None)
    if prefix is None or not url.startswith(prefix):
        return "(unknown)"
    head, slash, _ = url[len(prefix) :].partition("/")
    return head if slash and head else TOP_LEVEL


def _stats_of(rows: Sequence[ChunkSummary]) -> SectionStats:
    sizes = [r.char_count for r in rows]
    return SectionStats(
        pages=len({r.url for r in rows}),
        chunks=len(rows),
        avg_chars=round(sum(sizes) / len(sizes)) if sizes else 0,
        median_chars=round(median(sizes)) if sizes else 0,
        oversized=sum(1 for r in rows if r.is_oversized),
    )


def section_stats(
    summaries: Sequence[ChunkSummary], sources: Sequence[Source]
) -> dict[tuple[str, str], SectionStats]:
    """Group per-chunk summaries into per-`(source, section)` stats."""
    by_key: dict[tuple[str, str], list[ChunkSummary]] = defaultdict(list)
    for summary in summaries:
        section = section_of(summary.url, summary.source, sources)
        by_key[(summary.source, section)].append(summary)
    return {key: _stats_of(rows) for key, rows in by_key.items()}


def overall_stats(summaries: Sequence[ChunkSummary]) -> SectionStats:
    """The same stats as `section_stats`, across every chunk at once.

    Not derivable from summing/averaging the per-section `SectionStats`
    (an average of averages isn't the true average, and a median can't be
    reconstructed from other medians at all) -- computed straight from the
    full list instead.
    """
    return _stats_of(summaries)


def format_sections(
    sections: dict[tuple[str, str], SectionStats], overall: SectionStats
) -> list[str]:
    """A table of chunk counts and size stats per source/section."""
    if not sections:
        return ["per-section: no chunks yet"]
    rows = [
        (
            f"{source}/{section}",
            s.pages,
            s.chunks,
            s.avg_chars,
            s.median_chars,
            s.oversized,
        )
        for (source, section), s in sorted(sections.items())
    ]
    rows.append(
        (
            "total",
            overall.pages,
            overall.chunks,
            overall.avg_chars,
            overall.median_chars,
            overall.oversized,
        )
    )
    header = (
        "source/section",
        "pages",
        "chunks",
        "avg chars",
        "median chars",
        "oversized",
    )
    width = max(len(str(r[0])) for r in (*rows, header))
    lines = ["per-section (pages parsed, chunks produced, size stats):"]
    lines.append(f"  {header[0]:<{width}}" + "".join(f"{h:>14}" for h in header[1:]))
    for row in rows:
        if row[0] == "total":
            lines.append("  " + "-" * (width + 70))
        lines.append(f"  {row[0]:<{width}}" + "".join(f"{n:>14}" for n in row[1:]))
    return lines
