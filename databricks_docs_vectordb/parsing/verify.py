"""Check the chunk store against the manifest's `ok` rows and its own
consistency invariants. No network; changes nothing.

Reports every problem found, not just the first, mirroring
`scraping/verify.py`'s shape. `--by-section`'s table reuses `report.py`'s
per-section stats (the same ones `parse status` prints on its own) rather
than recomputing them -- there's nothing verify-specific about that data;
`parse status` is the quick look, `parse verify` is the one that also
checks invariants and exits non-zero if any fail.
"""

from dataclasses import dataclass, field

from databricks_docs_vectordb.parsing.config import Settings
from databricks_docs_vectordb.parsing.report import (
    SectionStats,
    format_sections,
    overall_stats,
    section_stats,
)
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore
from databricks_docs_vectordb.scraping.config import Source
from databricks_docs_vectordb.scraping.manifest import Manifest
from databricks_docs_vectordb.scraping.verify import Finding


@dataclass
class VerifyReport:
    checked: int = 0  # ok manifest rows examined
    unparsed: int = 0  # of those, how many have no chunks yet
    total_chunks: int = 0
    violations: list[Finding] = field(default_factory=list)
    sections: dict[tuple[str, str], SectionStats] = field(default_factory=dict)
    overall: SectionStats = field(default_factory=lambda: overall_stats([]))

    @property
    def ok(self) -> bool:
        return not self.violations


def verify_store(
    manifest: Manifest,
    store: ChunkStore,
    settings: Settings,
    sources: tuple[Source, ...],
    *,
    complete: bool = False,
) -> VerifyReport:
    """Check the chunk store against the manifest and itself. No network.

    Invariants: no chunk's text is empty; no chunk exceeds the target size
    ceiling (the configured target plus the configured overlap, since
    overlap legitimately pushes a chunk a bit past the target -- see
    `chunker.py`) unless it's flagged `is_oversized`; every
    `prev_chunk_id`/`next_chunk_id` resolves to a real chunk of the same
    page; `url`, `source`, and `chunk_index` are set on every chunk.
    Reports every violation found, not just the first.

    An `ok` manifest row with no chunks yet is *not* a violation unless
    `complete` says the whole corpus is supposed to be parsed by now (a
    pilot run, Step 9, parses only a sample on purpose, so this can't be a
    hard rule by default) -- mirrors `scrape verify`'s `complete` flag and
    its treatment of `pending` rows.

    Chunks belonging to a page that is no longer `ok` (removed, gone, or
    re-fetched and failed since it was last parsed) are not checked and
    not flagged as orphans here -- Phase 7's job, not this one's.
    """
    report = VerifyReport()
    ceiling = settings.target_chunk_size + settings.overlap_size

    for page in manifest.pages_with_status("ok"):
        report.checked += 1
        chunks = store.get_chunks(page.url)
        if not chunks:
            report.unparsed += 1
            if complete:
                report.violations.append(
                    Finding(
                        "no-chunks",
                        page.url,
                        "an ok page has no chunks, but parsing is supposed to "
                        "be complete",
                    )
                )
            continue
        _check_chunks(page.url, chunks, ceiling, report)

    summaries = store.all_summaries()
    report.total_chunks = len(summaries)
    report.sections = section_stats(summaries, sources)
    report.overall = overall_stats(summaries)
    return report


def _check_chunks(
    url: str, chunks: list[Chunk], ceiling: int, report: VerifyReport
) -> None:
    ids = {c.chunk_id for c in chunks}
    for chunk in chunks:
        if not chunk.text.strip():
            report.violations.append(
                Finding("empty-chunk", chunk.chunk_id, "chunk text is empty")
            )
        if chunk.char_count > ceiling and not chunk.is_oversized:
            report.violations.append(
                Finding(
                    "chunk-too-large",
                    chunk.chunk_id,
                    f"{chunk.char_count} chars exceeds the {ceiling}-char ceiling "
                    "(target + overlap) but is_oversized is not set",
                )
            )
        for field_name, neighbor in (
            ("prev_chunk_id", chunk.prev_chunk_id),
            ("next_chunk_id", chunk.next_chunk_id),
        ):
            if neighbor is not None and neighbor not in ids:
                report.violations.append(
                    Finding(
                        "dangling-neighbor",
                        chunk.chunk_id,
                        f"{field_name} {neighbor!r} is not a chunk of {url!r}",
                    )
                )
        missing = [name for name in ("url", "source") if not getattr(chunk, name)]
        if chunk.chunk_index is None:
            missing.append("chunk_index")
        if missing:
            report.violations.append(
                Finding(
                    "missing-field",
                    chunk.chunk_id,
                    f"chunk has no {', '.join(missing)}",
                )
            )


def format_report(
    report: VerifyReport, settings: Settings, *, by_section: bool = False
) -> str:
    lines = [
        f"checked {report.checked} ok manifest page(s) against {settings.parsed_path}"
    ]
    lines.append(f"{report.total_chunks} chunk(s) in the store")
    if report.unparsed and not any(f.kind == "no-chunks" for f in report.violations):
        lines.append(
            f"note: {report.unparsed} ok page(s) have not been parsed yet "
            "(not violations unless --complete); `parse run` handles them"
        )
    if report.violations:
        lines.append(f"violations ({len(report.violations)}):")
        for f in report.violations:
            lines.append(f"  [{f.kind}] {f.where}: {f.detail}")
    if by_section:
        lines.append("")
        lines.extend(format_sections(report.sections, report.overall))
    lines.append("")
    lines.append("OK" if report.ok else f"FAILED: {len(report.violations)} violations")
    return "\n".join(lines)
