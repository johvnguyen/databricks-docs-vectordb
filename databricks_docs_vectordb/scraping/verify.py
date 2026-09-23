import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

from databricks_docs_vectordb.scraping.config import Settings, Source
from databricks_docs_vectordb.scraping.fetcher import invalid_body
from databricks_docs_vectordb.scraping.manifest import Manifest, Page
from databricks_docs_vectordb.scraping.report import format_counts
from databricks_docs_vectordb.scraping.urls import url_to_rel_path

# Matched against a lower-cased copy of the body, inside the <article> element.
_ARTICLE_OPEN = re.compile(rb"<article[\s>]")
_H1 = re.compile(rb"<h1[\s>]")
_CODE = re.compile(rb"<pre[\s>]")
_TABLE = re.compile(rb"<table[\s>]")

TOP_LEVEL = "(top-level)"


@dataclass(frozen=True)
class Finding:
    kind: str  # a stable short name, e.g. "missing-file"
    where: str  # the URL or file path concerned
    detail: str


@dataclass
class SectionStats:
    """Fetched pages in one section, and how many have each structural feature."""

    pages: int = 0
    article: int = 0
    h1: int = 0
    code: int = 0
    tables: int = 0


@dataclass
class VerifyReport:
    checked: int = 0  # `ok` rows examined
    statuses: Counter[str] = field(default_factory=Counter)
    violations: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    sections: dict[tuple[str, str], SectionStats] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations


def verify_snapshot(
    settings: Settings, manifest: Manifest, *, complete: bool = False
) -> VerifyReport:
    """Check the raw snapshot against the manifest. No network.

    Reports every problem it finds, not just the first. `complete` says the
    crawl is supposed to be finished, so `pending` rows are violations too.
    `failed` and `gone` rows are never violations: they are pages the site
    would not give us, and `scrape status` lists them.
    """
    report = VerifyReport()
    pages = manifest.all_pages()
    report.statuses = Counter(page.status for page in pages)
    raw = settings.raw_dir.resolve()

    _check_paths(pages, settings, raw, report)
    for page in pages:
        if page.status == "ok":
            _check_fetched_page(page, settings, raw, report)
    _check_stray_files(pages, raw, report)

    if complete and report.statuses["pending"]:
        report.violations.append(
            Finding(
                "pending-remaining",
                "manifest",
                f"{report.statuses['pending']} pages are still pending, but the "
                "crawl is supposed to be complete",
            )
        )
    return report


def section_of(page: Page, sources: tuple[Source, ...]) -> tuple[str, str]:
    """(source, top-level section) for a page.

    The section is the first path segment after the source's URL prefix, and
    only counts if more path follows it: `.../sql/` and `.../sql/x` are in
    `sql`, but a lone `.../introduction` is a top-level page.
    """
    prefix = next((s.url_prefix for s in sources if s.name == page.source), None)
    if prefix is None or not page.url.startswith(prefix):
        return page.source, "(unknown)"
    head, slash, _ = page.url[len(prefix) :].partition("/")
    return page.source, head if slash and head else TOP_LEVEL


def format_report(
    report: VerifyReport, settings: Settings, *, by_section: bool = False
) -> str:
    lines = [f"checked {report.checked} fetched pages against {settings.raw_dir}"]
    lines.append(f"manifest: {format_counts(report.statuses)}")
    unfetched = report.statuses["failed"] + report.statuses["gone"]
    if unfetched:
        lines.append(
            f"note: {unfetched} pages the site would not give us (failed or gone); "
            "`scrape status` lists them, and they are not violations"
        )
    for label, findings in (
        ("warnings", report.warnings),
        ("violations", report.violations),
    ):
        if findings:
            lines.append(f"{label} ({len(findings)}):")
            for f in findings:
                lines.append(f"  [{f.kind}] {f.where}: {f.detail}")
    if by_section:
        lines.append("")
        lines.extend(format_sections(report.sections))
    lines.append("")
    lines.append("OK" if report.ok else f"FAILED: {len(report.violations)} violations")
    return "\n".join(lines)


def format_sections(sections: dict[tuple[str, str], SectionStats]) -> list[str]:
    """A table of fetched pages per section, with structural feature counts."""
    if not sections:
        return ["per-section: no fetched pages"]
    rows = [
        (f"{source}/{section}", s.pages, s.article, s.h1, s.code, s.tables)
        for (source, section), s in sorted(sections.items())
    ]
    rows.append(
        (
            "total",
            *(sum(r[i] for r in rows) for i in range(1, 6)),
        )
    )
    header = ("source/section", "pages", "<article", "<h1>", "code", "tables")
    width = max(len(str(r[0])) for r in (*rows, header))
    lines = ["per-section (fetched pages, and how many have each feature):"]
    lines.append(f"  {header[0]:<{width}}" + "".join(f"{h:>10}" for h in header[1:]))
    for row in rows:
        if row[0] == "total":
            lines.append("  " + "-" * (width + 50))
        lines.append(f"  {row[0]:<{width}}" + "".join(f"{n:>10}" for n in row[1:]))
    return lines


# -- checks --------------------------------------------------------------


def _check_paths(
    pages: list[Page], settings: Settings, raw, report: VerifyReport
) -> None:
    by_path: dict[str, list[str]] = {}
    for page in pages:
        by_path.setdefault(page.rel_path, []).append(page.url)
        try:
            expected = url_to_rel_path(page.url, settings.sources)
        except ValueError as e:
            report.violations.append(
                Finding("path-mismatch", page.url, f"URL cannot be stored: {e}")
            )
        else:
            if expected != page.rel_path:
                report.violations.append(
                    Finding(
                        "path-mismatch",
                        page.url,
                        f"rel_path is {page.rel_path!r} but the URL maps to {expected!r}",
                    )
                )
        if not (raw / page.rel_path).resolve().is_relative_to(raw):
            report.violations.append(
                Finding(
                    "path-escape",
                    page.url,
                    f"{page.rel_path!r} leaves the raw directory",
                )
            )
    for rel_path, urls in by_path.items():
        if len(urls) > 1:
            report.violations.append(
                Finding(
                    "duplicate-path",
                    rel_path,
                    f"{len(urls)} rows share this file: {', '.join(urls)}",
                )
            )


def _check_fetched_page(
    page: Page, settings: Settings, raw, report: VerifyReport
) -> None:
    report.checked += 1
    missing = [
        name
        for name in ("sha256", "size_bytes", "fetched_at", "checked_at")
        if getattr(page, name) in (None, "")
    ]
    if missing:
        report.violations.append(
            Finding("incomplete-row", page.url, f"ok row has no {', '.join(missing)}")
        )

    path = (raw / page.rel_path).resolve()
    if not path.is_relative_to(raw):
        return  # already reported as path-escape; never read outside the raw dir
    if not path.is_file():
        report.violations.append(
            Finding(
                "missing-file",
                page.url,
                f"{settings.raw_dir / page.rel_path} does not exist",
            )
        )
        return

    body = path.read_bytes()
    if page.sha256 and hashlib.sha256(body).hexdigest() != page.sha256:
        report.violations.append(
            Finding(
                "hash-mismatch",
                page.url,
                "the file's SHA-256 differs from the manifest",
            )
        )
    if page.size_bytes is not None and len(body) != page.size_bytes:
        report.violations.append(
            Finding(
                "size-mismatch",
                page.url,
                f"file is {len(body)} bytes, manifest says {page.size_bytes}",
            )
        )
    if problem := invalid_body(body):
        report.violations.append(Finding("bad-content", page.url, problem))

    _count_features(page, body, settings, report)


def _count_features(
    page: Page, body: bytes, settings: Settings, report: VerifyReport
) -> None:
    stats = report.sections.setdefault(
        section_of(page, settings.sources), SectionStats()
    )
    stats.pages += 1
    low = body.lower()
    opening = _ARTICLE_OPEN.search(low)
    if not opening:
        return  # no article: no feature counts (bad-content already reported)
    stats.article += 1
    closing = low.rfind(b"</article")
    region = low[opening.start() : closing if closing > opening.start() else len(low)]
    stats.h1 += bool(_H1.search(region))
    stats.code += bool(_CODE.search(region))
    stats.tables += bool(_TABLE.search(region))


def _check_stray_files(pages: list[Page], raw, report: VerifyReport) -> None:
    if not raw.is_dir():
        return
    known = {page.rel_path for page in pages}
    for path in sorted(p for p in raw.rglob("*") if p.is_file()):
        rel = path.relative_to(raw).as_posix()
        if path.name.startswith("."):
            report.warnings.append(
                Finding(
                    "stray-temp-file",
                    rel,
                    "left by an interrupted write; harmless, safe to delete",
                )
            )
        elif rel not in known:
            report.violations.append(
                Finding("orphan-file", rel, "no manifest row points at this file")
            )
