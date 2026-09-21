from collections import Counter

from databricks_docs_vectordb.scraping.manifest import STATUSES, Page


def format_counts(counts: Counter[str]) -> str:
    """`5839 pending, 12 ok`, in a fixed status order."""
    return ", ".join(f"{counts[s]} {s}" for s in STATUSES if counts[s]) or "empty"


def format_manifest(totals: dict[str, Counter[str]]) -> list[str]:
    """One line per source, then a grand total when there are several."""
    lines = [
        f"  {source}: {format_counts(counts)}" for source, counts in totals.items()
    ]
    if len(totals) > 1:
        overall: Counter[str] = Counter()
        for counts in totals.values():
            overall.update(counts)
        lines.append(f"  all: {format_counts(overall)}")
    return lines


def format_problems(
    label: str, pages: list[Page], total: int, *, with_error: bool = True
) -> list[str]:
    """A heading and one line per page, for the pages that need attention."""
    lines = [f"{label} ({total}):"]
    for page in pages:
        detail = ""
        if with_error and page.error:
            detail = "  " + page.error.replace("\n", " ")[:120]
        elif page.http_status:
            detail = f"  HTTP {page.http_status}"
        lines.append(f"  {page.url}{detail}")
    if total > len(pages):
        lines.append(f"  ... and {total - len(pages)} more")
    return lines
