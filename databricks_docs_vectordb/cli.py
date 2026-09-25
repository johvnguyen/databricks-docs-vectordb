import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx

from databricks_docs_vectordb.parsing.config import Settings as ParsingSettings
from databricks_docs_vectordb.parsing.pipeline import is_current, parse_pages
from databricks_docs_vectordb.parsing.report import (
    format_sections as format_parse_sections,
)
from databricks_docs_vectordb.parsing.report import overall_stats, section_stats
from databricks_docs_vectordb.parsing.store import ChunkStore
from databricks_docs_vectordb.parsing.verify import format_report as format_parse_verify
from databricks_docs_vectordb.parsing.verify import verify_store
from databricks_docs_vectordb.scraping.config import SOURCES
from databricks_docs_vectordb.scraping.config import Settings as ScrapingSettings
from databricks_docs_vectordb.scraping.enumeration import (
    enumerate_sources,
    format_report,
)
from databricks_docs_vectordb.scraping.manifest import Manifest, Page
from databricks_docs_vectordb.scraping.report import (
    format_manifest,
    format_problems,
)
from databricks_docs_vectordb.scraping.robots import RobotsRules
from databricks_docs_vectordb.scraping.runner import Runner, RunSummary, make_client
from databricks_docs_vectordb.scraping.sampling import sample_pages
from databricks_docs_vectordb.scraping.verify import format_report as format_verify
from databricks_docs_vectordb.scraping.verify import verify_snapshot

PROG = "databricks-docs-vectordb"
PACKAGE_LOGGER = "databricks_docs_vectordb"

# 130 is the shell convention for "terminated by Ctrl-C".
EXIT_CODES = {"completed": 0, "interrupted": 130, "breaker_tripped": 1}


class CliError(RuntimeError):
    """A problem the user can fix; reported as `error: ...` with exit code 1."""


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not value > 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG)
    commands = parser.add_subparsers(dest="command", required=True)

    scrape = commands.add_parser("scrape", help="collect the Databricks docs")
    sub = scrape.add_subparsers(dest="scrape_command", required=True)

    sub.add_parser(
        "enumerate",
        help="fetch robots.txt and the sitemaps, and update the manifest",
    )

    fetch = sub.add_parser(
        "fetch",
        help="download the pages the manifest lists as pending",
        description=(
            "Download pending pages, politely and resumably. Safe to interrupt "
            "with Ctrl-C (once: finish in-flight requests then stop; twice: stop "
            "now) and to run again: finished pages are not fetched twice."
        ),
    )
    size = fetch.add_mutually_exclusive_group()
    size.add_argument(
        "--limit",
        type=positive_int,
        metavar="N",
        help="fetch at most N pages, then stop (default: all)",
    )
    size.add_argument(
        "--sample",
        type=positive_int,
        metavar="N",
        help="fetch about N pages spread across every section of every source "
        "(at least 2 per section, a few more from the largest); the same "
        "pending pages always give the same sample. Refuses an N too small to "
        "cover every section and says how many it needs",
    )
    fetch.add_argument(
        "--refresh",
        action="store_true",
        help="also re-check pages already fetched, with conditional requests",
    )
    fetch.add_argument(
        "--rate",
        type=positive_float,
        metavar="R",
        help="requests per second across all workers (default: 2)",
    )
    fetch.add_argument(
        "--concurrency",
        type=positive_int,
        metavar="C",
        help="requests in flight at once (default: 4)",
    )
    fetch.add_argument(
        "--retry-failed",
        action="store_true",
        help="put pages that failed earlier back in the queue first",
    )

    sub.add_parser(
        "status",
        help="show what has been fetched, and what failed (no network)",
    )

    verify = sub.add_parser(
        "verify",
        help="check the saved pages against the manifest (no network)",
        description=(
            "Check every fetched page: the file exists, matches its recorded "
            "hash and size, and looks like a docs page; and that no file is "
            "unaccounted for. Reports every problem; exits 1 if there are any."
        ),
    )
    verify.add_argument(
        "--by-section",
        action="store_true",
        help="also print a table of page features (<article, <h1>, code, tables) "
        "per source and top-level section",
    )
    verify.add_argument(
        "--complete",
        action="store_true",
        help="the crawl should be finished: pages still pending are violations",
    )

    parse = commands.add_parser(
        "parse", help="turn the raw HTML snapshot into chunked markdown"
    )
    parse_sub = parse.add_subparsers(dest="parse_command", required=True)

    run = parse_sub.add_parser(
        "run",
        help="parse pending/changed pages into the chunk store",
        description=(
            "Parse every ok manifest row whose chunks aren't already current "
            "(a matching sha256), or every ok row with --refresh. Safe to run "
            "again: pages already current are skipped, not reparsed."
        ),
    )
    run_size = run.add_mutually_exclusive_group()
    run_size.add_argument(
        "--limit",
        type=positive_int,
        metavar="N",
        help="parse at most N pages, then stop (default: all)",
    )
    run_size.add_argument(
        "--sample",
        type=positive_int,
        metavar="N",
        help="parse about N pages spread across every section of every source "
        "(at least 2 per section, a few more from the largest); the same "
        "candidate pages always give the same sample. Refuses an N too small "
        "to cover every section and says how many it needs",
    )
    run.add_argument(
        "--refresh",
        action="store_true",
        help="reparse pages even if their chunks already match the current "
        "sha256 (e.g. after a chunking-rule or target-size change)",
    )

    parse_sub.add_parser(
        "status",
        help="show chunk counts by source/section, size distribution, and "
        "the oversized-chunk count (no network, no re-parsing)",
    )

    parse_verify = parse_sub.add_parser(
        "verify",
        help="check the chunk store against the manifest and itself (no network)",
        description=(
            "Check every ok manifest page has chunks, no chunk is empty, no "
            "chunk exceeds the target size unless flagged oversized, and every "
            "chunk's prev/next neighbor resolves within its own page. Reports "
            "every problem; exits 1 if there are any."
        ),
    )
    parse_verify.add_argument(
        "--by-section",
        action="store_true",
        help="also print the per-source/section chunk-size table "
        "(the same one `parse status` prints)",
    )
    parse_verify.add_argument(
        "--complete",
        action="store_true",
        help="parsing should be finished: ok pages with no chunks yet are violations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        match args.command:
            case "scrape":
                return main_scrape(args)
            case "parse":
                return main_parse(args)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 2


def main_scrape(args: argparse.Namespace) -> int:
    # Only commands that talk to Databricks need the contact address.
    settings = ScrapingSettings.load(
        require_contact=args.scrape_command in ("enumerate", "fetch"),
        rate=getattr(args, "rate", None),
        concurrency=getattr(args, "concurrency", None),
    )
    match args.scrape_command:
        case "enumerate":
            return run_enumerate(settings)
        case "fetch":
            return run_fetch(settings, args)
        case "status":
            return run_status(settings)
        case "verify":
            return run_verify(settings, args)
    return 2


def main_parse(args: argparse.Namespace) -> int:
    # Parsing never touches the network, so there's no contact address to load.
    settings = ParsingSettings.load()
    match args.parse_command:
        case "run":
            return run_parse_run(settings, args)
        case "status":
            return run_parse_status(settings)
        case "verify":
            return run_parse_verify(settings, args)
    return 2


# -- enumerate -----------------------------------------------------------


def run_enumerate(settings: ScrapingSettings) -> int:
    client = httpx.Client(
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
        max_redirects=settings.max_redirects,
        timeout=httpx.Timeout(settings.read_timeout, connect=settings.connect_timeout),
        transport=httpx.HTTPTransport(retries=3),
    )
    with client, Manifest(settings.manifest_path) as manifest:
        result = enumerate_sources(settings, client, manifest)
    print(format_report(result))
    return 0


# -- fetch ---------------------------------------------------------------


def run_fetch(settings: ScrapingSettings, args: argparse.Namespace) -> int:
    # Everything that can be checked without a request is checked first.
    require_enumerated(settings)
    robots = RobotsRules(settings.robots_path.read_text())

    with (
        fetch_logging(settings) as log_path,
        Manifest(settings.manifest_path) as manifest,
    ):
        if args.retry_failed:
            requeued = manifest.requeue_failed()
            logging.getLogger(PACKAGE_LOGGER).info(
                "requeued %d failed page%s", requeued, "" if requeued == 1 else "s"
            )
        pages = choose_sample(settings, manifest, args) if args.sample else None
        summary = asyncio.run(crawl(settings, manifest, robots, args, pages))
        totals = manifest.counts_by_source()
        failed = manifest.counts()["failed"]

    print("manifest:")
    print("\n".join(format_manifest(totals)))
    for line in follow_up(summary, failed):
        print(line)
    print(f"log: {log_path}")
    return EXIT_CODES[summary.status]


def choose_sample(
    settings: ScrapingSettings, manifest: Manifest, args: argparse.Namespace
) -> list[Page]:
    """The pages `--sample N` will fetch, drawn from what would be fetched anyway."""
    candidates = manifest.next_batch(refresh=args.refresh)
    sample = sample_pages(candidates, settings.sources, args.sample)
    logging.getLogger(PACKAGE_LOGGER).info(
        "sample: %d of %d candidate pages, from %d sections (%d needed for 2 per section)",
        len(sample.pages),
        sample.candidates,
        sample.sections,
        sample.minimum,
    )
    return sample.pages


async def crawl(
    settings: ScrapingSettings,
    manifest: Manifest,
    robots: RobotsRules,
    args: argparse.Namespace,
    pages: list[Page] | None = None,
) -> RunSummary:
    async with make_client(settings) as client:
        runner = Runner(client, settings, manifest, robots)
        return await runner.run(refresh=args.refresh, limit=args.limit, pages=pages)


def follow_up(summary: RunSummary, failed: int) -> list[str]:
    """What to do next, given how the run ended."""
    lines = []
    if summary.status == "interrupted":
        lines.append(
            f"interrupted: {summary.pending_left} pages are still pending; "
            f"run the same command again to resume"
        )
    elif summary.status == "breaker_tripped":
        lines.append(
            "stopped: the site kept failing, so the circuit breaker ended the run; "
            f"{summary.pending_left} pages are still pending. Try again later"
        )
    if failed:
        lines.append(
            f"{failed} page{'s' if failed != 1 else ''} failed: "
            f"`{PROG} scrape status` shows why, and "
            f"`{PROG} scrape fetch --retry-failed` tries them again"
        )
    return lines


def require_enumerated(settings: ScrapingSettings) -> None:
    if not settings.manifest_path.exists() or not settings.robots_path.exists():
        raise CliError(
            f"nothing to fetch yet ({settings.manifest_path} or {settings.robots_path} "
            f"is missing); run `{PROG} scrape enumerate` first"
        )


@contextmanager
def fetch_logging(settings: ScrapingSettings) -> Iterator[Path]:
    """Log to the console and to `data/logs/fetch-<date>.log`.

    The handlers hang off this package's logger only, so httpx's own
    per-request lines never reach the file. Handlers and the level are
    restored afterwards.
    """
    package = logging.getLogger(PACKAGE_LOGGER)
    saved_level = package.level

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    path = settings.logs_dir / f"fetch-{datetime.now(UTC).date().isoformat()}.log"
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ"
    )
    formatter.converter = time.gmtime
    handlers: list[logging.Handler] = [
        logging.FileHandler(path, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        package.addHandler(handler)
    package.setLevel(logging.INFO)
    try:
        yield path
    finally:
        for handler in handlers:
            package.removeHandler(handler)
            handler.close()
        package.setLevel(saved_level)


# -- status --------------------------------------------------------------

STATUS_DETAIL_LIMIT = 10


def run_status(settings: ScrapingSettings) -> int:
    if not settings.manifest_path.exists():
        raise CliError(
            f"no manifest at {settings.manifest_path}; run `{PROG} scrape enumerate` first"
        )
    with Manifest(settings.manifest_path) as manifest:
        totals = manifest.counts_by_source()
        counts = manifest.counts()
        if not totals:
            print(f"the manifest is empty; run `{PROG} scrape enumerate`")
            return 0
        print(f"manifest: {settings.manifest_path}")
        print("\n".join(format_manifest(totals)))
        for status in ("failed", "gone"):
            if counts[status]:
                pages = manifest.pages_with_status(status, limit=STATUS_DETAIL_LIMIT)
                print("\n".join(format_problems(status, pages, counts[status])))
        if counts["pending"]:
            print(f"{counts['pending']} pages still to fetch: `{PROG} scrape fetch`")
    return 0


# -- verify --------------------------------------------------------------


def run_verify(settings: ScrapingSettings, args: argparse.Namespace) -> int:
    if not settings.manifest_path.exists():
        raise CliError(
            f"no manifest at {settings.manifest_path}; run `{PROG} scrape enumerate` first"
        )
    with Manifest(settings.manifest_path) as manifest:
        report = verify_snapshot(settings, manifest, complete=args.complete)
    print(format_verify(report, settings, by_section=args.by_section))
    return 0 if report.ok else 1


# -- parse -----------------------------------------------------------------


def require_scraped(settings: ParsingSettings) -> None:
    if not settings.manifest_path.exists():
        raise CliError(
            f"no manifest at {settings.manifest_path}; run `{PROG} scrape enumerate` "
            f"(and `{PROG} scrape fetch`) first"
        )


def run_parse_run(settings: ParsingSettings, args: argparse.Namespace) -> int:
    require_scraped(settings)
    with (
        Manifest(settings.manifest_path) as manifest,
        ChunkStore(settings.parsed_path) as store,
    ):
        ok_pages = manifest.pages_with_status("ok")
        candidates = (
            ok_pages
            if args.refresh
            else [p for p in ok_pages if not is_current(p, store)]
        )
        already_current = len(ok_pages) - len(candidates)

        if args.sample:
            sample = sample_pages(candidates, SOURCES, args.sample)
            pages = sample.pages
            logging.getLogger(PACKAGE_LOGGER).info(
                "sample: %d of %d candidate pages, from %d sections "
                "(%d needed for 2 per section)",
                len(pages),
                sample.candidates,
                sample.sections,
                sample.minimum,
            )
        else:
            pages = candidates
        if args.limit:
            pages = pages[: args.limit]

        results = list(
            parse_pages(
                pages,
                raw_dir=settings.raw_dir,
                store=store,
                settings=settings,
                force=args.refresh,
            )
        )

    parsed = sum(1 for r in results if r.outcome == "parsed")
    skipped = sum(1 for r in results if r.outcome == "skipped")
    chunks_written = sum(r.chunk_count for r in results if r.outcome == "parsed")
    print(
        f"{len(results)} page(s) considered ({already_current} already current): "
        f"{parsed} parsed, {skipped} skipped"
    )
    print(f"{chunks_written} chunk(s) (re)written")
    return 0


def run_parse_status(settings: ParsingSettings) -> int:
    if not settings.parsed_path.exists():
        print(f"no chunk store yet at {settings.parsed_path}; run `{PROG} parse run`")
        return 0
    with ChunkStore(settings.parsed_path) as store:
        counts = store.counts()
        if counts.chunks == 0:
            print(f"the chunk store is empty; run `{PROG} parse run`")
            return 0
        print(f"chunk store: {settings.parsed_path}")
        print(
            f"{counts.pages} page(s), {counts.chunks} chunk(s), "
            f"{counts.oversized} oversized"
        )
        summaries = store.all_summaries()
        sections = section_stats(summaries, SOURCES)
    print("\n".join(format_parse_sections(sections, overall_stats(summaries))))
    return 0


def run_parse_verify(settings: ParsingSettings, args: argparse.Namespace) -> int:
    require_scraped(settings)
    with (
        Manifest(settings.manifest_path) as manifest,
        ChunkStore(settings.parsed_path) as store,
    ):
        report = verify_store(
            manifest, store, settings, SOURCES, complete=args.complete
        )
    print(format_parse_verify(report, settings, by_section=args.by_section))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
