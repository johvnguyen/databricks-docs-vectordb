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

from databricks_docs_vectordb.scraping.config import Settings
from databricks_docs_vectordb.scraping.enumeration import (
    enumerate_sources,
    format_report,
)
from databricks_docs_vectordb.scraping.manifest import Manifest
from databricks_docs_vectordb.scraping.report import (
    format_manifest,
    format_problems,
)
from databricks_docs_vectordb.scraping.robots import RobotsRules
from databricks_docs_vectordb.scraping.runner import Runner, RunSummary, make_client
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
    fetch.add_argument(
        "--limit",
        type=positive_int,
        metavar="N",
        help="fetch at most N pages, then stop (default: all)",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Only commands that talk to Databricks need the contact address.
        settings = Settings.load(
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
    except (RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 2


# -- enumerate -----------------------------------------------------------


def run_enumerate(settings: Settings) -> int:
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


def run_fetch(settings: Settings, args: argparse.Namespace) -> int:
    # Everything that can be checked without a request is checked first.
    require_enumerated(settings)
    robots = RobotsRules(settings.robots_path.read_text())

    with fetch_logging(settings) as log_path, Manifest(
        settings.manifest_path
    ) as manifest:
        if args.retry_failed:
            requeued = manifest.requeue_failed()
            logging.getLogger(PACKAGE_LOGGER).info(
                "requeued %d failed page%s", requeued, "" if requeued == 1 else "s"
            )
        summary = asyncio.run(crawl(settings, manifest, robots, args))
        totals = manifest.counts_by_source()
        failed = manifest.counts()["failed"]

    print("manifest:")
    print("\n".join(format_manifest(totals)))
    for line in follow_up(summary, failed):
        print(line)
    print(f"log: {log_path}")
    return EXIT_CODES[summary.status]


async def crawl(
    settings: Settings,
    manifest: Manifest,
    robots: RobotsRules,
    args: argparse.Namespace,
) -> RunSummary:
    async with make_client(settings) as client:
        runner = Runner(client, settings, manifest, robots)
        return await runner.run(refresh=args.refresh, limit=args.limit)


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


def require_enumerated(settings: Settings) -> None:
    if not settings.manifest_path.exists() or not settings.robots_path.exists():
        raise CliError(
            f"nothing to fetch yet ({settings.manifest_path} or {settings.robots_path} "
            f"is missing); run `{PROG} scrape enumerate` first"
        )


@contextmanager
def fetch_logging(settings: Settings) -> Iterator[Path]:
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


def run_status(settings: Settings) -> int:
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


def run_verify(settings: Settings, args: argparse.Namespace) -> int:
    if not settings.manifest_path.exists():
        raise CliError(
            f"no manifest at {settings.manifest_path}; run `{PROG} scrape enumerate` first"
        )
    with Manifest(settings.manifest_path) as manifest:
        report = verify_snapshot(settings, manifest, complete=args.complete)
    print(format_verify(report, settings, by_section=args.by_section))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
