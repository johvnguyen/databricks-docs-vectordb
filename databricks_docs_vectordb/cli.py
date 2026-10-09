import argparse
import asyncio
import logging
import os
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import litellm
from dotenv import find_dotenv, load_dotenv
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from databricks_docs_vectordb.embedding.config import Settings as EmbeddingSettings
from databricks_docs_vectordb.embedding.provider import configured_model_info
from databricks_docs_vectordb.embedding.runner import Runner as EmbeddingRunner
from databricks_docs_vectordb.embedding.runner import RunSummary as EmbeddingSummary
from databricks_docs_vectordb.embedding.runner import select_candidates
from databricks_docs_vectordb.embedding.store import EmbeddingsStore, ModelInfo
from databricks_docs_vectordb.embedding.tokens import backfill_token_counts
from databricks_docs_vectordb.embedding.verify import (
    format_report as format_embed_verify,
)
from databricks_docs_vectordb.embedding.verify import verify_embeddings
from databricks_docs_vectordb.indexing.check import check_collection, embed_with
from databricks_docs_vectordb.indexing.check import format_report as format_index_check
from databricks_docs_vectordb.indexing.config import Settings as IndexingSettings
from databricks_docs_vectordb.indexing.runner import RunFailed, configured_dimensions
from databricks_docs_vectordb.indexing.runner import Runner as IndexRunner
from databricks_docs_vectordb.indexing.runner import RunSummary as IndexSummary
from databricks_docs_vectordb.indexing.store import QdrantStore, collection_name
from databricks_docs_vectordb.indexing.verify import (
    format_report as format_index_verify,
)
from databricks_docs_vectordb.indexing.verify import verify_collection
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

# 130 is the shell convention for "terminated by Ctrl-C". `provider_errors`
# is `embed run` only: repeated non-retryable errors (e.g. a bad API key).
EXIT_CODES = {
    "completed": 0,
    "interrupted": 130,
    "breaker_tripped": 1,
    "provider_errors": 1,
}


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
            "again: pages already current are skipped, not reparsed. Also "
            "backfills token_count for any chunk that doesn't have one yet, "
            "using the configured embedding model's tokenizer."
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

    embed = commands.add_parser(
        "embed", help="turn parsed chunks into embedding vectors"
    )
    embed_sub = embed.add_subparsers(dest="embed_command", required=True)

    backfill_tokens = embed_sub.add_parser(
        "backfill-tokens",
        help="fill in parsed.sqlite's token_count using the configured "
        "(or --model) tokenizer (no network, no re-chunking)",
        description=(
            "Compute a real token_count for every chunk not already tagged "
            "as tokenized by the target model's tiktoken encoding. A "
            "metadata fill-in only: parsed.sqlite's text and chunk "
            "boundaries are untouched. Safe to run again -- a chunk already "
            "tagged with the target model is left alone; `parse run` "
            "already does this for the configured model automatically, so "
            "with no --model this is just a manual escape hatch (e.g. after "
            "editing parsed.sqlite by hand). Pass --model to try a "
            "different tokenizer without touching your configured model: "
            "every chunk is re-tokenized under it and tagged accordingly, "
            "so switching back re-tokenizes again rather than reusing a "
            "stale count."
        ),
    )
    backfill_tokens.add_argument(
        "--model",
        metavar="MODEL",
        help="tokenize as this model instead of the configured one (e.g. "
        "to compare tokenizers); doesn't change your configured model",
    )

    embed_run = embed_sub.add_parser(
        "run",
        help="embed every chunk that needs it and store the vectors",
        description=(
            "Embed every chunk with no stored embedding, a changed text, a "
            "different configured model, or a failed earlier attempt -- all "
            "automatic, so there is no --retry-failed. Content-policy "
            "rejections are stored and not retried. Needs the provider's API "
            "key (for OpenAI, OPENAI_API_KEY in the environment or .env). "
            "Safe to interrupt with Ctrl-C (once: finish in-flight requests "
            "then stop; twice: stop now) and to run again: finished chunks "
            "are not embedded twice."
        ),
    )
    embed_size = embed_run.add_mutually_exclusive_group()
    embed_size.add_argument(
        "--limit",
        type=positive_int,
        metavar="N",
        help="embed at most N chunks, then stop (default: all)",
    )
    embed_size.add_argument(
        "--sample",
        type=positive_int,
        metavar="N",
        help="embed about N chunks spread across every section of every "
        "source (at least 2 per section), always including the longest and "
        "shortest chunk still to embed -- the pilot run. The same candidate "
        "chunks always give the same sample. Refuses an N too small to cover "
        "every section and says how many it needs",
    )
    embed_run.add_argument(
        "--rate",
        type=positive_float,
        metavar="R",
        help="requests per minute across all workers (default: 500)",
    )
    embed_run.add_argument(
        "--tpm",
        type=positive_float,
        metavar="T",
        help="input tokens per minute across all workers (default: 1000000); "
        "usually the tighter of the two limits",
    )
    embed_run.add_argument(
        "--concurrency",
        type=positive_int,
        metavar="C",
        help="requests in flight at once (default: 4)",
    )

    embed_sub.add_parser(
        "status",
        help="show what has been embedded, what failed or was rejected, and "
        "what's left to embed (no network)",
    )

    embed_verify = embed_sub.add_parser(
        "verify",
        help="check the embeddings store against parsed.sqlite and itself (no network)",
        description=(
            "Check every stored row set is well-formed (sub_index/sub_count, "
            "rows of a chunk agree, BLOB length matches dimensions) and "
            "belongs to a chunk still in parsed.sqlite (no orphans); report "
            "coverage by chunk and row counts by status and by provider/"
            "model/version. Reports every problem; exits 1 if there are any."
        ),
    )
    embed_verify.add_argument(
        "--complete",
        action="store_true",
        help="embedding should be finished: chunks never embedded, embedded "
        "under old text, or under a different model are violations "
        "(failed/rejected ones are not)",
    )

    index = commands.add_parser(
        "index", help="load the embeddings into the Qdrant vector database"
    )
    index_sub = index.add_subparsers(dest="index_command", required=True)

    index_run = index_sub.add_parser(
        "run",
        help="load the configured model's embeddings into Qdrant, page by page",
        description=(
            "Compare the configured model's embeddings with its Qdrant "
            "collection and bring the collection up to date: load new pages, "
            "replace changed ones, and remove pages that are gone or not ok "
            "in the manifest. Pages that are current are left alone, so a "
            "second run does nothing. Before replacing or removing anything, "
            "asks once whether to save a snapshot to data/qdrant_snapshots/ "
            "first; with no terminal, one of --save-old and --no-save-old is "
            "needed. Needs QDRANT_API_KEY. Safe to interrupt with Ctrl-C and "
            "to run again: the next run finishes what this one started."
        ),
    )
    index_run.add_argument(
        "--refresh",
        action="store_true",
        help="drop the collection and load every row again",
    )
    save = index_run.add_mutually_exclusive_group()
    save.add_argument(
        "--save-old",
        dest="save_old",
        action="store_const",
        const=True,
        help="answer the save question with yes, without asking",
    )
    save.add_argument(
        "--no-save-old",
        dest="save_old",
        action="store_const",
        const=False,
        help="answer the save question with no, without asking",
    )

    index_status = index_sub.add_parser(
        "status",
        help="show the collection, its point and page counts, and other "
        "models' collections",
    )
    index_status.add_argument(
        "--by-section",
        action="store_true",
        help="also break the counts down by source/section",
    )

    index_verify = index_sub.add_parser(
        "verify",
        help="check every point of the collection against the embeddings "
        "store, parsed.sqlite and the manifest",
        description=(
            "Check every point: its vector is its row's BLOB after "
            "normalization, its payload is its row's, its row is current (an "
            "ok row of the configured model, whose chunk is in parsed.sqlite "
            "and whose page is ok in the manifest), and the collection's "
            "settings are the configured ones and it is not red. Reports "
            "every problem; exits 1 if there are any. Needs QDRANT_API_KEY; "
            "changes nothing."
        ),
    )
    index_verify.add_argument(
        "--complete",
        action="store_true",
        help="loading should be finished: a current row with no point is a "
        "violation",
    )
    index_verify.add_argument(
        "--by-section",
        action="store_true",
        help="also break the counts down by source/section",
    )

    index_sub.add_parser(
        "check",
        help="run sanity checks on the collection and print a report for "
        "developing retrieval",
        description=(
            "Four sanity checks, which report rather than pass or fail: "
            "self-retrieval and index recall@10 (default search against exact) "
            "over a seeded sample of stored points, with a warning when recall "
            "is below 0.95; built-in questions embedded with the configured "
            "model, each hit read back from embeddings.sqlite; and a few of "
            "them again with a section filter and a url filter. The questions "
            "need the model's API key (for OpenAI, OPENAI_API_KEY in .env) and "
            "are skipped with a note without it. Needs QDRANT_API_KEY."
        ),
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
            case "embed":
                return main_embed(args)
            case "index":
                return main_index(args)
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
            # `run` also backfills token_count (Step 3), which needs the
            # configured embedding model's tokenizer -- parsing itself
            # still knows nothing about embedding, but loading this here
            # means `parse run` alone leaves parsed.sqlite with real token
            # counts, with no separate `embed backfill-tokens` step needed.
            return run_parse_run(settings, EmbeddingSettings.load(), args)
        case "status":
            return run_parse_status(settings)
        case "verify":
            return run_parse_verify(settings, args)
    return 2


def main_embed(args: argparse.Namespace) -> int:
    # Only `run` calls the provider, so only `run` needs an API key.
    settings = EmbeddingSettings.load(
        rate_rpm=getattr(args, "rate", None),
        rate_tpm=getattr(args, "tpm", None),
        concurrency=getattr(args, "concurrency", None),
    )
    match args.embed_command:
        case "backfill-tokens":
            return run_embed_backfill_tokens(settings, args)
        case "run":
            return run_embed_run(settings, args)
        case "status":
            return run_embed_status(settings)
        case "verify":
            return run_embed_verify(settings, args)
    return 2


def main_index(args: argparse.Namespace) -> int:
    # Every index command talks to Qdrant, so QDRANT_API_KEY is required.
    settings = IndexingSettings.load()
    embedding = EmbeddingSettings.load()
    model = configured_model_info(embedding.model)
    match args.index_command:
        case "run":
            return run_index_run(settings, model, args)
        case "status":
            return run_index_status(settings, model, args)
        case "verify":
            return run_index_verify(settings, model, args)
        case "check":
            return run_index_check(settings, model, embedding)
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
        run_logging(settings.logs_dir, "fetch") as log_path,
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
def run_logging(logs_dir: Path, name: str) -> Iterator[Path]:
    """Log to the console and to `data/logs/<name>-<date>.log` (`fetch`
    for `scrape fetch`, `embed` for `embed run`).

    The handlers hang off this package's logger only, so httpx's and
    LiteLLM's own lines never reach the file. Handlers and the level are
    restored afterwards.
    """
    package = logging.getLogger(PACKAGE_LOGGER)
    saved_level = package.level

    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{name}-{datetime.now(UTC).date().isoformat()}.log"
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


def run_parse_run(
    settings: ParsingSettings,
    embedding_settings: EmbeddingSettings,
    args: argparse.Namespace,
) -> int:
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
        # Every (re)written chunk starts with token_count = NULL (chunker.py
        # never sets it); fill it in here so a `parse run` alone leaves
        # parsed.sqlite ready for Phase 3, no separate `embed
        # backfill-tokens` step required (see cli.py's embed group, kept
        # around as a manual escape hatch, not a routine step).
        tokenized = backfill_token_counts(store, embedding_settings.model)

    parsed = sum(1 for r in results if r.outcome == "parsed")
    skipped = sum(1 for r in results if r.outcome == "skipped")
    chunks_written = sum(r.chunk_count for r in results if r.outcome == "parsed")
    print(
        f"{len(results)} page(s) considered ({already_current} already current): "
        f"{parsed} parsed, {skipped} skipped"
    )
    print(
        f"{chunks_written} chunk(s) (re)written, {tokenized} given a real "
        f"token_count (model: {embedding_settings.model})"
    )
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


# -- embed -------------------------------------------------------------------


def require_parsed(settings: EmbeddingSettings) -> None:
    if not settings.parsed_path.exists():
        raise CliError(
            f"no chunk store at {settings.parsed_path}; run `{PROG} parse run` first"
        )


def run_embed_backfill_tokens(
    settings: EmbeddingSettings, args: argparse.Namespace
) -> int:
    require_parsed(settings)
    model = args.model or settings.model
    with ChunkStore(settings.parsed_path) as store:
        updated = backfill_token_counts(store, model)
    if updated:
        print(f"{updated} chunk(s) given a real token_count (model: {model})")
    else:
        print(f"nothing to backfill; every chunk is already tagged with {model!r}")
    return 0


def load_provider_credentials(settings: EmbeddingSettings) -> None:
    """Put `.env` (searched from the cwd, like `Settings.load`) into
    `os.environ` so the provider SDK finds the API key, then fail before
    any request if the configured provider's keys are still missing.

    `import litellm` also loads a `.env`, but searching from its own install
    directory, which finds this repo's only because `.venv/` lives inside
    it -- this makes the lookup explicit. Real environment variables win
    over `.env` (no override).
    """
    load_dotenv(find_dotenv(usecwd=True), override=False)
    missing = missing_provider_keys(settings.model)
    if missing:
        raise CliError(
            f"{', '.join(missing)} not set (or blank), needed to call "
            f"{settings.model!r}; add it to .env (or export it) and run again"
        )


def missing_provider_keys(model: str) -> list[str]:
    """Treat a blank model API key as missing"""
    blank = {name: value for name, value in os.environ.items() if not value.strip()}
    for name in blank:
        del os.environ[name]
    try:
        return litellm.validate_environment(model).get("missing_keys") or []
    finally:
        os.environ.update(blank)


def run_embed_run(settings: EmbeddingSettings, args: argparse.Namespace) -> int:
    # Everything that can be checked without a request is checked first.
    require_parsed(settings)
    load_provider_credentials(settings)
    litellm.suppress_debug_info = True  # its "Give Feedback" banners on errors

    with (
        run_logging(settings.logs_dir, "embed") as log_path,
        ChunkStore(settings.parsed_path) as chunks,
        EmbeddingsStore(settings.embeddings_path) as embeddings,
    ):
        runner = EmbeddingRunner(settings, chunks, embeddings)
        summary = asyncio.run(runner.run(limit=args.limit, sample=args.sample))
        counts = embeddings.counts()
        failed, _ = embeddings.chunks_with_status("failed", limit=0)
        rejected, _ = embeddings.chunks_with_status("rejected", limit=0)

    print(f"embeddings: {settings.embeddings_path}")
    print(
        f"this run: {summary.processed}/{summary.selected} chunk(s) "
        f"({summary.outcomes['ok']} ok, {summary.outcomes['failed']} failed, "
        f"{summary.outcomes['rejected']} rejected), {summary.requests} request(s), "
        f"{summary.tokens_sent} token(s)"
    )
    print(
        f"stored: {counts.chunk_ids} chunk(s) in {counts.rows} row(s) "
        f"({counts.ok} ok, {counts.failed} failed, {counts.rejected} rejected)"
    )
    for line in embed_follow_up(summary, failed, rejected, settings):
        print(line)
    print(f"log: {log_path}")
    return EXIT_CODES[summary.status]


def embed_follow_up(
    summary: EmbeddingSummary, failed: int, rejected: int, settings: EmbeddingSettings
) -> list[str]:
    """What to do next, given how the run ended."""
    lines = []
    left = summary.unprocessed
    if summary.status == "interrupted":
        lines.append(
            f"interrupted: {left} chunk(s) not embedded yet; run the same "
            "command again to resume"
        )
    elif summary.status == "breaker_tripped":
        lines.append(
            "stopped: the provider kept failing (rate limits, timeouts, 5xx), "
            f"so the circuit breaker ended the run; {left} chunk(s) not "
            "embedded yet. Try again later, or lower --rate/--tpm"
        )
    elif summary.status == "provider_errors":
        lines.append(
            "stopped: repeated errors that retrying won't fix; check that the "
            f"API key for {settings.model!r} is valid (for OpenAI, "
            "OPENAI_API_KEY in .env) and the model name is right, or the "
            f"provider may be down. {left} chunk(s) not embedded yet"
        )
    for provider, model in sorted(summary.model_mismatches):
        lines.append(
            f"model name conflict: configured {settings.model!r} but the "
            f"provider reported {provider}/{model}; if that's expected, update "
            "`model` in databricks_docs_vectordb/embedding/config.py, or every "
            "run will re-embed these chunks"
        )
    if failed:
        lines.append(
            f"{failed} chunk(s) failed: `{PROG} embed status` shows why; "
            f"`{PROG} embed run` retries them automatically"
        )
    if rejected:
        lines.append(
            f"{rejected} chunk(s) rejected by the provider's content policy "
            f"(not retried): `{PROG} embed status` lists them"
        )
    return lines


def run_embed_status(settings: EmbeddingSettings) -> int:
    if not settings.embeddings_path.exists():
        print(
            f"no embeddings yet at {settings.embeddings_path}; run `{PROG} embed run`"
        )
        return 0
    with EmbeddingsStore(settings.embeddings_path) as embeddings:
        counts = embeddings.counts()
        by_model = embeddings.counts_by_model()
        problems = {
            status: embeddings.chunks_with_status(status, limit=STATUS_DETAIL_LIMIT)
            for status in ("failed", "rejected")
        }
        todo = None
        if settings.parsed_path.exists():
            with ChunkStore(settings.parsed_path) as chunks:
                todo = select_candidates(
                    chunks, embeddings, configured_model_info(settings.model)
                )

    print(f"embeddings: {settings.embeddings_path}")
    print(
        f"{counts.chunk_ids} chunk(s) in {counts.rows} row(s) "
        f"({counts.rows - counts.chunk_ids} extra from split oversized chunks)"
    )
    print(f"rows: {counts.ok} ok, {counts.failed} failed, {counts.rejected} rejected")
    print("by provider / model / version / dimensions:")
    for key, c in sorted(by_model.items(), key=lambda kv: tuple(map(str, kv[0]))):
        print(
            f"  {key.provider} / {key.model} / {key.model_version or '-'} / "
            f"{key.dimensions or '-'}: {c.chunk_ids} chunk(s), {c.rows} row(s) "
            f"({c.ok} ok, {c.failed} failed, {c.rejected} rejected)"
        )
    for status, (total, listed) in problems.items():
        if not total:
            continue
        shown = f", first {len(listed)}" if len(listed) < total else ""
        print(f"{status}: {total} chunk(s){shown}")
        for problem in listed:
            print(f"  {problem.chunk_id}: {problem.error or '(no error recorded)'}")
    if todo is not None:
        selected, reasons = todo
        if selected:
            why = ", ".join(
                f"{reasons[r]} {r}"
                for r in ("new", "changed", "model", "failed")
                if reasons[r]
            )
            print(
                f"{len(selected)} chunk(s) still to embed ({why}): `{PROG} embed run`"
            )
        else:
            print(f"every chunk in {settings.parsed_path} is embedded and current")
    return 0


def run_embed_verify(settings: EmbeddingSettings, args: argparse.Namespace) -> int:
    require_parsed(settings)
    model = configured_model_info(settings.model)
    with ChunkStore(settings.parsed_path) as chunks:
        # Opening the store would create it; with none yet, every chunk is
        # simply not embedded.
        if settings.embeddings_path.exists():
            with EmbeddingsStore(settings.embeddings_path) as embeddings:
                report = verify_embeddings(
                    chunks, embeddings, model, complete=args.complete
                )
        else:
            report = verify_embeddings(chunks, None, model, complete=args.complete)
    print(format_embed_verify(report, str(settings.embeddings_path)))
    return 0 if report.ok else 1


# -- index -------------------------------------------------------------------


def require_embedded(settings: IndexingSettings) -> None:
    for path, command in (
        (settings.embeddings_path, "embed run"),
        (settings.parsed_path, "parse run"),
        (settings.manifest_path, "scrape enumerate"),
    ):
        if not path.exists():
            raise CliError(f"no {path.name} at {path}; run `{PROG} {command}` first")


def run_index_run(
    settings: IndexingSettings, model: ModelInfo, args: argparse.Namespace
) -> int:
    require_embedded(settings)
    with (
        run_logging(settings.logs_dir, "index") as log_path,
        EmbeddingsStore(settings.embeddings_path) as embeddings,
        ChunkStore(settings.parsed_path) as chunks,
        Manifest(settings.manifest_path) as manifest,
        QdrantStore(settings) as qdrant,
    ):
        runner = IndexRunner(settings, model, embeddings, chunks, manifest, qdrant)
        summary = runner.run(refresh=args.refresh, save_old=args.save_old)

    if summary.collection:
        print(f"collection: {summary.collection} at {settings.url}")
    if summary.plan is not None:
        plan = summary.plan
        upserted = (
            f"{summary.written.points} point(s) upserted"
            if summary.written is not None
            else "write not finished"
        )
        print(
            f"this run: {upserted}; pages: {len(plan.new)} new, "
            f"{len(plan.changed)} changed, {len(plan.removed)} removed, "
            f"{len(plan.current)} current"
        )
    for path in summary.saved:
        print(f"saved: {path}")
    for line in index_follow_up(summary):
        print(line)
    print(f"log: {log_path}")
    return summary.exit_code


def index_follow_up(summary: IndexSummary) -> list[str]:
    """What to do next, given how the run ended."""
    lines = []
    if summary.status == "interrupted":
        lines.append(
            "interrupted: run the same command again to finish; every page is "
            "either current or replaced by the next run"
        )
    elif summary.status == "failed":
        lines.append(f"failed: {summary.error}")
    plan = summary.plan
    if plan is not None and plan.not_ok:
        lines.append(
            f"{len(plan.not_ok)} page(s) with embeddings left out: not ok in the "
            f"manifest (`{PROG} scrape status` shows why)"
        )
    if summary.deleted:
        lines.append(
            f"deleted other models' collection(s): {', '.join(summary.deleted)}"
        )
    kept = [
        c for c in (plan.old_collections if plan else ()) if c not in summary.deleted
    ]
    if kept:
        lines.append(f"other models' collection(s) left in place: {', '.join(kept)}")
    if summary.orphans:
        lines.append(
            f"{len(summary.orphans)} orphaned row(s) skipped, their chunk gone "
            f"from parsed.sqlite: `{PROG} embed verify` lists them"
        )
    return lines


def run_index_status(
    settings: IndexingSettings, model: ModelInfo, args: argparse.Namespace
) -> int:
    if not settings.embeddings_path.exists():
        print(
            f"no embeddings yet at {settings.embeddings_path}; run `{PROG} embed run`"
        )
        return 0
    with EmbeddingsStore(settings.embeddings_path) as embeddings:
        try:
            dimensions = configured_dimensions(embeddings, model)
        except RunFailed as e:
            raise CliError(str(e)) from e
    name = collection_name(settings.collection_prefix, model, dimensions)
    with qdrant_errors(settings), QdrantStore(settings) as qdrant:
        lines = index_status_lines(
            qdrant, settings, name, dimensions, by_section=args.by_section
        )
    print("\n".join(lines))
    return 0


@contextmanager
def qdrant_errors(settings: IndexingSettings) -> Iterator[None]:
    """Report a Qdrant error as a `CliError` naming the server. `status` and
    `verify` make one attempt, with no retries: they should answer at once."""
    try:
        yield
    except (ApiException, httpx.HTTPError) as e:
        cause = e.source if isinstance(e, ResponseHandlingException) else e
        raise CliError(
            f"Qdrant at {settings.url}: {type(cause).__name__}: {cause}"
        ) from e


def run_index_verify(
    settings: IndexingSettings, model: ModelInfo, args: argparse.Namespace
) -> int:
    require_embedded(settings)
    with (
        qdrant_errors(settings),
        EmbeddingsStore(settings.embeddings_path) as embeddings,
        ChunkStore(settings.parsed_path) as chunks,
        Manifest(settings.manifest_path) as manifest,
        QdrantStore(settings) as qdrant,
    ):
        try:
            report = verify_collection(
                qdrant,
                settings,
                model,
                embeddings,
                chunks,
                manifest,
                complete=args.complete,
            )
        except RunFailed as e:
            raise CliError(str(e)) from e
    print(format_index_verify(report, by_section=args.by_section))
    return 0 if report.ok else 1


def index_status_lines(
    qdrant: QdrantStore,
    settings: IndexingSettings,
    name: str,
    dimensions: int,
    *,
    by_section: bool,
) -> list[str]:
    lines = [f"collection: {name} at {settings.url}"]
    names = qdrant.collection_names()
    if name not in names:
        lines.append(f"not created yet; run `{PROG} index run`")
    else:
        points: dict[str, int] = {}  # section -> its points
        pages: dict[str, set[str]] = {}  # section -> its pages
        offset = None
        while True:
            batch, offset = qdrant.scroll(name, offset=offset, with_vectors=False)
            for point in batch:
                points[point.section] = points.get(point.section, 0) + 1
                pages.setdefault(point.section, set()).add(point.url)
            if offset is None:
                break
        total_points = sum(points.values())
        total_pages = len(set().union(*pages.values()))
        lines.append(f"{total_points} point(s) across {total_pages} page(s)")
        for mismatch in qdrant.settings_mismatches(name, dimensions=dimensions):
            lines.append(
                f"differs from the settings: {mismatch} (the next run rebuilds it)"
            )
        if by_section and points:
            width = max(len("source/section"), *map(len, points))
            lines.append("per-section (pages, points):")
            lines.append(f"  {'source/section':<{width}}{'pages':>10}{'points':>10}")
            for section in sorted(points):
                lines.append(
                    f"  {section:<{width}}{len(pages[section]):>10}"
                    f"{points[section]:>10}"
                )
            lines.append("  " + "-" * (width + 20))
            lines.append(f"  {'total':<{width}}{total_pages:>10}{total_points:>10}")
    prefix = f"{settings.collection_prefix}_"
    others = sorted(n for n in names if n.startswith(prefix) and n != name)
    if others:
        lines.append("other models' collections:")
        lines.extend(f"  {other}: {qdrant.count(other)} point(s)" for other in others)
    else:
        lines.append("other models' collections: none")
    return lines


def run_index_check(
    settings: IndexingSettings, model: ModelInfo, embedding: EmbeddingSettings
) -> int:
    if not settings.embeddings_path.exists():
        raise CliError(
            f"no {settings.embeddings_path.name} at {settings.embeddings_path}; "
            f"run `{PROG} embed run` first"
        )
    load_dotenv(find_dotenv(usecwd=True), override=False)
    litellm.suppress_debug_info = True
    missing = missing_provider_keys(embedding.model)
    embed = None if missing else embed_with(embedding.model)
    with (
        qdrant_errors(settings),
        EmbeddingsStore(settings.embeddings_path) as embeddings,
        QdrantStore(settings) as qdrant,
    ):
        try:
            report = check_collection(qdrant, settings, model, embeddings, embed=embed)
        except RunFailed as e:
            raise CliError(str(e)) from e
    print(format_index_check(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
