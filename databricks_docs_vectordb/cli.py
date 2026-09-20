import argparse
import sys
from collections.abc import Sequence

import httpx

from databricks_docs_vectordb.scraping.config import Settings
from databricks_docs_vectordb.scraping.enumeration import (
    enumerate_sources,
    format_report,
)
from databricks_docs_vectordb.scraping.manifest import Manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="databricks-docs-vectordb")
    commands = parser.add_subparsers(dest="command", required=True)

    scrape = commands.add_parser("scrape", help="collect the Databricks docs")
    scrape_commands = scrape.add_subparsers(dest="scrape_command", required=True)
    scrape_commands.add_parser(
        "enumerate",
        help="fetch robots.txt and the sitemaps, and update the manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.load()
        if (args.command, args.scrape_command) == ("scrape", "enumerate"):
            return run_enumerate(settings)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 2


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


if __name__ == "__main__":
    raise SystemExit(main())
