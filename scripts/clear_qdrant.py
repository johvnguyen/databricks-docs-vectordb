"""Delete every collection in the Qdrant database, through its API.

`scripts/clean_data.sh` calls this to reset Qdrant along with data/; it
asks first, and this doesn't. QDRANT_API_KEY and QDRANT_URL are read like
every `index` command reads them: the environment, then `.env`.

    uv run python scripts/clear_qdrant.py --dry-run   # list, delete nothing
    uv run python scripts/clear_qdrant.py

Exits 1 if the key is missing or Qdrant doesn't answer.
"""

import argparse
import sys
from collections.abc import Sequence

import httpx
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.store import QdrantStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clear_qdrant.py",
        description="Delete every collection in the Qdrant database.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the collections, with their point counts, and delete nothing",
    )
    args = parser.parse_args(argv)

    try:
        settings = Settings.load()
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        with QdrantStore(settings) as qdrant:
            names = sorted(qdrant.collection_names())
            if not names:
                print(f"Qdrant at {settings.url} has no collections")
                return 0
            if args.dry_run:
                print(f"collections in Qdrant at {settings.url}:")
                for name in names:
                    print(f"  {name}: {qdrant.count(name)} point(s)")
                return 0
            for name in names:
                qdrant.delete_collection(name)
                print(f"deleted the collection {name}")
    except (ApiException, httpx.HTTPError) as e:
        cause = e.source if isinstance(e, ResponseHandlingException) else e
        print(
            f"error: Qdrant at {settings.url}: {type(cause).__name__}: {cause}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
