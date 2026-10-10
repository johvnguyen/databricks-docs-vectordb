"""Save a snapshot of this project's Qdrant collections to data/qdrant_snapshots/.

Each collection is snapshotted on the server, downloaded, checked against
the server's sha256, and the server's copy deleted: the same save `index run`
offers before it replaces anything (`QdrantStore.save_snapshot`). Needs
QDRANT_API_KEY, read like every `index` command reads it (the environment,
then `.env`).

    uv run python scripts/snapshot_qdrant.py
    uv run python scripts/snapshot_qdrant.py --collection NAME [--collection NAME]
    uv run python scripts/snapshot_qdrant.py --out data/backups

With no --collection, every `databricks_docs_*` collection is saved: one per
embedding model, so usually one. To restore a snapshot into a collection:

    curl -X POST "http://qdrant:6333/collections/<name>/snapshots/upload?wait=true" \\
      -H "api-key: $QDRANT_API_KEY" \\
      -F "snapshot=@data/qdrant_snapshots/<file>.snapshot"
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.store import QdrantStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="snapshot_qdrant.py",
        description="Save a snapshot of this project's Qdrant collections.",
    )
    parser.add_argument(
        "--collection",
        action="append",
        metavar="NAME",
        help="save this collection (repeatable; default: every "
        "databricks_docs_* collection)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="where to save the snapshots (default: data/qdrant_snapshots/)",
    )
    args = parser.parse_args(argv)

    try:
        settings = Settings.load()
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    out = args.out or settings.snapshots_dir

    try:
        with QdrantStore(settings) as qdrant:
            names = qdrant.collection_names()
            prefix = f"{settings.collection_prefix}_"
            chosen = args.collection or sorted(n for n in names if n.startswith(prefix))
            unknown = [n for n in chosen if n not in names]
            if unknown:
                print(
                    f"error: no such collection: {', '.join(unknown)}", file=sys.stderr
                )
                return 1
            if not chosen:
                print(
                    f"error: no {prefix}* collections at {settings.url}; "
                    "run `databricks-docs-vectordb index run` first",
                    file=sys.stderr,
                )
                return 1
            for name in chosen:
                print(f"saving {name} ({qdrant.count(name)} points) ...")
                path = qdrant.save_snapshot(name, out)
                print(f"saved {path} ({path.stat().st_size / 1e6:.0f} MB)")
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
