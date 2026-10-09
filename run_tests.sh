#!/usr/bin/env bash
# Runs every test, the default ones and the `network` ones, from inside the
# dev container. Manages Qdrant collections, not containers: waits for the
# sidecar to answer at QDRANT_URL first, and deletes any leftover test_
# collections afterwards, pass or fail. Extra arguments go to pytest.
set -uo pipefail
cd "$(dirname "$0")"

QDRANT_WAIT_SECONDS="${QDRANT_WAIT_SECONDS:-60}"

# QDRANT_URL and QDRANT_API_KEY come from Settings.load, so the environment
# and .env are read the same way the indexing code reads them.
qdrant() {
  uv run python - "$@" <<'PY'
import sys
import time

import httpx

from databricks_docs_vectordb.indexing.config import Settings

try:
    settings = Settings.load()
except (RuntimeError, ValueError) as e:
    sys.exit(f"run_tests.sh: {e}")

client = httpx.Client(
    base_url=settings.url, headers={"api-key": settings.api_key}, timeout=5.0
)


def wait(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        try:
            response = client.get("/collections")
        except httpx.TransportError:
            response = None
        if response is not None:
            if response.status_code in (401, 403):
                sys.exit(
                    f"run_tests.sh: Qdrant at {settings.url} rejected "
                    f"QDRANT_API_KEY (HTTP {response.status_code})"
                )
            if response.is_success:
                return
        if time.monotonic() >= deadline:
            sys.exit(
                f"run_tests.sh: Qdrant did not answer at {settings.url} "
                f"within {seconds:g} s"
            )
        time.sleep(1)


def cleanup() -> None:
    try:
        response = client.get("/collections")
        response.raise_for_status()
        for collection in response.json()["result"]["collections"]:
            name = collection["name"]
            if name.startswith("test_"):
                client.delete(f"/collections/{name}").raise_for_status()
                print(f"run_tests.sh: deleted leftover collection {name}")
    except httpx.HTTPError as e:
        sys.exit(f"run_tests.sh: could not delete leftover test_ collections: {e}")


command = sys.argv[1]
if command == "wait":
    wait(float(sys.argv[2]))
elif command == "cleanup":
    cleanup()
PY
}

qdrant wait "$QDRANT_WAIT_SECONDS" || exit 1

uv run pytest -m "network or not network" "$@"
status=$?

if ! qdrant cleanup && [ "$status" -eq 0 ]; then
  status=1
fi
exit "$status"
