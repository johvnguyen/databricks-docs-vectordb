"""Fixtures for the tests that run against the Qdrant sidecar (`network`)."""

import uuid
from pathlib import Path

import httpx
import pytest

from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.store import connect


@pytest.fixture
def settings() -> Settings:
    return Settings.load()


@pytest.fixture
def client(settings):
    """A client of its own, to read back what the code under test wrote."""
    c = connect(settings)
    yield c
    c.close()


@pytest.fixture
def name(client):
    """A throwaway collection name (Q24). Every collection whose name starts
    with it is deleted afterwards, whatever happens."""
    n = f"test_{uuid.uuid4().hex}"
    yield n
    for collection in client.get_collections().collections:
        if collection.name.startswith(n):
            client.delete_collection(collection.name)


@pytest.fixture
def restore(settings, client):
    """Upload a saved snapshot into a new collection, the way to restore one,
    and return that collection's point count."""

    def restore(path: Path, collection: str) -> int:
        with path.open("rb") as file:
            httpx.post(
                f"{settings.url}/collections/{collection}/snapshots/upload",
                params={"wait": "true"},
                files={"snapshot": (path.name, file)},
                headers={"api-key": settings.api_key},
                timeout=120,
            ).raise_for_status()
        return client.count(collection, exact=True).count

    return restore
