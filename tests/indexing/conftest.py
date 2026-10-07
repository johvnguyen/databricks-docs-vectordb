"""Fixtures for the tests that run against the Qdrant sidecar (`network`)."""

import uuid

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
    """A throwaway collection (Q24), deleted afterwards whatever happens."""
    n = f"test_{uuid.uuid4().hex}"
    yield n
    client.delete_collection(n)
