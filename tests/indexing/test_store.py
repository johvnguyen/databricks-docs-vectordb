import array
import uuid
from dataclasses import replace

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from databricks_docs_vectordb.embedding.config import Settings as EmbeddingSettings
from databricks_docs_vectordb.embedding.provider import configured_model_info
from databricks_docs_vectordb.embedding.store import ModelInfo
from databricks_docs_vectordb.indexing import store as store_module
from databricks_docs_vectordb.indexing.store import (
    KEYWORD_INDEXES,
    SPARSE_VECTOR_NAME,
    QdrantStore,
    collection_name,
    connect,
)

# -- collection_name: no server needed ----------------------------------------


def test_collection_name_for_the_configured_model():
    model = configured_model_info(EmbeddingSettings().model)
    assert (
        collection_name("databricks_docs", model, 1536)
        == "databricks_docs_openai_text_embedding_3_small_1536"
    )


def test_collection_name_lowercases_and_collapses_other_characters():
    model = ModelInfo("Azure", "My--Deployment.V2/x", None)
    assert (
        collection_name("databricks_docs", model, 3072)
        == "databricks_docs_azure_my_deployment_v2_x_3072"
    )


def test_model_version_is_not_part_of_the_name():
    unversioned = ModelInfo("openai", "text-embedding-3-small", None)
    versioned = unversioned._replace(model_version="2024-01-25")
    assert collection_name("p", versioned, 1536) == collection_name(
        "p", unversioned, 1536
    )


def test_dimensions_are_part_of_the_name():
    """OpenAI's v3 models can return fewer than their native dimensions."""
    model = ModelInfo("openai", "text-embedding-3-small", None)
    assert collection_name("p", model, 512) != collection_name("p", model, 1536)


@pytest.mark.parametrize("dimensions", [0, -1])
def test_nonpositive_dimensions_are_rejected(dimensions):
    model = ModelInfo("openai", "text-embedding-3-small", None)
    with pytest.raises(ValueError, match="dimensions"):
        collection_name("p", model, dimensions)


# -- against the sidecar (fixtures in conftest.py) ------------------------------


@pytest.fixture
def store(settings):
    with QdrantStore(settings) as s:
        yield s


@pytest.mark.network
@pytest.mark.parametrize("distance", ["Cosine", "Dot"])
def test_created_collection_reads_back_as_designed(settings, client, name, distance):
    with QdrantStore(replace(settings, distance=distance)) as store:
        store.create_collection(name, dimensions=8)

    info = client.get_collection(name)
    params = info.config.params
    assert isinstance(params.vectors, models.VectorParams)  # unnamed dense vector
    assert params.vectors.size == 8
    assert params.vectors.distance == models.Distance(distance)
    assert not params.vectors.on_disk
    assert params.on_disk_payload is True

    hnsw = info.config.hnsw_config
    assert (hnsw.m, hnsw.ef_construct, hnsw.full_scan_threshold) == (16, 100, 10_000)
    assert hnsw.on_disk is False

    quantization = info.config.quantization_config
    assert isinstance(quantization, models.ScalarQuantization)
    assert quantization.scalar.type == models.ScalarType.INT8
    assert quantization.scalar.always_ram is True

    assert set(params.sparse_vectors or {}) == {SPARSE_VECTOR_NAME}
    assert {field: index.data_type for field, index in info.payload_schema.items()} == {
        field: models.PayloadSchemaType.KEYWORD for field in KEYWORD_INDEXES
    }
    assert info.points_count == 0


@pytest.mark.network
def test_points_need_only_the_dense_vector(store, client, name):
    """The sparse vector is declared but stays empty."""
    store.create_collection(name, dimensions=2)
    client.upsert(
        name,
        points=[
            models.PointStruct(id=1, vector=[1.0, 0.0], payload={"section": "a"}),
            models.PointStruct(id=2, vector=[0.0, 1.0], payload={"section": "b"}),
        ],
        wait=True,
    )
    hits = client.query_points(name, query=[0.9, 0.1], limit=1).points
    assert [hit.id for hit in hits] == [1]


@pytest.mark.network
def test_creating_an_existing_collection_raises_and_keeps_it(store, client, name):
    store.create_collection(name, dimensions=4)
    with pytest.raises(UnexpectedResponse):
        store.create_collection(name, dimensions=8)
    assert client.get_collection(name).config.params.vectors.size == 4


@pytest.mark.network
def test_a_failed_payload_index_leaves_no_collection(store, client, name, monkeypatch):
    def fail(self, *args, **kwargs):
        raise ConnectionError("sidecar went away")

    monkeypatch.setattr(QdrantClient, "create_payload_index", fail)
    with pytest.raises(ConnectionError):
        store.create_collection(name, dimensions=4)
    assert not client.collection_exists(name)


@pytest.mark.network
def test_connect_does_not_warn_about_plain_http(settings, recwarn):
    connect(settings).close()
    assert not [w for w in recwarn if "insecure" in str(w.message)]


@pytest.mark.network
def test_collection_exists_and_collection_names(store, name):
    assert not store.collection_exists(name)
    assert name not in store.collection_names()
    store.create_collection(name, dimensions=2)
    assert store.collection_exists(name)
    assert name in store.collection_names()


@pytest.mark.network
def test_scroll_pages_through_every_point(store, client, name, monkeypatch):
    monkeypatch.setattr(store_module, "SCROLL_LIMIT", 2)
    store.create_collection(name, dimensions=2)
    ids = [str(uuid.uuid4()) for _ in range(5)]
    client.upsert(
        name,
        points=[
            models.PointStruct(id=i, vector=[0.6, 0.8], payload={"url": f"u{n}"})
            for n, i in enumerate(ids)
        ],
        wait=True,
    )
    seen, offset, requests = [], None, 0
    while True:
        points, offset = store.scroll(name, offset=offset, with_vectors=True)
        seen += points
        requests += 1
        if offset is None:
            break
    assert requests == 3
    assert {p.id for p in seen} == set(ids)
    assert sorted(p.url for p in seen) == [f"u{n}" for n in range(5)]
    assert all(array.array("f", p.vector) == array.array("f", [0.6, 0.8]) for p in seen)


@pytest.mark.network
def test_scroll_without_vectors_reads_none(store, client, name):
    store.create_collection(name, dimensions=2)
    client.upsert(
        name,
        points=[models.PointStruct(id=1, vector=[0.6, 0.8], payload={"url": "u"})],
        wait=True,
    )
    points, offset = store.scroll(name, with_vectors=False)
    assert [(p.id, p.url, p.vector) for p in points] == [("1", "u", None)]
    assert offset is None
