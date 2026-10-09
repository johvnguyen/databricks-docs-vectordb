import array
import uuid
from dataclasses import replace

import httpx
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


@pytest.mark.network
@pytest.mark.parametrize("distance", ["Cosine", "Dot"])
def test_a_collection_create_collection_made_has_no_mismatches(
    settings, name, distance
):
    with QdrantStore(replace(settings, distance=distance)) as store:
        store.create_collection(name, dimensions=4)
        assert store.settings_mismatches(name, dimensions=4) == []


@pytest.mark.network
def test_a_different_size_or_distance_is_a_mismatch(store, settings, name):
    store.create_collection(name, dimensions=4)
    with QdrantStore(replace(settings, distance="Dot")) as dot:
        assert dot.settings_mismatches(name, dimensions=8) == [
            "vector size is 4, rows have 8",
            "distance is Cosine, configured Dot",
        ]


@pytest.mark.network
def test_a_collection_made_some_other_way_mismatches(store, client, name):
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    assert store.settings_mismatches(name, dimensions=4) == [
        "quantization is not scalar int8 with always_ram",
        f"no sparse vector named {SPARSE_VECTOR_NAME!r}",
        "no keyword index on 'url'",
        "no keyword index on 'section'",
    ]


@pytest.mark.network
def test_named_dense_vectors_are_a_mismatch(store, client, name):
    client.create_collection(
        name,
        vectors_config={
            "dense": models.VectorParams(size=4, distance=models.Distance.COSINE)
        },
    )
    assert store.settings_mismatches(name, dimensions=4)[0] == (
        "its dense vectors are named, not one unnamed vector"
    )


@pytest.mark.network
def test_an_extra_payload_index_is_not_a_mismatch(store, client, name):
    """Fields can be added and indexed later without a reload (Q9)."""
    store.create_collection(name, dimensions=4)
    client.create_payload_index(
        name, "source", field_schema=models.PayloadSchemaType.KEYWORD, wait=True
    )
    assert store.settings_mismatches(name, dimensions=4) == []


@pytest.fixture
def ten_points(store, client, name) -> str:
    store.create_collection(name, dimensions=2)
    client.upsert(
        name,
        points=[
            models.PointStruct(id=i, vector=[0.6, 0.8], payload={"url": f"u{i}"})
            for i in range(10)
        ],
        wait=True,
    )
    return name


@pytest.mark.network
def test_save_snapshot_downloads_a_whole_file_and_leaves_none_on_the_server(
    store, client, ten_points, tmp_path, restore
):
    directory = tmp_path / "qdrant_snapshots"
    path = store.save_snapshot(ten_points, directory)
    assert path.parent == directory
    assert path.name.startswith(ten_points)
    assert path.suffix == ".snapshot"
    assert [p.name for p in directory.iterdir()] == [path.name]  # no .partial
    assert client.list_snapshots(ten_points) == []
    assert restore(path, f"{ten_points}_restored") == 10


@pytest.mark.network
def test_a_failed_download_keeps_no_file_and_no_server_snapshot(
    store, client, ten_points, tmp_path, monkeypatch
):
    def fail(*args, **kwargs):
        raise httpx.ConnectError("sidecar went away")

    monkeypatch.setattr(store_module.httpx, "stream", fail)
    with pytest.raises(httpx.ConnectError):
        store.save_snapshot(ten_points, tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert client.list_snapshots(ten_points) == []


@pytest.mark.network
def test_a_checksum_mismatch_keeps_no_file_and_no_server_snapshot(
    store, client, ten_points, tmp_path, monkeypatch
):
    create = QdrantClient.create_snapshot

    def corrupt(self, *args, **kwargs):
        return create(self, *args, **kwargs).model_copy(update={"checksum": "0" * 64})

    monkeypatch.setattr(QdrantClient, "create_snapshot", corrupt)
    with pytest.raises(RuntimeError, match="checksum"):
        store.save_snapshot(ten_points, tmp_path)
    assert list(tmp_path.iterdir()) == []
    assert client.list_snapshots(ten_points) == []
