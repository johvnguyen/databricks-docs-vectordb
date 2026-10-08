"""The Qdrant side of Phase 4, through `qdrant-client` directly (Q4).

vector_db_integration.md's "Record and collection schema" and "Index and
search configuration": one collection per embedding model, named from the
model, holding a dense vector at the rows' dimensions, room for one sparse
vector, and `keyword` indexes on the two payload fields that are filtered
on.

Nothing here retries: the runner wraps the store in `RetryingStore` (Q34),
so the client's own retries are not relied on. Each method is one attempt
there, and every method a run calls is safe to repeat.
"""

import contextlib
import hashlib
import math
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import NamedTuple, Self

import httpx
from qdrant_client import QdrantClient, models

from databricks_docs_vectordb.embedding.store import ModelInfo
from databricks_docs_vectordb.indexing.config import Settings

# Declared when a collection is created and left empty, because Qdrant cannot
# add a sparse vector to an existing collection (Q14, measured). What fills it
# is decided when it is first used; its settings are Qdrant's defaults until
# then.
SPARSE_VECTOR_NAME = "sparse"

# The page filter and the section filter (Q9). Every field filtered on gets a
# payload index; `chunk_id` and `sub_index` are lookup keys, not filters.
KEYWORD_INDEXES = ("url", "section")

# Points per scroll request: what the full scans under "Qdrant, measured" used.
SCROLL_LIMIT = 1000


def collection_name(prefix: str, model: ModelInfo, dimensions: int) -> str:
    """`<prefix>_<provider>_<model>_<dimensions>` (Q10).

    Provider and model are lowercased, with every run of characters other
    than `a-z` and `0-9` turned into `_`. `model` is what
    `configured_model_info` resolves the configured model to; its
    `model_version` is not part of the name. The dimensions are, because
    OpenAI's v3 models can return fewer than their native size.
    """
    if dimensions <= 0:
        raise ValueError(f"dimensions must be > 0, got {dimensions!r}")
    parts = [
        re.sub(r"[^a-z0-9]+", "_", p.lower()) for p in (model.provider, model.model)
    ]
    return "_".join([prefix, *parts, str(dimensions)])


class Point(NamedTuple):
    """One embeddings row as a Qdrant point ("Record and collection
    schema"): what `index run` writes and what `index verify` expects."""

    id: str
    vector: list[float]
    payload: dict[str, str | int]  # chunk_id, sub_index, url, section


class StoredPoint(NamedTuple):
    """A point as read back: its ID, its page and section, and its vector if
    asked for."""

    id: str
    url: str
    section: str
    vector: list[float] | None


def connect(settings: Settings) -> QdrantClient:
    """A REST client for the configured server, sending the API key."""
    with warnings.catch_warnings():
        # The sidecar speaks plain HTTP by design ("Vector database and
        # hosting"), so the client's warning about sending the key over it
        # would print on every command.
        warnings.filterwarnings(
            "ignore", message="Api key is used with an insecure connection"
        )
        return QdrantClient(
            url=settings.url,
            api_key=settings.api_key,
            timeout=math.ceil(settings.request_timeout),
        )


class QdrantStore:
    """This phase's collections in the Qdrant sidecar."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = connect(settings)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def create_collection(self, name: str, *, dimensions: int) -> None:
        """Create `name` empty, with its payload indexes.

        The dense vector is unnamed, at `dimensions`, under the configured
        distance. HNSW is left at Qdrant's defaults, and so is where vectors
        and the graph live (RAM). The quantized copy is scalar `int8` with
        `always_ram`; the originals are kept and are what a retrieve
        returns. Payloads stay on disk (`on_disk_payload`'s default).

        Raises if `name` already exists. If creating a payload index fails,
        the collection is deleted again, so a retry starts clean rather
        than finding a collection with no indexes.
        """
        self._client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance(self._settings.distance),
            ),
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
            # `always_ram` is deprecated in qdrant-client 1.19 in favour of
            # `memory`, but the server still takes it and reads it back.
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, always_ram=True
                )
            ),
        )
        try:
            for field_name in KEYWORD_INDEXES:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
        except BaseException:
            with contextlib.suppress(Exception):
                self._client.delete_collection(name)
            raise

    def recreate_collection(self, name: str, *, dimensions: int) -> None:
        """Drop `name` if it exists, then `create_collection` it. Safe to
        repeat, unlike `create_collection`: a retry after a create that went
        through but timed out drops what that attempt made."""
        if self._client.collection_exists(name):
            self._client.delete_collection(name)
        self.create_collection(name, dimensions=dimensions)

    def settings_mismatches(self, name: str, *, dimensions: int) -> list[str]:
        """How `name` differs from what `create_collection` would make of it
        now, one phrase per difference; empty if it matches.

        Only what `create_collection` sets is checked, and an extra payload
        index is not a difference: fields can be indexed later (Q9).
        """
        info = self._client.get_collection(name)
        found = []
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            found.append("its dense vectors are named, not one unnamed vector")
        else:
            if vectors.size != dimensions:
                found.append(f"vector size is {vectors.size}, rows have {dimensions}")
            if vectors.distance != models.Distance(self._settings.distance):
                found.append(
                    f"distance is {vectors.distance.value}, "
                    f"configured {self._settings.distance}"
                )
        quantization = info.config.quantization_config
        if not (
            isinstance(quantization, models.ScalarQuantization)
            and quantization.scalar.type == models.ScalarType.INT8
            and quantization.scalar.always_ram
        ):
            found.append("quantization is not scalar int8 with always_ram")
        if SPARSE_VECTOR_NAME not in (info.config.params.sparse_vectors or {}):
            found.append(f"no sparse vector named {SPARSE_VECTOR_NAME!r}")
        for field_name in KEYWORD_INDEXES:
            index = info.payload_schema.get(field_name)
            if index is None or index.data_type != models.PayloadSchemaType.KEYWORD:
                found.append(f"no keyword index on {field_name!r}")
        return found

    def save_snapshot(self, name: str, directory: Path) -> Path:
        """Snapshot `name` as it is now, download it into `directory`, and
        delete the server's copy, since nothing mounts the server's
        snapshots path (Q29, Q40). Returns the file, named as Qdrant named
        the snapshot.

        The download goes to a `.partial` file, renamed once its sha256
        matches the server's checksum, so a `.snapshot` file there is always
        whole. If anything fails, the server's copy is deleted all the same.
        """
        snapshot = self._client.create_snapshot(name, wait=True)
        if snapshot is None:
            raise RuntimeError(f"Qdrant returned no snapshot of {name}")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / snapshot.name
        partial = path.with_name(f"{path.name}.partial")
        try:
            digest = hashlib.sha256()
            with (
                httpx.stream(
                    "GET",
                    f"{self._settings.url.rstrip('/')}/collections/{name}"
                    f"/snapshots/{snapshot.name}",
                    headers={"api-key": self._settings.api_key},
                    timeout=self._settings.request_timeout,
                ) as response,
                partial.open("wb") as file,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    file.write(chunk)
                    digest.update(chunk)
            if snapshot.checksum and digest.hexdigest() != snapshot.checksum:
                raise RuntimeError(
                    f"the downloaded snapshot of {name} does not match its checksum"
                )
            partial.rename(path)
        except BaseException:
            partial.unlink(missing_ok=True)
            with contextlib.suppress(Exception):
                self._client.delete_snapshot(name, snapshot.name, wait=True)
            raise
        self._client.delete_snapshot(name, snapshot.name, wait=True)
        return path

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)

    def delete_pages(self, name: str, urls: Sequence[str]) -> None:
        """Delete every point of `name` whose `url` is one of `urls`, in one
        request."""
        self._client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url", match=models.MatchAny(any=list(urls))
                        )
                    ]
                )
            ),
            wait=True,
        )

    def upsert(self, name: str, points: Sequence[Point]) -> None:
        """Write `points` to `name` in one request, replacing any point with
        the same ID. Sent as one `Batch`, which measured about four times
        faster than a list of `PointStruct`s at 64 points."""
        self._client.upsert(
            collection_name=name,
            points=models.Batch(
                ids=[p.id for p in points],
                vectors=[p.vector for p in points],
                payloads=[p.payload for p in points],
            ),
            wait=True,
        )

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def count(self, name: str) -> int:
        """How many points `name` holds, counted exactly."""
        return self._client.count(name, exact=True).count

    def collection_names(self) -> list[str]:
        """Every collection on the server, this phase's or not."""
        return [c.name for c in self._client.get_collections().collections]

    def scroll(
        self, name: str, *, offset: str | None = None, with_vectors: bool
    ) -> tuple[list[StoredPoint], str | None]:
        """Up to `SCROLL_LIMIT` points of `name`, from `offset` (the start
        when None), and the offset of the next request, None after the last.

        One request, so the runner can retry it on its own. Only the `url`
        and `section` payload fields are read.
        """
        records, next_offset = self._client.scroll(
            collection_name=name,
            limit=SCROLL_LIMIT,
            offset=offset,
            with_payload=["url", "section"],
            with_vectors=with_vectors,
        )
        points = [
            StoredPoint(
                id=str(r.id),
                url=str((r.payload or {}).get("url", "")),
                section=str((r.payload or {}).get("section", "")),
                vector=r.vector if with_vectors else None,
            )
            for r in records
        ]
        return points, None if next_offset is None else str(next_offset)
