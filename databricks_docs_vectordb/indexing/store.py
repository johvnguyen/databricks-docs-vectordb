"""The Qdrant side of Phase 4, through `qdrant-client` directly (Q4).

vector_db_integration.md's "Record and collection schema" and "Index and
search configuration": one collection per embedding model, named from the
model, holding a dense vector at the rows' dimensions, room for one sparse
vector, and `keyword` indexes on the two payload fields that are filtered
on.

Nothing here retries: the runner owns retries and backoff (Q34), so the
client's own are not relied on.
"""

import contextlib
import math
import re
import warnings
from types import TracebackType
from typing import NamedTuple, Self

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


class StoredPoint(NamedTuple):
    """A point as read back: its ID, its page, and its vector if asked for."""

    id: str
    url: str
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

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def collection_names(self) -> list[str]:
        """Every collection on the server, this phase's or not."""
        return [c.name for c in self._client.get_collections().collections]

    def scroll(
        self, name: str, *, offset: str | None = None, with_vectors: bool
    ) -> tuple[list[StoredPoint], str | None]:
        """Up to `SCROLL_LIMIT` points of `name`, from `offset` (the start
        when None), and the offset of the next request, None after the last.

        One request, so the runner can retry it on its own. Only the `url`
        payload field is read.
        """
        records, next_offset = self._client.scroll(
            collection_name=name,
            limit=SCROLL_LIMIT,
            offset=offset,
            with_payload=["url"],
            with_vectors=with_vectors,
        )
        points = [
            StoredPoint(
                id=str(r.id),
                url=str((r.payload or {}).get("url", "")),
                vector=r.vector if with_vectors else None,
            )
            for r in records
        ]
        return points, None if next_offset is None else str(next_offset)
