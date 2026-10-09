import array
import logging
import math
import os
import random
import re
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import httpx
import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException

from databricks_docs_vectordb.embedding.runner import text_hash
from databricks_docs_vectordb.embedding.store import (
    EmbeddingRow,
    EmbeddingsStore,
    ModelInfo,
    StoredVector,
    pack_embedding,
    unpack_embedding,
)
from databricks_docs_vectordb.indexing import store as store_module
from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.runner import (
    EXIT_CODES,
    Plan,
    QdrantUnreachable,
    RetryingStore,
    Runner,
    SaveQuestionUnanswered,
    compare,
    point_id,
    save_affected,
    should_save,
    skip_orphans,
    to_point,
    unit_vector,
    write,
)
from databricks_docs_vectordb.indexing.store import QdrantStore, collection_name
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore
from databricks_docs_vectordb.scraping.manifest import Manifest

MODEL = ModelInfo("openai", "text-embedding-3-small", None)
PAGE = "https://docs.databricks.com/aws/en/delta/"
VECTOR = [0.6, 0.8]


@pytest.fixture
def warnings_logged(caplog):
    caplog.set_level(logging.WARNING, logger="databricks_docs_vectordb.indexing")
    return caplog


def stored(index: int, sub_index: int = 0) -> StoredVector:
    return StoredVector(
        f"{PAGE}#{index}", sub_index, PAGE, "aws-en", 2, pack_embedding(VECTOR)
    )


def row_of(
    url: str = PAGE,
    source: str = "aws-en",
    *,
    index: int = 0,
    sub_index: int = 0,
    vector: list[float] = VECTOR,
) -> StoredVector:
    return StoredVector(
        f"{url}#{index}", sub_index, url, source, len(vector), pack_embedding(vector)
    )


def make_chunk(index: int) -> Chunk:
    text = f"chunk {index} text"
    return Chunk(
        chunk_id=f"{PAGE}#{index}",
        url=PAGE,
        source="aws-en",
        cloud="aws",
        chunk_index=index,
        prev_chunk_id=None,
        next_chunk_id=None,
        heading_path="",
        section_path="Delta Lake",
        doc_title="Delta Lake",
        description=None,
        text=text,
        char_count=len(text),
        token_count=None,
        token_count_model=None,
        is_oversized=False,
        last_updated=None,
        page_sha256="ab" * 32,
        fetched_at=None,
    )


def make_row(
    index: int, *, status: str = "ok", model: ModelInfo = MODEL
) -> EmbeddingRow:
    ok = status == "ok"
    return EmbeddingRow(
        chunk_id=f"{PAGE}#{index}",
        sub_index=0,
        sub_count=1,
        url=PAGE,
        source="aws-en",
        cloud="aws",
        chunk_index=index,
        prev_chunk_id=None,
        next_chunk_id=None,
        heading_path="",
        section_path="Delta Lake",
        doc_title="Delta Lake",
        description=None,
        last_updated=None,
        page_sha256="ab" * 32,
        chunk_text_hash=text_hash(f"chunk {index} text"),
        embedded_text=f"Delta Lake\n\nchunk {index} text",
        token_count=8,
        provider=model.provider,
        model=model.model,
        model_version=model.model_version,
        dimensions=len(VECTOR) if ok else None,
        embedding=pack_embedding(VECTOR) if ok else None,
        status=status,
        attempts=1,
        error=None if ok else "boom",
    )


# -- skip_orphans ----------------------------------------------------------------


def test_rows_still_in_parsed_pass_through_in_order(warnings_logged):
    rows = [stored(0), stored(1, 0), stored(1, 1)]
    skipped: list[StoredVector] = []
    assert list(skip_orphans(rows, {f"{PAGE}#0", f"{PAGE}#1"}, skipped)) == rows
    assert skipped == []
    assert warnings_logged.records == []


def test_an_orphan_is_skipped_with_a_warning_naming_it(warnings_logged):
    orphan = stored(7, sub_index=1)
    skipped: list[StoredVector] = []
    rows = [stored(0), orphan, stored(1)]
    assert list(skip_orphans(rows, {f"{PAGE}#0", f"{PAGE}#1"}, skipped)) == [
        stored(0),
        stored(1),
    ]
    assert skipped == [orphan]
    [record] = warnings_logged.records
    assert record.levelno == logging.WARNING
    assert f"{PAGE}#7 (sub_index 1)" in record.getMessage()


def test_every_slice_of_an_orphaned_split_chunk_is_skipped(warnings_logged):
    slices = [stored(7, i) for i in range(3)]
    skipped: list[StoredVector] = []
    assert list(skip_orphans(slices, set(), skipped)) == []
    assert skipped == slices
    assert len(warnings_logged.records) == 3


def test_skipped_is_optional(warnings_logged):
    assert list(skip_orphans([stored(7)], set())) == []
    assert len(warnings_logged.records) == 1


# -- with both stores -------------------------------------------------------------


def test_only_the_orphan_warns_among_the_rows_left_out(tmp_path, warnings_logged):
    """Step 4's Verify: a `failed` row, another model's row and an orphan
    are each left out, and only the orphan produces a warning."""
    with ChunkStore(tmp_path / "parsed.sqlite") as chunks:
        # The page was re-parsed into three chunks, leaving #3 behind.
        chunks.replace_page_chunks(PAGE, [make_chunk(i) for i in range(3)])
        chunk_ids = chunks.chunk_ids()

    other_model = ModelInfo("openai", "text-embedding-3-large", None)
    with EmbeddingsStore(tmp_path / "embeddings.sqlite") as embeddings:
        embeddings.upsert(f"{PAGE}#0", [make_row(0)])
        embeddings.upsert(f"{PAGE}#1", [make_row(1, status="failed")])
        embeddings.upsert(f"{PAGE}#2", [make_row(2, model=other_model)])
        embeddings.upsert(f"{PAGE}#3", [make_row(3)])
        kept = list(skip_orphans(embeddings.ok_vectors(MODEL), chunk_ids))

    assert [row.chunk_id for row in kept] == [f"{PAGE}#0"]
    [record] = warnings_logged.records
    assert f"{PAGE}#3" in record.getMessage()


# -- point_id / unit_vector / to_point ------------------------------------------


def test_point_id_is_pinned():
    """Changing the namespace or the key's format would change every ID."""
    assert point_id(f"{PAGE}#0", 0) == "3ebb2d4a-9f33-50b5-9e39-63242bb011fe"


def test_point_ids_differ_per_chunk_and_slice():
    ids = {point_id(f"{PAGE}#{i}", s) for i in range(3) for s in range(3)}
    assert len(ids) == 9


def test_the_id_depends_only_on_the_natural_key():
    """A re-embed (new vector, new dimensions) keeps the row's ID."""
    before = to_point(row_of(vector=[3.0, 4.0]), normalize=True)
    after = to_point(row_of(vector=[1.0, 2.0, 2.0]), normalize=True)
    assert before.id == after.id == point_id(f"{PAGE}#0", 0)


def test_every_slice_of_a_split_chunk_gets_its_own_point():
    points = [to_point(row_of(sub_index=i), normalize=True) for i in range(3)]
    assert len({p.id for p in points}) == 3
    assert [p.payload["sub_index"] for p in points] == [0, 1, 2]


def test_a_normalized_vector_is_the_blob_scaled_to_unit_length():
    point = to_point(row_of(vector=[3.0, 4.0]), normalize=True)
    assert point.vector == array.array("f", [0.6, 0.8]).tolist()


def test_without_normalization_the_vector_is_the_blob_unchanged():
    vector = [3.0, 4.0, 0.1]
    point = to_point(row_of(vector=vector), normalize=False)
    assert point.vector == unpack_embedding(pack_embedding(vector)).tolist()


@pytest.mark.parametrize(
    "vector",
    [
        [3.0, 4.0],
        [1e-3] * 1536,
        [random.Random(0).gauss(0, 1) for _ in range(1536)],
    ],
    ids=["3-4-5", "tiny", "gaussian"],
)
def test_unit_vector_is_unit_length_in_float32(vector):
    unit = unit_vector(vector)
    assert array.array("f", unit).tolist() == unit
    assert math.fsum(x * x for x in unit) == pytest.approx(1, abs=1e-6)


def test_a_zero_vector_raises_naming_the_row():
    with pytest.raises(ValueError, match=rf"{re.escape(PAGE)}#0 \(sub_index 2\)"):
        to_point(row_of(sub_index=2, vector=[0.0, 0.0]), normalize=True)


def test_the_payload_is_the_lookup_key_and_the_two_filters():
    point = to_point(row_of(index=4, sub_index=1), normalize=True)
    assert point.payload == {
        "chunk_id": f"{PAGE}#4",
        "sub_index": 1,
        "url": PAGE,
        "section": "aws-en/delta",
    }


@pytest.mark.parametrize(
    ("url", "source", "section"),
    [
        (
            "https://docs.databricks.com/aws/en/sql/language-manual/functions/abs",
            "aws-en",
            "aws-en/sql",
        ),
        ("https://docs.databricks.com/api/jobs/v2/run", "api", "api/jobs"),
        (
            "https://docs.databricks.com/aws/en/introduction",
            "aws-en",
            "aws-en/(top-level)",
        ),
    ],
)
def test_the_section_is_the_source_and_its_section(url, source, section):
    assert to_point(row_of(url, source), normalize=True).payload["section"] == section


@pytest.mark.network
@pytest.mark.parametrize("distance", ["Cosine", "Dot"])
def test_a_normalized_point_reads_back_unchanged_as_float32(
    settings, client, name, distance
):
    """What Step 5's change detection and `index verify` rely on: Qdrant
    stores a unit vector as given, under `Cosine` too, and its components
    read back equal once taken as float32."""
    rng = random.Random(0)
    rows = [
        row_of(index=i, vector=[rng.gauss(0, 1) for _ in range(64)]) for i in range(20)
    ]
    points = [to_point(r, normalize=True) for r in rows]
    with QdrantStore(replace(settings, distance=distance)) as store:
        store.create_collection(name, dimensions=64)
    client.upsert(
        name,
        points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ],
        wait=True,
    )
    back = {
        str(record.id): record
        for record in client.retrieve(
            name, [p.id for p in points], with_vectors=True, with_payload=True
        )
    }
    assert set(back) == {p.id for p in points}
    for p in points:
        assert array.array("f", back[p.id].vector) == array.array("f", p.vector)
        assert back[p.id].payload == p.payload


# -- compare (against the sidecar) ------------------------------------------------

PAGES = [f"https://docs.databricks.com/aws/en/p{i}/" for i in range(3)]
DIMENSIONS = 4


def page_rows(url: str, chunks: int = 2, *, seed: int = 0) -> list[StoredVector]:
    """`chunks` rows for `url`, with vectors that are not unit length."""
    rng = random.Random(f"{url} {seed}")
    return [
        row_of(url, index=i, vector=[rng.gauss(0, 1) for _ in range(DIMENSIONS)])
        for i in range(chunks)
    ]


ROWS = [row for url in PAGES for row in page_rows(url)]


@pytest.fixture
def prefixed(settings, name):
    """Settings whose collection prefix is the throwaway `name`, so that
    `<name>_*` collections stand in for other models'."""
    return replace(settings, collection_prefix=name)


@pytest.fixture
def qdrant(prefixed):
    with QdrantStore(prefixed) as s:
        yield s


@pytest.fixture
def collection(name) -> str:
    return collection_name(name, MODEL, DIMENSIONS)


def load(client, collection: str, rows, *, normalize: bool = True) -> None:
    points = [to_point(row, normalize=normalize) for row in rows]
    client.upsert(
        collection,
        points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ],
        wait=True,
    )


@pytest.fixture
def loaded(qdrant, client, collection) -> str:
    """`collection`, holding the points of every row in `ROWS`."""
    qdrant.create_collection(collection, dimensions=DIMENSIONS)
    load(client, collection, ROWS)
    return collection


def changed_vector(row: StoredVector) -> StoredVector:
    return row._replace(embedding=pack_embedding([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.network
def test_with_no_collection_every_page_is_new(qdrant, collection, prefixed):
    plan = compare(
        qdrant, collection, ROWS, set(PAGES), prefixed, dimensions=DIMENSIONS
    )
    assert not plan.exists
    assert not plan.refresh
    assert plan.mismatches == ()
    assert plan.new == set(PAGES)
    assert plan.current == plan.changed == plan.removed == plan.not_ok == set()
    assert plan.old_collections == ()
    assert not plan.replaces_anything


@pytest.mark.network
def test_after_a_load_every_page_is_current(qdrant, loaded, prefixed):
    plan = compare(qdrant, loaded, ROWS, set(PAGES), prefixed, dimensions=DIMENSIONS)
    assert plan.exists
    assert not plan.refresh
    assert plan.mismatches == ()
    assert plan.current == set(PAGES)
    assert plan.new == plan.changed == plan.removed == set()
    assert not plan.replaces_anything


@pytest.mark.network
@pytest.mark.parametrize(
    "edit",
    [
        lambda rows: [changed_vector(r) if r == rows[2] else r for r in rows],
        lambda rows: [r for r in rows if r != rows[3]],
        lambda rows: [*rows, *page_rows(PAGES[1], 3)[2:]],
        lambda rows: [*rows, rows[2]._replace(sub_index=1)],
    ],
    ids=["re-embedded", "fewer-chunks", "more-chunks", "split"],
)
def test_any_difference_changes_only_that_page(qdrant, loaded, prefixed, edit):
    """Each edit touches a row of PAGES[1] only."""
    plan = compare(
        qdrant, loaded, edit(ROWS), set(PAGES), prefixed, dimensions=DIMENSIONS
    )
    assert plan.changed == {PAGES[1]}
    assert plan.current == {PAGES[0], PAGES[2]}
    assert plan.new == plan.removed == set()
    assert plan.replaces_anything


@pytest.mark.network
def test_a_page_not_ok_in_the_manifest_is_removed_and_not_loaded(
    qdrant, loaded, prefixed
):
    extra = page_rows("https://docs.databricks.com/aws/en/gone/")
    ok = {PAGES[0], PAGES[2]}
    plan = compare(qdrant, loaded, [*ROWS, *extra], ok, prefixed, dimensions=DIMENSIONS)
    assert plan.removed == {PAGES[1]}
    assert plan.not_ok == {PAGES[1], extra[0].url}
    assert plan.current == ok
    assert plan.new == set()


@pytest.mark.network
def test_a_page_with_points_but_no_rows_is_removed(qdrant, loaded, prefixed):
    rows = [r for r in ROWS if r.url != PAGES[1]]
    plan = compare(qdrant, loaded, rows, set(PAGES), prefixed, dimensions=DIMENSIONS)
    assert plan.removed == {PAGES[1]}
    assert plan.current == {PAGES[0], PAGES[2]}
    assert plan.not_ok == set()


@pytest.mark.network
def test_other_collections_under_the_prefix_are_old(
    qdrant, client, loaded, name, prefixed
):
    for other in (
        f"{name}_openai_text_embedding_3_large_4",
        f"{name}_a_b_4",
        f"{name}x",
    ):
        client.create_collection(
            other,
            vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
        )
    plan = compare(qdrant, loaded, ROWS, set(PAGES), prefixed, dimensions=DIMENSIONS)
    assert plan.old_collections == (
        f"{name}_a_b_4",
        f"{name}_openai_text_embedding_3_large_4",
    )
    assert plan.current == set(PAGES)
    assert plan.replaces_anything


@pytest.mark.network
def test_refresh_counts_every_page_with_points_as_changed(
    qdrant, loaded, prefixed, monkeypatch
):
    with_vectors = []
    scroll = QdrantStore.scroll

    def spy(self, *args, **kwargs):
        with_vectors.append(kwargs["with_vectors"])
        return scroll(self, *args, **kwargs)

    monkeypatch.setattr(QdrantStore, "scroll", spy)
    extra = page_rows("https://docs.databricks.com/aws/en/new/")
    rows = [r for r in ROWS if r.url != PAGES[2]] + extra
    plan = compare(
        qdrant,
        loaded,
        rows,
        {*PAGES, extra[0].url},
        prefixed,
        dimensions=DIMENSIONS,
        refresh=True,
    )
    assert plan.refresh
    assert plan.changed == {PAGES[0], PAGES[1]}
    assert plan.removed == {PAGES[2]}
    assert plan.new == {extra[0].url}
    assert plan.current == set()
    assert with_vectors and not any(with_vectors)


@pytest.mark.network
def test_a_changed_distance_rebuilds_the_collection(
    loaded, prefixed, warnings_logged, monkeypatch
):
    with_vectors = []
    scroll = QdrantStore.scroll

    def spy(self, *args, **kwargs):
        with_vectors.append(kwargs["with_vectors"])
        return scroll(self, *args, **kwargs)

    monkeypatch.setattr(QdrantStore, "scroll", spy)
    dot = replace(prefixed, distance="Dot")
    with QdrantStore(dot) as qdrant:
        plan = compare(qdrant, loaded, ROWS, set(PAGES), dot, dimensions=DIMENSIONS)
    assert plan.refresh
    assert plan.mismatches == ("distance is Cosine, configured Dot",)
    assert plan.changed == set(PAGES)
    assert plan.current == set()
    assert plan.replaces_anything
    assert not any(with_vectors)
    [record] = warnings_logged.records
    assert loaded in record.getMessage()
    assert "distance is Cosine, configured Dot" in record.getMessage()


@pytest.mark.network
def test_compare_reads_every_scroll_page(qdrant, loaded, prefixed, monkeypatch):
    monkeypatch.setattr(store_module, "SCROLL_LIMIT", 2)
    plan = compare(qdrant, loaded, ROWS, set(PAGES), prefixed, dimensions=DIMENSIONS)
    assert plan.current == set(PAGES)


@pytest.mark.network
def test_vectors_are_compared_under_the_configured_normalization(
    client, collection, prefixed
):
    """Under `Dot`, Qdrant stores vectors as given, so a load without
    normalization is current only to a compare without it too."""
    dot = replace(prefixed, distance="Dot", normalize=False)
    with QdrantStore(dot) as qdrant:
        qdrant.create_collection(collection, dimensions=DIMENSIONS)
        load(client, collection, ROWS, normalize=False)
        assert compare(
            qdrant, collection, ROWS, set(PAGES), dot, dimensions=DIMENSIONS
        ).current == set(PAGES)
        normalized = replace(dot, normalize=True)
        plan = compare(
            qdrant, collection, ROWS, set(PAGES), normalized, dimensions=DIMENSIONS
        )
        assert plan.changed == set(PAGES)


# -- the save question ------------------------------------------------------------

COLLECTION = "databricks_docs_openai_text_embedding_3_small_4"
OLD = "databricks_docs_openai_text_embedding_3_large_4"


def make_plan(**overrides) -> Plan:
    fields = {
        "collection": COLLECTION,
        "dimensions": 4,
        "exists": True,
        "refresh": False,
        "mismatches": (),
        "current": frozenset(),
        "new": frozenset(),
        "changed": frozenset(),
        "removed": frozenset(),
        "not_ok": frozenset(),
        "old_collections": (),
    }
    fields.update(overrides)
    return Plan(**fields)


@pytest.fixture
def index_settings(tmp_path) -> Settings:
    return Settings(api_key="test-key", data_dir=tmp_path)


def never_asked(question: str) -> str:
    raise AssertionError(f"asked: {question}")


@pytest.mark.parametrize(
    ("overrides", "affected"),
    [
        ({"current": frozenset(PAGES), "new": frozenset({"n"})}, []),
        ({"changed": frozenset({"c"})}, [COLLECTION]),
        ({"removed": frozenset({"r"})}, [COLLECTION]),
        ({"old_collections": (OLD,)}, [OLD]),
        ({"changed": frozenset({"c"}), "old_collections": (OLD,)}, [COLLECTION, OLD]),
    ],
    ids=["new-only", "changed", "removed", "old-only", "both"],
)
def test_affected_collections(overrides, affected):
    plan = make_plan(**overrides)
    assert plan.affected_collections == affected
    assert plan.replaces_anything == bool(affected)


@pytest.mark.parametrize("save_old", [None, True, False])
def test_nothing_to_replace_asks_nothing(index_settings, save_old):
    """A first load, or a run with nothing to do."""
    plan = make_plan(exists=False, new=frozenset(PAGES))
    assert not should_save(
        plan, index_settings, save_old, ask=never_asked, interactive=False
    )


@pytest.mark.parametrize("save_old", [True, False])
def test_the_flags_answer_in_advance(index_settings, save_old):
    plan = make_plan(changed=frozenset({PAGES[0]}))
    assert (
        should_save(plan, index_settings, save_old, ask=never_asked, interactive=False)
        is save_old
    )


def test_with_no_terminal_and_no_flag_the_run_stops(index_settings):
    plan = make_plan(changed=frozenset({PAGES[0]}), old_collections=(OLD,))
    with pytest.raises(SaveQuestionUnanswered, match="--save-old or --no-save-old"):
        should_save(plan, index_settings, None, ask=never_asked, interactive=False)


@pytest.mark.parametrize(
    ("answer", "save"), [("y", True), ("YES ", True), ("n", False), ("No", False)]
)
def test_the_question_is_asked_once_on_a_terminal(index_settings, answer, save):
    plan = make_plan(
        changed=frozenset(PAGES[:2]),
        removed=frozenset({PAGES[2]}),
        old_collections=(OLD,),
    )
    asked = []

    def ask(question: str) -> str:
        asked.append(question)
        return answer

    assert should_save(plan, index_settings, None, ask=ask, interactive=True) is save
    [question] = asked
    assert f"replace 2 page(s) in {COLLECTION}" in question
    assert f"remove 1 page(s) from {COLLECTION}" in question
    assert f"delete the old collection(s) {OLD}" in question
    assert f"{COLLECTION}, {OLD} to {index_settings.snapshots_dir}/" in question


def test_a_rebuild_is_described_as_one(index_settings):
    plan = make_plan(
        refresh=True, changed=frozenset(PAGES[:2]), removed=frozenset({PAGES[2]})
    )
    asked = []
    should_save(
        plan,
        index_settings,
        None,
        ask=lambda q: asked.append(q) or "n",
        interactive=True,
    )
    assert f"drop and rebuild {COLLECTION}, which holds 3 page(s)" in asked[0]


def test_any_other_answer_asks_again(index_settings):
    answers = iter(["", "maybe", "n"])
    asked = []

    def ask(question: str) -> str:
        asked.append(question)
        return next(answers)

    plan = make_plan(changed=frozenset({PAGES[0]}))
    assert not should_save(plan, index_settings, None, ask=ask, interactive=True)
    assert asked[1:] == ["Please answer y or n: "] * 2


def test_no_answer_stops_the_run(index_settings):
    def ask(question: str) -> str:
        raise EOFError

    plan = make_plan(changed=frozenset({PAGES[0]}))
    with pytest.raises(SaveQuestionUnanswered):
        should_save(plan, index_settings, None, ask=ask, interactive=True)


@pytest.mark.network
def test_save_affected_snapshots_every_affected_collection(
    qdrant, client, loaded, name, prefixed, tmp_path, restore
):
    old = f"{name}_openai_text_embedding_3_large_4"
    client.create_collection(
        old, vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
    )
    rows = [r for r in ROWS if r.url != PAGES[1]]
    plan = compare(qdrant, loaded, rows, set(PAGES), prefixed, dimensions=DIMENSIONS)
    assert plan.affected_collections == [loaded, old]

    settings = replace(prefixed, data_dir=tmp_path)
    paths = save_affected(qdrant, plan, settings)
    assert [p.parent for p in paths] == [settings.snapshots_dir] * 2
    assert paths[0].name.startswith(loaded) and paths[1].name.startswith(old)
    assert client.list_snapshots(loaded) == client.list_snapshots(old) == []
    # The saved copy holds the collection as it was before the run.
    assert restore(paths[0], f"{name}_restored") == len(ROWS)


# -- write (against the sidecar) --------------------------------------------------

WRITES = ("create_collection", "delete_collection", "delete_pages", "upsert")


@pytest.fixture
def calls(monkeypatch):
    """Every write call on a QdrantStore, as (method, args), in order."""
    recorded = []
    for method in WRITES:
        original = getattr(QdrantStore, method)

        def spy(self, *args, _method=method, _original=original, **kwargs):
            recorded.append((_method, args))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(QdrantStore, method, spy)
    return recorded


def upserted_urls(calls) -> set[str]:
    return {p.payload["url"] for m, args in calls if m == "upsert" for p in args[1]}


def run(qdrant, collection, rows, ok, settings, *, refresh=False):
    """`index run`'s compare then write, with no save question."""
    plan = compare(
        qdrant, collection, rows, ok, settings, dimensions=DIMENSIONS, refresh=refresh
    )
    return plan, write(qdrant, plan, rows, settings)


def assert_current(qdrant, collection, rows, ok, settings):
    """Loading twice in a row changes nothing: every page is current."""
    plan = compare(qdrant, collection, rows, ok, settings, dimensions=DIMENSIONS)
    assert plan.current == {r.url for r in rows if r.url in ok}
    assert plan.new == plan.changed == plan.removed == set()
    assert not plan.refresh


@pytest.mark.network
def test_a_first_load_creates_the_collection_and_loads_every_row(
    qdrant, client, collection, prefixed
):
    plan, written = run(qdrant, collection, ROWS, set(PAGES), prefixed)
    assert written == (0, len(PAGES), len(ROWS))
    assert qdrant.settings_mismatches(collection, dimensions=DIMENSIONS) == []
    assert client.count(collection, exact=True).count == len(ROWS)
    assert_current(qdrant, collection, ROWS, set(PAGES), prefixed)


@pytest.mark.network
def test_a_changed_page_alone_is_replaced_and_leaves_nothing_behind(
    qdrant, client, loaded, prefixed, calls
):
    calls.clear()
    dropped = ROWS[3]  # PAGES[1] re-parsed into one chunk fewer
    rows = [r for r in ROWS if r != dropped]
    plan, written = run(qdrant, loaded, rows, set(PAGES), prefixed)
    assert written == (1, 1, 1)
    assert [(m, args[1]) for m, args in calls if m == "delete_pages"] == [
        ("delete_pages", (PAGES[1],))
    ]
    assert upserted_urls(calls) == {PAGES[1]}
    assert client.retrieve(loaded, [point_id(dropped.chunk_id, 0)]) == []
    assert client.count(loaded, exact=True).count == len(rows)
    assert_current(qdrant, loaded, rows, set(PAGES), prefixed)


@pytest.mark.network
def test_a_removed_page_loses_every_point(qdrant, client, loaded, prefixed):
    ok = {PAGES[0], PAGES[1]}
    plan, written = run(qdrant, loaded, ROWS, ok, prefixed)
    assert plan.removed == {PAGES[2]}
    assert written == (1, 0, 0)
    gone = models.Filter(
        must=[models.FieldCondition(key="url", match=models.MatchValue(value=PAGES[2]))]
    )
    assert client.count(loaded, count_filter=gone, exact=True).count == 0
    assert_current(qdrant, loaded, ROWS, ok, prefixed)


@pytest.mark.network
def test_nothing_to_do_writes_nothing(qdrant, loaded, prefixed, calls):
    calls.clear()
    plan, written = run(qdrant, loaded, ROWS, set(PAGES), prefixed)
    assert written == (0, 0, 0)
    assert calls == []


@pytest.mark.network
def test_refresh_drops_and_recreates_the_collection(qdrant, client, loaded, prefixed):
    # An extra index, which only the old collection has.
    client.create_payload_index(
        loaded, "source", field_schema=models.PayloadSchemaType.KEYWORD, wait=True
    )
    plan, written = run(qdrant, loaded, ROWS, set(PAGES), prefixed, refresh=True)
    assert written == (len(PAGES), len(PAGES), len(ROWS))
    assert "source" not in client.get_collection(loaded).payload_schema
    assert client.count(loaded, exact=True).count == len(ROWS)
    assert_current(qdrant, loaded, ROWS, set(PAGES), prefixed)


@pytest.mark.network
def test_a_settings_mismatch_rebuilds_under_the_new_settings(loaded, prefixed):
    dot = replace(prefixed, distance="Dot")
    with QdrantStore(dot) as qdrant:
        plan, written = run(qdrant, loaded, ROWS, set(PAGES), dot)
        assert plan.refresh
        assert qdrant.settings_mismatches(loaded, dimensions=DIMENSIONS) == []
        assert_current(qdrant, loaded, ROWS, set(PAGES), dot)


@pytest.mark.network
def test_deletes_come_first_and_batches_keep_to_batch_size(
    qdrant, loaded, prefixed, calls
):
    calls.clear()
    rows = [changed_vector(r) if r.chunk_id.endswith("#0") else r for r in ROWS]
    small = replace(prefixed, batch_size=2)
    run(qdrant, loaded, rows, set(PAGES), small)
    assert [m for m, _ in calls] == ["delete_pages"] * 2 + ["upsert"] * 3
    assert [len(args[1]) for _, args in calls] == [2, 1, 2, 2, 2]


@pytest.mark.network
def test_a_write_stopped_part_way_finishes_on_the_next_run(
    qdrant, client, loaded, prefixed, monkeypatch
):
    rows = [changed_vector(r) if r.url != PAGES[1] else r for r in ROWS]
    small = replace(prefixed, batch_size=2)
    upsert = QdrantStore.upsert
    upserts = []

    def stop_on_second(self, *args, **kwargs):
        upserts.append(args)
        if len(upserts) == 2:
            raise KeyboardInterrupt
        return upsert(self, *args, **kwargs)

    monkeypatch.setattr(QdrantStore, "upsert", stop_on_second)
    with pytest.raises(KeyboardInterrupt):
        run(qdrant, loaded, rows, set(PAGES), small)
    monkeypatch.setattr(QdrantStore, "upsert", upsert)

    plan = compare(qdrant, loaded, rows, set(PAGES), small, dimensions=DIMENSIONS)
    assert plan.current == {PAGES[0], PAGES[1]}
    assert plan.new == {PAGES[2]}  # deleted, and stopped before its upsert
    plan, written = run(qdrant, loaded, rows, set(PAGES), small)
    assert written == (0, 1, 2)
    assert client.count(loaded, exact=True).count == len(ROWS)
    assert_current(qdrant, loaded, rows, set(PAGES), small)


# -- RetryingStore ----------------------------------------------------------------


class FlakyStore:
    """`scroll` fails `failures` times with `error`, then answers."""

    def __init__(self, failures: int, error: Exception) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    def scroll(self, *args, **kwargs) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "answered"


@pytest.fixture
def logged(caplog):
    caplog.set_level(logging.INFO, logger="databricks_docs_vectordb.indexing")
    return caplog


def retrying(store, *, attempts: int = 10, unit: float = 1.0):
    """A RetryingStore that records its sleeps instead of sleeping. A `unit`
    of 1.0 makes every delay the nominal one."""
    sleeps: list[float] = []
    settings = Settings(api_key="test-key", max_attempts=attempts)
    return RetryingStore(store, settings, sleep=sleeps.append, rng=lambda: unit), sleeps


REFUSED = ResponseHandlingException(
    httpx.ConnectError("[Errno 111] Connection refused")
)


@pytest.mark.parametrize(
    "error",
    [
        REFUSED,
        ResponseHandlingException(httpx.ConnectTimeout("timed out")),
        httpx.ReadTimeout("timed out"),
        httpx.RemoteProtocolError("Server disconnected"),
    ],
    ids=["refused", "connect-timeout", "read-timeout", "disconnected"],
)
def test_connection_errors_and_timeouts_are_retried(error, logged):
    store = FlakyStore(3, error)
    qdrant, sleeps = retrying(store)
    assert qdrant.scroll("c", with_vectors=False) == "answered"
    assert store.calls == 4
    assert sleeps == [1.0, 2.0, 4.0]
    assert qdrant.retries == 3
    messages = [r.getMessage() for r in logged.records]
    assert len(messages) == 3
    assert all(m.startswith("qdrant scroll: ") for m in messages)
    assert "(attempt 3/10)" in messages[2]


@pytest.mark.parametrize(
    "error",
    [ValueError("a bug"), ResponseHandlingException(ValueError("bad response"))],
    ids=["other", "wrapped-other"],
)
def test_other_errors_are_raised_at_once(error):
    store = FlakyStore(1, error)
    qdrant, sleeps = retrying(store)
    with pytest.raises(type(error)):
        qdrant.scroll("c", with_vectors=False)
    assert store.calls == 1
    assert sleeps == []


def test_after_the_last_attempt_it_gives_up_having_logged_every_one(logged):
    store = FlakyStore(99, REFUSED)
    qdrant, sleeps = retrying(store)
    with pytest.raises(QdrantUnreachable, match="in 10 attempts"):
        qdrant.scroll("c", with_vectors=False)
    assert store.calls == 10
    assert len(sleeps) == 9
    attempts = [r for r in logged.records if "attempt" in r.getMessage()]
    assert [f"(attempt {n}/10)" in r.getMessage() for n, r in enumerate(attempts, 1)]
    assert len(attempts) == 10
    assert attempts[-1].levelno == logging.ERROR


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        (1.0, [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]),
        (0.0, [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]),
    ],
    ids=["most", "least"],
)
def test_backoff_doubles_up_to_the_cap_with_jitter(unit, expected):
    qdrant, sleeps = retrying(FlakyStore(99, REFUSED), unit=unit)
    with pytest.raises(QdrantUnreachable):
        qdrant.scroll("c", with_vectors=False)
    assert sleeps == expected


# -- Runner -----------------------------------------------------------------------


def chunk_for(row: StoredVector) -> Chunk:
    index = int(row.chunk_id.rsplit("#", 1)[1])
    return replace(make_chunk(index), chunk_id=row.chunk_id, url=row.url)


def embedding_row_for(row: StoredVector) -> EmbeddingRow:
    index = int(row.chunk_id.rsplit("#", 1)[1])
    return replace(
        make_row(index),
        chunk_id=row.chunk_id,
        url=row.url,
        dimensions=row.dimensions,
        embedding=row.embedding,
    )


@contextmanager
def open_stores(settings: Settings) -> Iterator[tuple]:
    with (
        EmbeddingsStore(settings.embeddings_path) as embeddings,
        ChunkStore(settings.parsed_path) as chunks,
        Manifest(settings.manifest_path) as manifest,
    ):
        yield embeddings, chunks, manifest


def fill(settings: Settings, rows=ROWS, *, orphans=()) -> None:
    """Every row in embeddings.sqlite, its chunk in parsed.sqlite (but not
    the `orphans`' chunks), and every page `ok` in the manifest."""
    with open_stores(settings) as (embeddings, chunks, manifest):
        for row in [*rows, *orphans]:
            embeddings.upsert(row.chunk_id, [embedding_row_for(row)])
        pages = sorted({row.url for row in rows})
        for url in pages:
            chunks.replace_page_chunks(
                url, [chunk_for(row) for row in rows if row.url == url]
            )
        manifest.add_pending("aws-en", pages)
        for url in pages:
            manifest.record_ok(url, http_status=200, sha256="ab" * 32, size_bytes=1)


@pytest.fixture
def run_settings(prefixed, tmp_path) -> Settings:
    """The throwaway prefix, with the stores (and snapshots) in tmp_path."""
    return replace(prefixed, data_dir=tmp_path)


def index_run(settings: Settings, qdrant=None, **run):
    """One `index run`, with no terminal, no real sleeps, and stores opened
    for it alone."""
    with open_stores(settings) as (embeddings, chunks, manifest):
        with qdrant or QdrantStore(settings) as store:
            runner = Runner(
                settings,
                MODEL,
                embeddings,
                chunks,
                manifest,
                store,
                sleep=lambda seconds: None,
                interactive=False,
            )
            return runner.run(**run)


@pytest.mark.network
def test_a_run_loads_every_row_and_a_second_changes_nothing(
    run_settings, collection, client, logged
):
    fill(run_settings)
    first = index_run(run_settings)
    assert (first.status, first.exit_code) == ("completed", 0)
    assert first.collection == collection
    assert first.written == (0, len(PAGES), len(ROWS))
    assert first.retries == 0
    second = index_run(run_settings)
    assert second.status == "completed"
    assert second.plan.current == set(PAGES)
    assert second.written == (0, 0, 0)
    assert client.count(collection, exact=True).count == len(ROWS)
    summaries = [r for r in logged.records if r.getMessage().startswith("completed in")]
    assert len(summaries) == 2
    assert f"{len(ROWS)} points upserted" in summaries[0].getMessage()


def test_with_qdrant_unreachable_the_run_fails_after_its_retries(
    settings, tmp_path, logged
):
    closed = replace(
        settings, url="http://localhost:1", data_dir=tmp_path, max_attempts=3
    )
    fill(closed)
    summary = index_run(closed)
    assert (summary.status, summary.exit_code) == ("failed", 1)
    assert "did not answer collection_exists in 3 attempts" in summary.error
    assert summary.retries == 2
    attempts = [r for r in logged.records if "(attempt" in r.getMessage()]
    assert len(attempts) == 3
    assert summary.plan is None


@pytest.mark.network
def test_ctrl_c_mid_write_exits_130_and_the_next_run_finishes(
    run_settings, collection, client, monkeypatch
):
    fill(run_settings)
    small = replace(run_settings, batch_size=2)
    upsert = QdrantStore.upsert
    upserts = []

    def stop_on_second(self, *args, **kwargs):
        upserts.append(args)
        if len(upserts) == 2:
            raise KeyboardInterrupt
        return upsert(self, *args, **kwargs)

    monkeypatch.setattr(QdrantStore, "upsert", stop_on_second)
    stopped = index_run(small)
    assert (stopped.status, stopped.exit_code) == ("interrupted", 130)
    assert stopped.written is None
    monkeypatch.setattr(QdrantStore, "upsert", upsert)

    resumed = index_run(small)
    assert resumed.status == "completed"
    assert resumed.plan.current == {PAGES[0]}  # its batch went in before the stop
    assert client.count(collection, exact=True).count == len(ROWS)
    assert index_run(small).plan.current == set(PAGES)


@pytest.mark.network
def test_sigterm_stops_the_run_like_ctrl_c(run_settings, monkeypatch):
    fill(run_settings)
    before = signal.getsignal(signal.SIGTERM)
    upsert = QdrantStore.upsert

    def terminated(self, *args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return upsert(self, *args, **kwargs)

    monkeypatch.setattr(QdrantStore, "upsert", terminated)
    summary = index_run(run_settings)
    assert (summary.status, summary.exit_code) == ("interrupted", 130)
    assert signal.getsignal(signal.SIGTERM) is before


@pytest.mark.network
def test_the_orphan_safeguard_warns_once_as_read_and_names_it_after_the_summary(
    run_settings, collection, client, logged, monkeypatch
):
    """The precondition keeps orphans out (Q32); this is the safeguard
    behind it, so the precondition is skipped here."""
    monkeypatch.setattr(Runner, "_check_embeddings", lambda self: None)
    orphan = row_of(PAGES[0], index=9, vector=[1.0, 0.0, 0.0, 0.0])
    fill(run_settings, orphans=[orphan])
    summary = index_run(run_settings)
    assert summary.status == "completed"
    assert summary.orphans == ((orphan.chunk_id, 0),)
    assert client.count(collection, exact=True).count == len(ROWS)
    about = [r for r in logged.records if orphan.chunk_id in r.getMessage()]
    assert [r.levelno for r in about] == [logging.WARNING, logging.WARNING]
    assert about[0].getMessage().startswith("skipping orphaned row")
    assert about[1].getMessage().startswith("skipped 1 orphaned row(s)")
    [line] = [r for r in logged.records if r.getMessage().startswith("completed in")]
    assert "1 orphaned rows skipped" in line.getMessage()


def reembed(settings: Settings, row: StoredVector) -> None:
    with EmbeddingsStore(settings.embeddings_path) as embeddings:
        embeddings.upsert(row.chunk_id, [embedding_row_for(changed_vector(row))])


@pytest.mark.network
def test_an_unanswered_save_question_fails_before_writing(run_settings, collection):
    fill(run_settings)
    index_run(run_settings)
    reembed(run_settings, ROWS[2])
    summary = index_run(run_settings)
    assert (summary.status, summary.exit_code) == ("failed", 1)
    assert "--save-old or --no-save-old" in summary.error
    assert summary.plan.changed == {PAGES[1]}
    assert summary.written is None
    # Nothing was written: the page is still due for replacement.
    assert index_run(run_settings, save_old=False).plan.changed == {PAGES[1]}


@pytest.mark.network
def test_save_old_saves_before_writing(run_settings, collection, restore, name):
    fill(run_settings)
    index_run(run_settings)
    reembed(run_settings, ROWS[2])
    summary = index_run(run_settings, save_old=True)
    assert summary.status == "completed"
    [path] = summary.saved
    assert path.parent == run_settings.snapshots_dir
    assert summary.written == (1, 1, 2)
    assert restore(path, f"{name}_restored") == len(ROWS)


@pytest.mark.network
def test_a_qdrant_error_fails_the_run_without_retrying(run_settings):
    fill(run_settings)
    summary = index_run(replace(run_settings, api_key="not-the-key"))
    assert (summary.status, summary.exit_code) == ("failed", 1)
    assert summary.retries == 0
    assert summary.error


def test_rows_that_disagree_on_dimensions_stop_before_qdrant(settings, tmp_path):
    closed = replace(settings, url="http://localhost:1", data_dir=tmp_path)
    wide = row_of(PAGES[0], index=5, vector=[1.0] * 8)
    fill(closed, [*ROWS, wide])
    summary = index_run(closed)
    assert summary.status == "failed"
    assert (
        "disagree on dimensions (6 rows at 4, 1 rows at 8); "
        f"not at 4: {wide.chunk_id} (sub_index 0) at 8" in summary.error
    )
    assert summary.retries == 0  # Qdrant was never called
    assert summary.collection is None


def test_more_odd_rows_than_named_are_counted(settings, tmp_path):
    closed = replace(settings, url="http://localhost:1", data_dir=tmp_path)
    wide = [row_of(PAGES[0], index=i, vector=[1.0] * 8) for i in range(10, 17)]
    fill(closed, [*ROWS, *wide])
    summary = index_run(closed)
    # The 7 rows at 8 outnumber the 6 at 4, so the rows at 4 are the odd ones.
    assert "(6 rows at 4, 7 rows at 8); not at 8: " in summary.error
    assert summary.error.count("(sub_index 0) at 4") == 5
    assert summary.error.endswith("and 1 more")


# -- preconditions: `embed verify --complete` (Q32) -----------------------------


def failed_precondition(settings, tmp_path, edit) -> "RunSummary":
    """Fill the stores, `edit` them, then run against a closed port: a
    precondition that fails stops the run before any Qdrant call."""
    closed = replace(settings, url="http://localhost:1", data_dir=tmp_path)
    fill(closed)
    with open_stores(closed) as stores:
        edit(*stores)
    return index_run(closed)


def test_a_stale_row_stops_the_run_before_anything(settings, tmp_path):
    def reparse(embeddings, chunks, manifest):
        changed = replace(chunk_for(ROWS[0]), text="re-parsed text")
        chunks.replace_page_chunks(PAGES[0], [changed, chunk_for(ROWS[1])])

    summary = failed_precondition(settings, tmp_path, reparse)
    assert (summary.status, summary.exit_code) == ("failed", 1)
    assert summary.error.startswith(
        "`embed verify --complete` is not clean (1 stale-hash), so nothing was loaded"
    )
    assert f"First: [stale-hash] {ROWS[0].chunk_id}." in summary.error
    assert (summary.collection, summary.plan, summary.retries) == (None, None, 0)


def test_an_orphan_stops_the_run_before_anything(settings, tmp_path):
    def orphan(embeddings, chunks, manifest):
        chunks.replace_page_chunks(PAGES[0], [chunk_for(ROWS[0])])

    summary = failed_precondition(settings, tmp_path, orphan)
    assert "(1 orphaned-rows)" in summary.error
    assert f"[orphaned-rows] {ROWS[1].chunk_id}" in summary.error


def test_a_chunk_not_embedded_or_embedded_by_another_model_stops_the_run(
    settings, tmp_path
):
    def gaps(embeddings, chunks, manifest):
        extra = make_chunk(7)
        chunks.replace_page_chunks(
            PAGES[0],
            [
                chunk_for(ROWS[0]),
                chunk_for(ROWS[1]),
                replace(extra, chunk_id=f"{PAGES[0]}#7", url=PAGES[0]),
            ],
        )
        other = replace(embedding_row_for(ROWS[2]), model="text-embedding-3-large")
        embeddings.upsert(ROWS[2].chunk_id, [other])

    summary = failed_precondition(settings, tmp_path, gaps)
    assert "(1 model-mismatch, 1 not-embedded)" in summary.error


def test_a_long_list_of_violations_is_cut_short(settings, tmp_path):
    def reparse_everything(embeddings, chunks, manifest):
        for url in PAGES:
            page = [replace(chunk_for(r), text="new") for r in ROWS if r.url == url]
            chunks.replace_page_chunks(url, page)

    summary = failed_precondition(settings, tmp_path, reparse_everything)
    assert f"({len(ROWS)} stale-hash)" in summary.error
    assert (
        f"; and {len(ROWS) - 5} more. `embed verify --complete` lists them all"
        in summary.error
    )


@pytest.mark.network
def test_failed_rows_do_not_stop_the_run(run_settings, collection, client):
    """A failed row set counts as covered in `embed verify`: it is not
    loaded, and the run goes ahead."""
    fill(run_settings)
    failed = replace(
        embedding_row_for(ROWS[0]),
        status="failed",
        dimensions=None,
        embedding=None,
        error="boom",
    )
    with EmbeddingsStore(run_settings.embeddings_path) as embeddings:
        embeddings.upsert(ROWS[0].chunk_id, [failed])
    summary = index_run(run_settings)
    assert summary.status == "completed"
    assert client.count(collection, exact=True).count == len(ROWS) - 1


def test_no_rows_of_the_model_stops_the_run(settings, tmp_path):
    closed = replace(settings, url="http://localhost:1", data_dir=tmp_path)
    summary = index_run(closed)
    assert summary.status == "failed"
    assert "no ok rows of openai/text-embedding-3-small" in summary.error


def test_exit_codes_match_embed_run():
    assert EXIT_CODES == {"completed": 0, "interrupted": 130, "failed": 1}


# -- old models' collections (Q37) ------------------------------------------------


def make_old_collection(client, name: str) -> None:
    """A collection with one point, standing in for an old model's."""
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )
    client.upsert(
        name, points=[models.PointStruct(id=1, vector=[1, 0, 0, 0])], wait=True
    )


@pytest.mark.network
def test_old_collections_are_deleted_once_the_new_one_verifies(
    run_settings, collection, client, name, logged
):
    old = f"{name}_openai_text_embedding_3_large_4"
    unrelated = f"{name}x"  # starts with the prefix, but not `<prefix>_`
    make_old_collection(client, old)
    make_old_collection(client, unrelated)
    fill(run_settings)
    summary = index_run(run_settings, save_old=False)
    assert (summary.status, summary.exit_code) == ("completed", 0)
    assert summary.deleted == (old,)
    assert not client.collection_exists(old)
    assert client.collection_exists(unrelated)
    assert client.count(collection, exact=True).count == len(ROWS)
    messages = [r.getMessage() for r in logged.records]
    verifying = next(i for i, m in enumerate(messages) if m.startswith("verifying"))
    deleting = messages.index(f"deleted the old collection {old}")
    assert verifying < deleting
    [line] = [m for m in messages if m.startswith("completed in")]
    assert f"old collections deleted: {old}" in line


@pytest.mark.network
def test_a_failed_verification_keeps_the_old_collections_and_exits_1(
    run_settings, client, name, monkeypatch
):
    old = f"{name}_openai_text_embedding_3_large_4"
    make_old_collection(client, old)
    fill(run_settings)
    # Only verification asks for the status; red is a violation there.
    monkeypatch.setattr(QdrantStore, "collection_status", lambda self, name: "red")
    summary = index_run(run_settings, save_old=False)
    assert (summary.status, summary.exit_code) == ("failed", 1)
    assert summary.deleted == ()
    assert "failed `index verify --complete` (1 index-red)" in summary.error
    assert f"the old collection(s) were kept: {old}" in summary.error
    assert client.collection_exists(old)
    assert summary.written == (0, len(PAGES), len(ROWS))  # the load itself went in


@pytest.mark.network
def test_ctrl_c_during_verification_keeps_the_old_collections(
    run_settings, client, name, monkeypatch
):
    from databricks_docs_vectordb.indexing import verify

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    old = f"{name}_openai_text_embedding_3_large_4"
    make_old_collection(client, old)
    fill(run_settings)
    monkeypatch.setattr(verify, "verify_collection", interrupted)
    summary = index_run(run_settings, save_old=False)
    assert (summary.status, summary.exit_code) == ("interrupted", 130)
    assert summary.deleted == ()
    assert client.collection_exists(old)


@pytest.mark.network
def test_with_no_old_collections_nothing_is_verified(run_settings, monkeypatch):
    from databricks_docs_vectordb.indexing import verify

    def never(*args, **kwargs):
        raise AssertionError("verified with nothing to delete")

    monkeypatch.setattr(verify, "verify_collection", never)
    fill(run_settings)
    assert index_run(run_settings).status == "completed"
