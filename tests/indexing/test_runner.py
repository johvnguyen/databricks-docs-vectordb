import array
import logging
import math
import random
import re
from dataclasses import replace

import pytest
from qdrant_client import models

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
    Plan,
    SaveQuestionUnanswered,
    compare,
    point_id,
    save_affected,
    should_save,
    skip_orphans,
    to_point,
    unit_vector,
)
from databricks_docs_vectordb.indexing.store import QdrantStore, collection_name
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore

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
        chunk_text_hash="cd" * 32,
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
