import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest

from databricks_docs_vectordb.embedding.runner import text_hash
from databricks_docs_vectordb.embedding.store import (
    EmbeddingRow,
    EmbeddingsStore,
    ModelInfo,
    pack_embedding,
)
from databricks_docs_vectordb.indexing import check
from databricks_docs_vectordb.indexing.check import (
    FILTERED,
    QUESTIONS,
    TOP_K,
    CheckReport,
    FilteredResult,
    QueryHit,
    QueryResult,
    Question,
    check_collection,
    format_report,
)
from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.runner import RunFailed, Runner
from databricks_docs_vectordb.indexing.store import QdrantStore, collection_name
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore
from databricks_docs_vectordb.scraping.config import SOURCES
from databricks_docs_vectordb.scraping.manifest import Manifest

MODEL = ModelInfo("openai", "text-embedding-3-small", None)
PAGES = [f"https://docs.databricks.com/aws/en/p{i}/" for i in range(3)]
CHUNKS_PER_PAGE = 2
POINTS = len(PAGES) * CHUNKS_PER_PAGE
TEST_QUESTIONS = tuple(Question(f"about page {i}", url) for i, url in enumerate(PAGES))


def vector_of(chunk_id: str) -> list[float]:
    rng = random.Random(chunk_id)
    return [rng.gauss(0, 1) for _ in range(4)]


def make_chunk(url: str, index: int) -> Chunk:
    text = f"chunk {index} of {url}"
    return Chunk(
        chunk_id=f"{url}#{index}",
        url=url,
        source="aws-en",
        cloud="aws",
        chunk_index=index,
        prev_chunk_id=None,
        next_chunk_id=None,
        heading_path=f"Heading {index}",
        section_path="Section",
        doc_title="Title",
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


def make_row(chunk: Chunk) -> EmbeddingRow:
    vector = vector_of(chunk.chunk_id)
    return EmbeddingRow(
        chunk_id=chunk.chunk_id,
        sub_index=0,
        sub_count=1,
        url=chunk.url,
        source=chunk.source,
        cloud=chunk.cloud,
        chunk_index=chunk.chunk_index,
        prev_chunk_id=None,
        next_chunk_id=None,
        heading_path=chunk.heading_path,
        section_path=chunk.section_path,
        doc_title=chunk.doc_title,
        description=None,
        last_updated=None,
        page_sha256=chunk.page_sha256,
        chunk_text_hash=text_hash(chunk.text),
        embedded_text=chunk.text,
        token_count=8,
        provider=MODEL.provider,
        model=MODEL.model,
        model_version=None,
        dimensions=len(vector),
        embedding=pack_embedding(vector),
        status="ok",
        attempts=1,
        error=None,
    )


@contextmanager
def stores(settings: Settings) -> Iterator[tuple]:
    with (
        EmbeddingsStore(settings.embeddings_path) as embeddings,
        ChunkStore(settings.parsed_path) as chunks,
        Manifest(settings.manifest_path) as manifest,
    ):
        yield embeddings, chunks, manifest


def fill(settings: Settings) -> None:
    with stores(settings) as (embeddings, chunks, manifest):
        for url in PAGES:
            page = [make_chunk(url, i) for i in range(CHUNKS_PER_PAGE)]
            chunks.replace_page_chunks(url, page)
            for chunk in page:
                embeddings.upsert(chunk.chunk_id, [make_row(chunk)])
        manifest.add_pending("aws-en", PAGES)
        for url in PAGES:
            manifest.record_ok(url, http_status=200, sha256="ab" * 32, size_bytes=1)


def load(settings: Settings) -> None:
    with stores(settings) as (embeddings, chunks, manifest), QdrantStore(settings) as q:
        runner = Runner(
            settings, MODEL, embeddings, chunks, manifest, q, interactive=False
        )
        assert runner.run().status == "completed"


def run_check(settings: Settings, embed=None) -> CheckReport:
    with EmbeddingsStore(settings.embeddings_path) as embeddings:
        with QdrantStore(settings) as q:
            return check_collection(q, settings, MODEL, embeddings, embed=embed)


def embed_as_page(texts):
    """A stand-in for the provider: each test question gets the stored
    vector of its page's first chunk, scaled away from unit length."""
    by_text = {q.text: q.expected_url for q in TEST_QUESTIONS}
    return [[3 * x for x in vector_of(f"{by_text[t]}#0")] for t in texts]


@pytest.fixture
def csettings(settings, name, tmp_path) -> Settings:
    return replace(settings, collection_prefix=name, data_dir=tmp_path)


@pytest.fixture
def loaded(csettings) -> str:
    fill(csettings)
    load(csettings)
    return collection_name(csettings.collection_prefix, MODEL, 4)


@pytest.fixture
def test_questions(monkeypatch):
    monkeypatch.setattr(check, "QUESTIONS", TEST_QUESTIONS)


# -- against the sidecar ------------------------------------------------------------


@pytest.mark.network
def test_self_retrieval_and_recall_on_a_loaded_collection(csettings, loaded):
    report = run_check(csettings)
    assert (report.collection, report.points) == (loaded, POINTS)
    assert len(report.sample_ids) == POINTS  # fewer points than the sample size
    assert report.self_top1 == POINTS
    assert report.self_misses == []
    assert report.recall == 1.0
    assert not report.recall_warning
    assert report.default_ms > 0 and report.exact_ms > 0


@pytest.mark.network
def test_the_sample_is_seeded_and_capped_at_the_sample_size(csettings, loaded):
    small = replace(csettings, recall_sample_size=4)
    first, second = run_check(small), run_check(small)
    assert len(first.sample_ids) == 4
    assert first.sample_ids == second.sample_ids


@pytest.mark.network
def test_recall_below_the_threshold_warns(csettings, loaded, monkeypatch):
    search = QdrantStore.search

    def lossy(self, *args, exact=False, **kwargs):
        hits = search(self, *args, exact=exact, **kwargs)
        return hits if exact else hits[:-1]  # default search misses one

    monkeypatch.setattr(QdrantStore, "search", lossy)
    report = run_check(csettings)
    assert report.recall == pytest.approx((POINTS - 1) / POINTS)
    assert report.recall_warning
    assert "warning: recall@10 is below 0.95" in format_report(report)


@pytest.mark.network
def test_a_point_that_is_not_its_own_top1_is_named(csettings, loaded, monkeypatch):
    search = QdrantStore.search

    def reversed_default(self, *args, exact=False, **kwargs):
        hits = search(self, *args, exact=exact, **kwargs)
        return hits if exact else hits[::-1]

    monkeypatch.setattr(QdrantStore, "search", reversed_default)
    report = run_check(csettings)
    assert report.self_top1 == 0
    assert len(report.self_misses) == POINTS
    assert all(": top-1 was " in miss for miss in report.self_misses)
    assert report.recall == 1.0  # the same points, in another order


@pytest.mark.network
def test_without_a_key_text_and_filtered_queries_are_skipped_with_a_note(
    csettings, loaded
):
    report = run_check(csettings, embed=None)
    assert report.queries == report.filtered == []
    assert "real text and filtered queries skipped" in report.notes[0]


@pytest.mark.network
def test_text_query_hits_are_read_back_from_embeddings(
    csettings, loaded, test_questions
):
    report = run_check(csettings, embed=embed_as_page)
    assert [r.expected_rank for r in report.queries] == [1, 1, 1]
    for result in report.queries:
        top = result.hits[0]
        assert top.chunk_id == f"{result.question.expected_url}#0"
        assert (top.url, top.heading_path) == (
            result.question.expected_url,
            "Heading 0",
        )
        assert len(result.hits) == POINTS  # fewer points than TOP_K


@pytest.mark.network
def test_questions_are_normalized_like_the_stored_vectors(
    settings, name, tmp_path, test_questions
):
    dot = replace(settings, collection_prefix=name, data_dir=tmp_path, distance="Dot")
    fill(dot)
    load(dot)
    report = run_check(dot, embed=embed_as_page)
    # Under Dot, Qdrant normalizes nothing: a query left at 3x would score 3.
    assert [r.hits[0].score for r in report.queries] == pytest.approx([1.0] * 3)


@pytest.mark.network
def test_every_hit_of_a_filtered_query_matches_its_filter(
    csettings, loaded, test_questions
):
    report = run_check(csettings, embed=embed_as_page)
    assert [(f.field, f.value) for f in report.filtered] == [
        pair
        for i, url in enumerate(PAGES)
        for pair in (("section", f"aws-en/p{i}"), ("url", url))
    ]
    assert all(f.ok for f in report.filtered)
    assert all(f.hits == CHUNKS_PER_PAGE for f in report.filtered)


@pytest.mark.network
def test_a_missing_collection_raises(csettings):
    fill(csettings)
    with pytest.raises(RunFailed, match="does not exist yet"):
        run_check(csettings)


@pytest.mark.network
def test_query_embeddings_of_another_size_raise(csettings, loaded, test_questions):
    with pytest.raises(RunFailed, match=r"have \[3\] dimensions, but .* has 4"):
        run_check(csettings, embed=lambda texts: [[1.0, 0.0, 0.0]] * len(texts))


@pytest.mark.network
def test_an_empty_collection_has_nothing_to_sample(csettings):
    fill(csettings)
    with QdrantStore(csettings) as q:
        q.create_collection(
            collection_name(csettings.collection_prefix, MODEL, 4), dimensions=4
        )
    report = run_check(csettings)
    assert report.recall is None
    assert "the collection is empty" in report.notes[0]


# -- without the sidecar ------------------------------------------------------------


def test_every_question_is_on_a_configured_source():
    prefixes = tuple(s.url_prefix for s in SOURCES)
    assert all(q.expected_url.startswith(prefixes) for q in QUESTIONS)
    assert 0 < FILTERED <= len(QUESTIONS)


def report_with_everything() -> CheckReport:
    hit = QueryHit(0.71, f"{PAGES[0]}#0", 0, PAGES[0], "Heading 0")
    return CheckReport(
        collection="databricks_docs_x_4",
        points=6,
        recall_warn_below=0.95,
        sample_ids=("a", "b"),
        self_top1=1,
        self_misses=["b (sub_index 0): top-1 was a (sub_index 0), score 0.9000"],
        recall=0.9,
        default_ms=5.9,
        exact_ms=18.5,
        queries=[
            QueryResult(TEST_QUESTIONS[0], [hit]),
            QueryResult(TEST_QUESTIONS[1], [hit]),
        ],
        filtered=[
            FilteredResult(TEST_QUESTIONS[0], "url", PAGES[0], 1, []),
            FilteredResult(TEST_QUESTIONS[0], "section", "aws-en/p0", 2, ["aws-en/p1"]),
        ],
        notes=["something worth knowing"],
    )


def test_the_report_covers_all_four_checks():
    text = format_report(report_with_everything())
    assert "1. self-retrieval: 1 of 2 sampled points are their own top-1" in text
    assert "   b (sub_index 0): top-1 was a (sub_index 0), score 0.9000" in text
    assert f"2. index recall@{TOP_K}: 0.9000 over 2 searches" in text
    assert "median 5.9 ms against 18.5 ms" in text
    assert "   warning: recall@10 is below 0.95" in text
    assert '   "about page 0": expected page at rank 1' in text
    assert '   "about page 1": expected page not found' in text
    assert f"      1. 0.7100  {PAGES[0]}  Heading 0" in text
    assert "the expected page came back for 1 of 2 questions" in text
    assert f'"about page 0" with url = {PAGES[0]}: 1 hit(s), every hit matches' in text
    assert (
        "with section = aws-en/p0: 2 hit(s), 1 hit(s) do not match: aws-en/p1" in text
    )
    assert text.endswith("note: something worth knowing")


def test_an_empty_heading_path_shows_as_a_dash():
    report = report_with_everything()
    report.queries = [
        QueryResult(TEST_QUESTIONS[0], [QueryHit(0.5, "c", 0, PAGES[0], "")])
    ]
    assert f"      1. 0.5000  {PAGES[0]}  -" in format_report(report)
