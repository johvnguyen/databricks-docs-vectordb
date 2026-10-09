import random
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest
from qdrant_client import models

from databricks_docs_vectordb.embedding.runner import text_hash
from databricks_docs_vectordb.embedding.store import (
    EmbeddingRow,
    EmbeddingsStore,
    ModelInfo,
    pack_embedding,
)
from databricks_docs_vectordb.indexing import store as store_module
from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.runner import RunFailed, Runner, point_id
from databricks_docs_vectordb.indexing.store import QdrantStore, collection_name
from databricks_docs_vectordb.indexing.verify import (
    LIST_LIMIT,
    SectionCounts,
    VerifyReport,
    format_report,
    verify_collection,
)
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore
from databricks_docs_vectordb.scraping.manifest import Manifest
from databricks_docs_vectordb.scraping.verify import Finding

MODEL = ModelInfo("openai", "text-embedding-3-small", None)
PAGES = [f"https://docs.databricks.com/aws/en/p{i}/" for i in range(3)]
CHUNKS_PER_PAGE = 2
ROWS = len(PAGES) * CHUNKS_PER_PAGE
EDITED = f"{PAGES[1]}#0"  # the row the tests tamper with
EDITED_ID = point_id(EDITED, 0)
WHERE = f"{EDITED} (sub_index 0)"


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
        heading_path="",
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
    rng = random.Random(chunk.chunk_id)
    vector = [rng.gauss(0, 1) for _ in range(4)]
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
    """Every page's chunks, an ok embedding per chunk, every page `ok`."""
    with stores(settings) as (embeddings, chunks, manifest):
        for url in PAGES:
            page = [make_chunk(url, i) for i in range(CHUNKS_PER_PAGE)]
            chunks.replace_page_chunks(url, page)
            for chunk in page:
                embeddings.upsert(chunk.chunk_id, [make_row(chunk)])
        manifest.add_pending("aws-en", PAGES)
        for url in PAGES:
            manifest.record_ok(url, http_status=200, sha256="ab" * 32, size_bytes=1)


def load(settings: Settings, **run) -> None:
    with stores(settings) as (embeddings, chunks, manifest), QdrantStore(settings) as q:
        runner = Runner(
            settings, MODEL, embeddings, chunks, manifest, q, interactive=False
        )
        assert runner.run(**run).status == "completed"


def check(settings: Settings, *, complete: bool = False) -> VerifyReport:
    with stores(settings) as (embeddings, chunks, manifest), QdrantStore(settings) as q:
        return verify_collection(
            q, settings, MODEL, embeddings, chunks, manifest, complete=complete
        )


@pytest.fixture
def vsettings(settings, name, tmp_path) -> Settings:
    """The throwaway prefix, with the stores in tmp_path."""
    return replace(settings, collection_prefix=name, data_dir=tmp_path)


@pytest.fixture
def collection(name) -> str:
    return collection_name(name, MODEL, 4)


@pytest.fixture
def loaded(vsettings, collection) -> str:
    fill(vsettings)
    load(vsettings)
    return collection


# -- against the sidecar ------------------------------------------------------------


@pytest.mark.network
def test_a_full_load_verifies_clean_even_with_complete(vsettings, loaded):
    report = check(vsettings, complete=True)
    assert report.ok, format_report(report)
    assert (report.exists, report.status) == (True, "green")
    assert (report.points, report.rows, report.missing) == (ROWS, ROWS, 0)
    assert report.sections == {
        f"aws-en/p{i}": SectionCounts(rows=2, points=2) for i in range(3)
    }


@pytest.mark.network
def test_after_a_refresh_it_verifies_clean(vsettings, loaded):
    load(vsettings, refresh=True, save_old=False)
    assert check(vsettings, complete=True).ok


@pytest.mark.network
def test_a_missing_point_is_a_gap_only_with_complete(vsettings, loaded, client):
    client.delete(
        loaded, points_selector=models.PointIdsList(points=[EDITED_ID]), wait=True
    )
    report = check(vsettings)
    assert report.ok
    assert (report.points, report.rows, report.missing) == (ROWS - 1, ROWS, 1)

    report = check(vsettings, complete=True)
    assert [(f.kind, f.where) for f in report.violations] == [("missing-point", WHERE)]
    assert report.sections["aws-en/p1"] == SectionCounts(
        rows=2, points=1, missing=1, violations=1
    )


@pytest.mark.network
@pytest.mark.parametrize("complete", [False, True])
def test_a_changed_vector_is_a_violation_either_way(
    vsettings, loaded, client, complete
):
    client.update_vectors(
        loaded,
        points=[models.PointVectors(id=EDITED_ID, vector=[1.0, 0.0, 0.0, 0.0])],
        wait=True,
    )
    [finding] = check(vsettings, complete=complete).violations
    assert (finding.kind, finding.where) == ("vector-mismatch", WHERE)
    assert EDITED_ID in finding.detail


@pytest.mark.network
@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        (
            {"section": "aws-en/elsewhere"},
            "section is 'aws-en/elsewhere', expected 'aws-en/p1'",
        ),
        ({"sub_index": "0"}, "sub_index is '0', expected 0"),
        ({"url": PAGES[0]}, f"url is '{PAGES[0]}', expected '{PAGES[1]}'"),
    ],
    ids=["section", "sub_index-type", "url"],
)
def test_a_changed_payload_field_is_a_violation(
    vsettings, loaded, client, payload, detail
):
    client.set_payload(loaded, payload=payload, points=[EDITED_ID], wait=True)
    [finding] = check(vsettings).violations
    assert (finding.kind, finding.where) == ("payload-mismatch", WHERE)
    assert detail in finding.detail


@pytest.mark.network
def test_a_point_with_no_row_is_extra(vsettings, loaded, client):
    ghost = "https://docs.databricks.com/aws/en/ghost/#0"
    payload = {
        "chunk_id": ghost,
        "sub_index": 0,
        "url": ghost[:-2],
        "section": "aws-en/ghost",
    }
    random_id = str(uuid.uuid4())
    client.upsert(
        loaded,
        points=[
            models.PointStruct(id=random_id, vector=[1, 0, 0, 0], payload=payload),
            models.PointStruct(
                id=point_id(ghost, 0), vector=[1, 0, 0, 0], payload=payload
            ),
        ],
        wait=True,
    )
    report = check(vsettings)
    findings = {f.detail.split(":")[0]: f for f in report.violations}
    assert {f.kind for f in report.violations} == {"extra-point"}
    by_id = {f"point {random_id}", f"point {point_id(ghost, 0)}"}
    assert set(findings) == by_id
    random_one = findings[f"point {random_id}"]
    assert random_one.where == f"{ghost} (sub_index 0)"
    assert "no ok row of the configured model has this ID" in random_one.detail
    assert "not the UUID5" in random_one.detail
    assert "not the UUID5" not in findings[f"point {point_id(ghost, 0)}"].detail
    assert report.sections["aws-en/ghost"] == SectionCounts(points=2, violations=2)


@pytest.mark.network
def test_points_of_a_page_not_ok_in_the_manifest_are_extra(vsettings, loaded):
    with Manifest(vsettings.manifest_path) as manifest:
        manifest.mark_removed("aws-en", [PAGES[0], PAGES[2]])
    report = check(vsettings, complete=True)
    assert report.rows == ROWS - CHUNKS_PER_PAGE
    assert {f.kind for f in report.violations} == {"extra-point"}
    assert len(report.violations) == CHUNKS_PER_PAGE
    assert all(
        "its page is not ok in the manifest" in f.detail for f in report.violations
    )


@pytest.mark.network
def test_the_point_of_an_orphaned_row_is_extra(vsettings, loaded):
    with ChunkStore(vsettings.parsed_path) as chunks:
        chunks.replace_page_chunks(PAGES[1], [make_chunk(PAGES[1], 0)])
    [finding] = check(vsettings).violations
    assert (finding.kind, finding.where) == (
        "extra-point",
        f"{PAGES[1]}#1 (sub_index 0)",
    )
    assert "its row is an orphan" in finding.detail


@pytest.mark.network
def test_settings_that_differ_are_a_collection_violation(vsettings, loaded):
    dot = replace(vsettings, distance="Dot")
    with stores(dot) as (embeddings, chunks, manifest), QdrantStore(dot) as q:
        report = verify_collection(q, dot, MODEL, embeddings, chunks, manifest)
    assert report.violations == [
        Finding("collection", loaded, "distance is Cosine, configured Dot")
    ]


@pytest.mark.network
def test_with_no_collection_every_row_is_a_gap(vsettings, collection):
    fill(vsettings)
    report = check(vsettings)
    assert report.ok
    assert (report.exists, report.status, report.missing) == (False, None, ROWS)
    assert f"{collection} does not exist yet" in report.notes[0]
    complete = check(vsettings, complete=True)
    assert [f.kind for f in complete.violations] == ["missing-point"] * ROWS


@pytest.mark.network
@pytest.mark.parametrize(
    ("status", "violations", "notes"), [("red", 1, 0), ("yellow", 0, 1), ("grey", 0, 1)]
)
def test_red_is_a_violation_and_still_optimizing_a_note(
    vsettings, loaded, monkeypatch, status, violations, notes
):
    monkeypatch.setattr(QdrantStore, "collection_status", lambda self, name: status)
    report = check(vsettings)
    assert len(report.violations) == violations
    assert len([n for n in report.notes if status in n]) == notes


@pytest.mark.network
def test_every_point_is_read_across_scroll_requests(vsettings, loaded, monkeypatch):
    monkeypatch.setattr(store_module, "SCROLL_LIMIT", 2)
    report = check(vsettings, complete=True)
    assert report.ok
    assert report.points == ROWS


# -- without the sidecar ------------------------------------------------------------


def test_no_rows_of_the_model_raise_before_qdrant(settings, tmp_path):
    closed = replace(settings, url="http://localhost:1", data_dir=tmp_path)
    with pytest.raises(RunFailed, match="no ok rows"):
        check(closed)


def make_report(**fields) -> VerifyReport:
    report = VerifyReport("databricks_docs_x_4", exists=True, status="green")
    for key, value in fields.items():
        setattr(report, key, value)
    return report


def test_a_clean_report_says_ok():
    text = format_report(make_report(points=6, rows=6))
    assert "checked 6 point(s) in databricks_docs_x_4 against 6 current row(s)" in text
    assert "coverage (current rows): 6 with a point, 0 without" in text
    assert text.endswith("\nOK")


def test_gaps_without_complete_are_a_note():
    text = format_report(make_report(points=3, rows=6, missing=3))
    assert (
        "note: 3 current row(s) have no point (not violations unless --complete)"
        in text
    )
    assert text.endswith("\nOK")


def test_violations_are_grouped_by_kind_and_long_lists_cut_short():
    gaps = [
        Finding("missing-point", f"row {i}", "no point") for i in range(LIST_LIMIT + 5)
    ]
    vector = Finding("vector-mismatch", "row x", "point p: not the BLOB")
    text = format_report(make_report(violations=[*gaps, vector]))
    assert "[missing-point] ... and 5 more" in text
    assert "  [vector-mismatch] row x: point p: not the BLOB" in text
    assert text.endswith(f"FAILED: {LIST_LIMIT + 6} violations")


def test_by_section_adds_a_table():
    report = make_report(points=3, rows=4, missing=1)
    report.sections["aws-en/sql"] = SectionCounts(rows=4, points=3, missing=1)
    text = format_report(report, by_section=True)
    assert "per-section (current rows, points, without a point, violations):" in text
    assert ["aws-en/sql", "4", "3", "1", "0"] in [
        line.split() for line in text.splitlines()
    ]
    assert "per-section" not in format_report(report)
