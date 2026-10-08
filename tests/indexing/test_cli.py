import io
import random
import sys
from pathlib import Path

import pytest
from qdrant_client import models

from databricks_docs_vectordb.cli import IndexingSettings, main
from databricks_docs_vectordb.embedding.store import (
    EmbeddingRow,
    EmbeddingsStore,
    pack_embedding,
)
from databricks_docs_vectordb.indexing.runner import point_id
from databricks_docs_vectordb.indexing.store import QdrantStore
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore
from databricks_docs_vectordb.scraping.manifest import Manifest

PAGES = [f"https://docs.databricks.com/aws/en/p{i}/" for i in range(3)]
CHUNKS_PER_PAGE = 2
POINTS = len(PAGES) * CHUNKS_PER_PAGE


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


def make_row(chunk: Chunk, *, seed: int = 0) -> EmbeddingRow:
    rng = random.Random(f"{chunk.chunk_id} {seed}")
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
        chunk_text_hash="cd" * 32,
        embedded_text=chunk.text,
        token_count=8,
        provider="openai",
        model="text-embedding-3-small",
        model_version=None,
        dimensions=len(vector),
        embedding=pack_embedding(vector),
        status="ok",
        attempts=1,
        error=None,
    )


def fill(data: Path) -> None:
    """Every page's chunks in parsed.sqlite, an ok 4-dimensional embedding
    per chunk, and every page `ok` in the manifest."""
    with (
        EmbeddingsStore(data / "embeddings.sqlite") as embeddings,
        ChunkStore(data / "parsed.sqlite") as chunks,
        Manifest(data / "manifest.sqlite") as manifest,
    ):
        for url in PAGES:
            page = [make_chunk(url, i) for i in range(CHUNKS_PER_PAGE)]
            chunks.replace_page_chunks(url, page)
            for chunk in page:
                embeddings.upsert(chunk.chunk_id, [make_row(chunk)])
        manifest.add_pending("aws-en", PAGES)
        for url in PAGES:
            manifest.record_ok(url, http_status=200, sha256="ab" * 32, size_bytes=1)


def reembed(data: Path, url: str = PAGES[1]) -> None:
    with EmbeddingsStore(data / "embeddings.sqlite") as embeddings:
        chunk = make_chunk(url, 0)
        embeddings.upsert(chunk.chunk_id, [make_row(chunk, seed=1)])


def force_settings(monkeypatch, **overrides) -> None:
    """Make every `IndexingSettings.load()` in the CLI apply `overrides`."""
    load = IndexingSettings.load
    monkeypatch.setattr(
        IndexingSettings,
        "load",
        staticmethod(lambda **kwargs: load(**{**overrides, **kwargs})),
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    """No real .env, data in a temp dir, and an API key for a Qdrant that
    tests here never reach."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABRICKS_DOCS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:1")
    return tmp_path / "data"


@pytest.fixture
def sidecar(settings, name, env, monkeypatch):
    """`env`, but pointed at the real sidecar, with the throwaway `name` as
    the collection prefix so nothing touches `databricks_docs_*`.
    `settings` and `name` come first, so they are made from the real
    settings before `env` replaces them."""
    monkeypatch.setenv("QDRANT_API_KEY", settings.api_key)
    monkeypatch.setenv("QDRANT_URL", settings.url)
    force_settings(monkeypatch, collection_prefix=name)
    fill(env)
    return env


def collection(name: str) -> str:
    return f"{name}_openai_text_embedding_3_small_4"


# -- help and usage -----------------------------------------------------------------


def test_index_help_lists_its_subcommands(capsys):
    with pytest.raises(SystemExit) as exit:
        main(["index", "--help"])
    assert exit.value.code == 0
    assert "{run,status,verify}" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("command", "flags"),
    [
        ("run", ["--refresh", "--save-old", "--no-save-old"]),
        ("status", ["--by-section"]),
        ("verify", ["--complete", "--by-section"]),
    ],
)
def test_each_subcommand_lists_its_flags(capsys, command, flags):
    with pytest.raises(SystemExit) as exit:
        main(["index", command, "--help"])
    assert exit.value.code == 0
    out = capsys.readouterr().out
    assert all(flag in out for flag in flags)


def test_save_old_and_no_save_old_cannot_be_combined(env, capsys):
    with pytest.raises(SystemExit) as exit:
        main(["index", "run", "--save-old", "--no-save-old"])
    assert exit.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


# -- without the sidecar ------------------------------------------------------------


def test_a_missing_api_key_is_a_clear_error(env, monkeypatch, capsys):
    monkeypatch.delenv("QDRANT_API_KEY")
    assert main(["index", "status"]) == 1
    assert "QDRANT_API_KEY is not set" in capsys.readouterr().err


def test_run_without_embeddings_says_to_embed_first(env, capsys):
    assert main(["index", "run"]) == 1
    assert "embed run" in capsys.readouterr().err


def test_status_without_embeddings_points_at_embed_run(env, capsys):
    assert main(["index", "status"]) == 0
    assert "embed run" in capsys.readouterr().out


def test_status_with_qdrant_unreachable_fails_at_once(env, capsys):
    fill(env)
    assert main(["index", "status"]) == 1
    assert (
        "error: Qdrant at http://localhost:1: ConnectError" in capsys.readouterr().err
    )


def test_run_with_qdrant_unreachable_exits_1_after_its_retries(
    env, monkeypatch, capsys
):
    fill(env)
    force_settings(monkeypatch, max_attempts=2, backoff_base=0.01, backoff_max=0.01)
    assert main(["index", "run"]) == 1
    captured = capsys.readouterr()
    assert "failed: Qdrant at http://localhost:1 did not answer" in captured.out
    [log] = (env / "logs").glob("index-*.log")
    assert "(attempt 2/2)" in log.read_text()


# -- against the sidecar ------------------------------------------------------------


@pytest.mark.network
def test_run_loads_everything_and_a_second_run_changes_nothing(sidecar, name, capsys):
    assert main(["index", "run"]) == 0
    out = capsys.readouterr().out
    assert f"collection: {collection(name)}" in out
    assert f"this run: {POINTS} point(s) upserted; pages: {len(PAGES)} new" in out
    [log] = (sidecar / "logs").glob("index-*.log")
    assert f"log: {log}" in out
    assert "completed in" in log.read_text()

    assert main(["index", "run"]) == 0
    assert (
        f"0 point(s) upserted; pages: 0 new, 0 changed, 0 removed, {len(PAGES)} current"
        in capsys.readouterr().out
    )


@pytest.mark.network
def test_ctrl_c_exits_130_and_leaves_a_log(sidecar, monkeypatch, capsys):
    def interrupt(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(QdrantStore, "upsert", interrupt)
    assert main(["index", "run"]) == 130
    out = capsys.readouterr().out
    assert "interrupted: run the same command again to finish" in out
    [log] = (sidecar / "logs").glob("index-*.log")
    assert "interrupted" in log.read_text()


@pytest.mark.network
def test_an_unanswered_save_question_exits_1_before_writing(
    sidecar, monkeypatch, capsys
):
    assert main(["index", "run"]) == 0
    reembed(sidecar)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # no terminal
    assert main(["index", "run"]) == 1
    out = capsys.readouterr().out
    assert "failed: this run would change" in out
    assert "--save-old or --no-save-old" in out
    assert "write not finished; pages: 0 new, 1 changed" in out


@pytest.mark.network
def test_save_old_saves_a_snapshot_then_writes(sidecar, name, capsys):
    assert main(["index", "run"]) == 0
    reembed(sidecar)
    assert main(["index", "run", "--save-old"]) == 0
    out = capsys.readouterr().out
    [snapshot] = (sidecar / "qdrant_snapshots").iterdir()
    assert snapshot.name.startswith(collection(name))
    assert f"saved: {snapshot}" in out
    assert "2 point(s) upserted; pages: 0 new, 1 changed" in out


@pytest.mark.network
def test_refresh_reloads_every_row(sidecar, capsys):
    assert main(["index", "run"]) == 0
    assert main(["index", "run", "--refresh", "--no-save-old"]) == 0
    out = capsys.readouterr().out
    assert (
        f"this run: {POINTS} point(s) upserted; pages: 0 new, {len(PAGES)} changed"
        in out
    )
    assert not (sidecar / "qdrant_snapshots").exists()


@pytest.mark.network
def test_status_before_any_run_says_not_created_yet(sidecar, name, capsys):
    assert main(["index", "status"]) == 0
    out = capsys.readouterr().out
    assert f"collection: {collection(name)}" in out
    assert "not created yet" in out


@pytest.mark.network
def test_status_after_a_run_counts_points_pages_and_sections(sidecar, capsys):
    main(["index", "run"])
    capsys.readouterr()
    assert main(["index", "status", "--by-section"]) == 0
    out = capsys.readouterr().out
    assert f"{POINTS} point(s) across {len(PAGES)} page(s)" in out
    rows = [line.split() for line in out.splitlines() if "aws-en/p" in line]
    assert rows == [[f"aws-en/p{i}", "1", str(CHUNKS_PER_PAGE)] for i in range(3)]
    assert "other models' collections: none" in out


@pytest.mark.network
def test_status_lists_other_models_collections(sidecar, client, name, capsys):
    old = f"{name}_openai_text_embedding_3_large_4"
    client.create_collection(
        old, vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
    )
    client.upsert(
        old, points=[models.PointStruct(id=1, vector=[1, 0, 0, 0])], wait=True
    )
    assert main(["index", "status"]) == 0
    assert f"  {old}: 1 point(s)" in capsys.readouterr().out


@pytest.mark.network
def test_status_reports_a_settings_mismatch(sidecar, monkeypatch, name, capsys):
    main(["index", "run"])
    capsys.readouterr()
    force_settings(monkeypatch, collection_prefix=name, distance="Dot")
    assert main(["index", "status"]) == 0
    assert (
        "differs from the settings: distance is Cosine, configured Dot"
        in capsys.readouterr().out
    )


# -- index verify -------------------------------------------------------------------


def test_verify_without_embeddings_says_to_embed_first(env, capsys):
    assert main(["index", "verify"]) == 1
    assert "embed run" in capsys.readouterr().err


def test_verify_with_qdrant_unreachable_fails_at_once(env, capsys):
    fill(env)
    assert main(["index", "verify"]) == 1
    assert (
        "error: Qdrant at http://localhost:1: ConnectError" in capsys.readouterr().err
    )


@pytest.mark.network
def test_verify_passes_after_a_run_even_with_complete(sidecar, capsys):
    main(["index", "run"])
    capsys.readouterr()
    assert main(["index", "verify"]) == 0
    assert capsys.readouterr().out.endswith("\nOK\n")
    assert main(["index", "verify", "--complete", "--by-section"]) == 0
    out = capsys.readouterr().out
    rows = [line.split() for line in out.splitlines() if "aws-en/p" in line]
    assert rows == [[f"aws-en/p{i}", "2", "2", "0", "0"] for i in range(3)]
    assert out.endswith("\nOK\n")


@pytest.mark.network
def test_verify_before_any_run_fails_only_with_complete(sidecar, name, capsys):
    assert main(["index", "verify"]) == 0
    assert f"{collection(name)} does not exist yet" in capsys.readouterr().out
    assert main(["index", "verify", "--complete"]) == 1
    assert f"FAILED: {POINTS} violations" in capsys.readouterr().out


@pytest.mark.network
def test_verify_exits_1_naming_a_tampered_point(sidecar, client, name, capsys):
    main(["index", "run"])
    capsys.readouterr()
    tampered = f"{PAGES[1]}#0"
    client.update_vectors(
        collection(name),
        points=[models.PointVectors(id=point_id(tampered, 0), vector=[1, 0, 0, 0])],
        wait=True,
    )
    assert main(["index", "verify"]) == 1
    out = capsys.readouterr().out
    assert f"[vector-mismatch] {tampered} (sub_index 0)" in out
    assert out.endswith("FAILED: 1 violations\n")
