import asyncio
import logging
import os
import signal
import sqlite3
import threading
from pathlib import Path

import httpx
import litellm
import pytest
from litellm import exceptions as llm_errors
from litellm.types.utils import Embedding, EmbeddingResponse

from databricks_docs_vectordb.cli import main
from databricks_docs_vectordb.embedding.config import Settings
from databricks_docs_vectordb.embedding.store import EmbeddingsStore
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore

URL_A = "https://docs.databricks.com/aws/en/delta/"


@pytest.fixture
def env(monkeypatch, tmp_path):
    """No real .env, data in a temp dir. Embedding's backfill-tokens never
    needs a contact address or a provider API key."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABRICKS_DOCS_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


def make_chunk(index: int, **overrides) -> Chunk:
    defaults = {
        "chunk_id": f"{URL_A}#{index}",
        "url": URL_A,
        "source": "aws-en",
        "cloud": "aws",
        "chunk_index": index,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "heading_path": "Delta Lake",
        "section_path": "Tables > Table formats > Delta Lake",
        "doc_title": "What is Delta Lake?",
        "description": "Delta Lake is the optimized storage layer.",
        "text": "hello world",
        "char_count": len("hello world"),
        "token_count": None,
        "token_count_model": None,
        "is_oversized": False,
        "last_updated": "2026-09-11",
        "page_sha256": "ab" * 32,
        "fetched_at": "2026-09-11T00:00:00Z",
    }
    defaults.update(overrides)
    return Chunk(**defaults)


# -- embed backfill-tokens ----------------------------------------------


def test_backfill_tokens_without_a_chunk_store_is_a_clear_error(env, capsys):
    assert main(["embed", "backfill-tokens"]) == 1
    assert "parse run" in capsys.readouterr().err


def test_backfill_tokens_fills_in_every_null_row(env, capsys):
    with ChunkStore(env / "parsed.sqlite") as store:
        store.replace_page_chunks(URL_A, [make_chunk(0, text="hello world")])

    assert main(["embed", "backfill-tokens"]) == 0
    out = capsys.readouterr().out
    assert "1 chunk(s) given a real token_count" in out
    assert "text-embedding-3-small" in out

    with ChunkStore(env / "parsed.sqlite") as store:
        assert store.get_chunks(URL_A)[0].token_count == 2  # "hello world"


def test_backfill_tokens_leaves_text_and_boundaries_untouched(env, capsys):
    with ChunkStore(env / "parsed.sqlite") as store:
        store.replace_page_chunks(URL_A, [make_chunk(0, text="hello world")])

    main(["embed", "backfill-tokens"])

    with ChunkStore(env / "parsed.sqlite") as store:
        chunk = store.get_chunks(URL_A)[0]
        assert chunk.text == "hello world"
        assert chunk.char_count == len("hello world")


def test_running_backfill_tokens_twice_is_a_no_op_the_second_time(env, capsys):
    with ChunkStore(env / "parsed.sqlite") as store:
        store.replace_page_chunks(URL_A, [make_chunk(0, text="hello world")])

    main(["embed", "backfill-tokens"])
    capsys.readouterr()

    assert main(["embed", "backfill-tokens"]) == 0
    assert "nothing to backfill" in capsys.readouterr().out


def test_backfill_tokens_on_an_empty_store_says_so(env, capsys):
    with ChunkStore(env / "parsed.sqlite"):
        pass  # creates the (empty) store, nothing more
    assert main(["embed", "backfill-tokens"]) == 0
    assert "nothing to backfill" in capsys.readouterr().out


# -- embed backfill-tokens --model (testing a different tokenizer) ----------


def test_backfill_tokens_model_flag_tags_rows_with_the_override_not_the_configured_model(
    env, capsys
):
    with ChunkStore(env / "parsed.sqlite") as store:
        store.replace_page_chunks(URL_A, [make_chunk(0, text="hello world")])

    assert main(["embed", "backfill-tokens", "--model", "text-embedding-3-large"]) == 0
    out = capsys.readouterr().out
    assert "model: text-embedding-3-large" in out

    with ChunkStore(env / "parsed.sqlite") as store:
        chunk = store.get_chunks(URL_A)[0]
        assert chunk.token_count_model == "text-embedding-3-large"


def test_backfill_tokens_model_flag_re_tokenizes_a_row_already_done_under_another_model(
    env, capsys
):
    with ChunkStore(env / "parsed.sqlite") as store:
        store.replace_page_chunks(URL_A, [make_chunk(0, text="hello world")])
    main(["embed", "backfill-tokens"])  # tags it with the configured model
    capsys.readouterr()

    # Switching tokenizers via --model re-tokenizes rather than treating the
    # existing count as current.
    assert main(["embed", "backfill-tokens", "--model", "text-embedding-3-large"]) == 0
    out = capsys.readouterr().out
    assert "1 chunk(s) given a real token_count" in out

    # And switching back re-tokenizes again, not a permanent no-op either.
    assert main(["embed", "backfill-tokens"]) == 0
    out = capsys.readouterr().out
    assert "1 chunk(s) given a real token_count" in out
    assert "model: text-embedding-3-small" in out


# =========================================================================
# embed run
# =========================================================================

MODEL = "text-embedding-3-small"
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def page_url(n: int) -> str:
    return f"https://docs.databricks.com/aws/en/p{n:03d}"


def add_pages(data_dir: Path, n: int) -> None:
    with ChunkStore(data_dir / "parsed.sqlite") as store:
        for p in range(n):
            url = page_url(p)
            store.replace_page_chunks(
                url, [make_chunk(0, chunk_id=f"{url}#0", url=url, text=f"text {p}")]
            )


def http_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=REQUEST)


class FakeLiteLLM:
    """Stands in for `litellm.aembedding`, so `embed run` goes through the
    real provider.py, runner, and store. `respond(call_number, input)`
    returns None (succeed) or an exception to raise."""

    def __init__(self, respond=None, *, delay: float = 0.0, model: str = MODEL):
        self.respond = respond or (lambda n, texts: None)
        self.delay = delay
        self.model = model
        self.calls: list[list[str]] = []

    async def __call__(self, *, model, input, timeout, max_retries):
        self.calls.append(list(input))
        if self.delay:
            await asyncio.sleep(self.delay)
        outcome = self.respond(len(self.calls), input)
        if isinstance(outcome, BaseException):
            raise outcome
        response = EmbeddingResponse(
            model=self.model,
            data=[
                Embedding(embedding=[0.1, 0.2, 0.3], index=i, object="embedding")
                for i in range(len(input))
            ],
        )
        response._hidden_params["custom_llm_provider"] = "openai"
        return response


@pytest.fixture
def embed_env(env, monkeypatch):
    """A parsed store with 5 chunks, a (fake) API key, and a fake provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    add_pages(env, 5)
    fake = FakeLiteLLM()
    monkeypatch.setattr(litellm, "aembedding", fake)
    return fake


def patch_settings(monkeypatch, **fields) -> None:
    """Make every `Settings.load` in the CLI apply `fields` on top of
    whatever it would otherwise load (stacks with earlier patches)."""
    original = Settings.load

    def load(cls, **kwargs):
        return original(**{**kwargs, **fields})

    monkeypatch.setattr(Settings, "load", classmethod(load))


@pytest.fixture
def fast_failures(monkeypatch):
    """One attempt per request, one chunk per batch, and a tiny cool-down,
    so failure tests are quick and have enough batches to count."""
    patch_settings(
        monkeypatch,
        max_attempts=1,
        breaker_cooldown=0.01,
        backoff_base=0.001,
        batch_size=1,
    )


def embed_run(*flags: str) -> int:
    return main(["embed", "run", "--rate", "100000", "--tpm", "100000000", *flags])


def log_text(data_dir: Path) -> str:
    (path,) = (data_dir / "logs").glob("embed-*.log")
    return path.read_text()


def stored_statuses(data_dir: Path) -> dict[str, int]:
    with EmbeddingsStore(data_dir / "embeddings.sqlite") as store:
        c = store.counts()
    return {"ok": c.ok, "failed": c.failed, "rejected": c.rejected}


def test_run_embeds_every_chunk_and_reports(env, embed_env, capsys):
    assert embed_run() == 0

    out = capsys.readouterr().out
    assert f"embeddings: {env / 'embeddings.sqlite'}" in out
    assert "this run: 5/5 chunk(s) (5 ok, 0 failed, 0 rejected)" in out
    assert "stored: 5 chunk(s) in 5 row(s) (5 ok, 0 failed, 0 rejected)" in out
    assert f"log: {next((env / 'logs').glob('embed-*.log'))}" in out
    assert stored_statuses(env) == {"ok": 5, "failed": 0, "rejected": 0}


def test_run_logs_to_a_file_and_the_console_without_the_key(env, embed_env, capsys):
    embed_run()
    captured = capsys.readouterr()
    text = log_text(env)
    assert "starting: 5 chunks (5 new) with openai/text-embedding-3-small" in text
    assert "completed: 5/5 chunks (5 ok)" in text
    assert "completed: 5/5 chunks" in captured.err
    for output in (text, captured.out, captured.err):
        assert "sk-test" not in output


def test_a_rerun_with_nothing_to_do_makes_no_requests(env, embed_env, capsys):
    embed_run()
    embed_env.calls.clear()
    assert embed_run() == 0
    assert embed_env.calls == []
    assert "this run: 0/0 chunk(s)" in capsys.readouterr().out


def test_limit_embeds_only_that_many_and_a_rerun_finishes_the_rest(
    env, embed_env, capsys
):
    assert embed_run("--limit", "2") == 0
    assert "this run: 2/2 chunk(s)" in capsys.readouterr().out
    assert embed_run() == 0
    assert "this run: 3/3 chunk(s)" in capsys.readouterr().out
    sent = [t for call in embed_env.calls for t in call]
    assert len(sent) == len(set(sent)) == 5  # nothing embedded twice


def test_rate_tpm_and_concurrency_flags_reach_the_run(env, embed_env):
    assert (
        main(["embed", "run", "--rate", "6000", "--tpm", "50000", "--concurrency", "3"])
        == 0
    )
    assert "6000 requests/min, 50000 tokens/min, 3 in flight" in log_text(env)


def test_defaults_are_the_conservative_ones(env, embed_env):
    main(["embed", "run", "--limit", "1"])
    assert "500 requests/min, 1000000 tokens/min, 4 in flight" in log_text(env)


@pytest.mark.parametrize(
    "flags",
    [
        ["--rate", "0"],
        ["--rate", "fast"],
        ["--tpm", "-1"],
        ["--concurrency", "0"],
        ["--concurrency", "2.5"],
        ["--limit", "0"],
    ],
    ids=lambda f: " ".join(f),
)
def test_nonsense_flag_values_are_usage_errors(env, flags):
    with pytest.raises(SystemExit) as excinfo:
        main(["embed", "run", *flags])
    assert excinfo.value.code == 2


def add_sectioned_pages(data_dir: Path, sections: dict[str, int]) -> None:
    with ChunkStore(data_dir / "parsed.sqlite") as store:
        for section, pages in sections.items():
            for p in range(pages):
                url = f"https://docs.databricks.com/aws/en/{section}/p{p:02d}"
                store.replace_page_chunks(
                    url,
                    [
                        make_chunk(
                            0, chunk_id=f"{url}#0", url=url, text=f"{section} {p}"
                        )
                    ],
                )


def test_sample_embeds_a_spread_and_a_rerun_samples_what_is_left(
    env, embed_env, monkeypatch, caplog, capsys
):
    add_sectioned_pages(
        env, {"delta": 20, "sql": 20, "ml": 20}
    )  # plus 5 from embed_env
    assert embed_run("--sample", "10") == 0
    out = capsys.readouterr().out
    first = stored_statuses(env)["ok"]
    assert 10 <= first <= 12  # N, plus up to the two extremes
    assert f"this run: {first}/{first} chunk(s)" in out
    text = log_text(env)
    assert "sample: " in text and "candidate chunks, from 4 sections" in text

    assert embed_run("--sample", "10") == 0
    sent = [t for call in embed_env.calls for t in call]
    assert len(sent) == len(set(sent))  # the second draw skipped the first's


def test_sample_too_small_is_refused_with_the_number_needed(env, embed_env, capsys):
    add_sectioned_pages(env, {"delta": 5, "sql": 5, "ml": 5})
    assert embed_run("--sample", "3") == 1
    err = capsys.readouterr().err
    assert "--sample 3 is too small" in err and "chunks" in err
    assert embed_env.calls == []


def test_sample_and_limit_cannot_be_combined(env):
    with pytest.raises(SystemExit) as excinfo:
        main(["embed", "run", "--sample", "5", "--limit", "5"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("value", ["0", "-1", "ten"])
def test_sample_must_be_a_positive_whole_number(env, value):
    with pytest.raises(SystemExit) as excinfo:
        main(["embed", "run", "--sample", value])
    assert excinfo.value.code == 2


def test_there_is_no_retry_failed_flag(env):
    with pytest.raises(SystemExit) as excinfo:
        main(["embed", "run", "--retry-failed"])
    assert excinfo.value.code == 2


def test_run_without_a_chunk_store_says_so(env, monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert main(["embed", "run"]) == 1
    assert "parse run" in capsys.readouterr().err
    assert not (env / "embeddings.sqlite").exists()


def test_run_without_an_api_key_fails_before_any_request(env, monkeypatch, capsys):
    add_pages(env, 2)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake = FakeLiteLLM()
    monkeypatch.setattr(litellm, "aembedding", fake)

    assert main(["embed", "run"]) == 1

    err = capsys.readouterr().err
    assert "OPENAI_API_KEY not set" in err
    assert ".env" in err
    assert fake.calls == []
    assert not (env / "embeddings.sqlite").exists()
    assert not (env / "logs").exists()


def test_run_reads_the_api_key_from_dotenv_in_the_working_directory(
    env, monkeypatch, capsys
):
    add_pages(env, 1)
    # setenv then delenv, so monkeypatch removes whatever load_dotenv sets
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.delenv("OPENAI_API_KEY")
    (env.parent / ".env").write_text('OPENAI_API_KEY="sk-from-dotenv"\n')
    monkeypatch.setattr(litellm, "aembedding", FakeLiteLLM())

    assert main(["embed", "run"]) == 0
    assert os.environ["OPENAI_API_KEY"] == "sk-from-dotenv"


def test_a_real_environment_variable_wins_over_dotenv(env, monkeypatch):
    add_pages(env, 1)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    (env.parent / ".env").write_text('OPENAI_API_KEY="sk-from-dotenv"\n')
    monkeypatch.setattr(litellm, "aembedding", FakeLiteLLM())
    main(["embed", "run"])
    assert os.environ["OPENAI_API_KEY"] == "sk-from-shell"


def test_the_circuit_breaker_ending_a_run_exits_nonzero(
    env, embed_env, fast_failures, capsys
):
    add_pages(env, 40)
    embed_env.respond = lambda n, texts: llm_errors.RateLimitError(
        "slow down", llm_provider="openai", model=MODEL, response=http_response(429)
    )
    assert embed_run("--concurrency", "1") == 1

    out = capsys.readouterr().out
    assert "circuit breaker ended the run" in out
    assert "not embedded yet" in out
    assert "lower --rate/--tpm" in out
    text = log_text(env)
    assert "circuit breaker tripped (10 consecutive failures, trip 1 of 3)" in text
    assert "breaker_tripped:" in text


def test_repeated_non_retryable_errors_exit_nonzero_with_a_hint(
    env, embed_env, fast_failures, capsys
):
    add_pages(env, 40)
    embed_env.respond = lambda n, texts: llm_errors.AuthenticationError(
        "Incorrect API key provided",
        llm_provider="openai",
        model=MODEL,
        response=http_response(401),
    )
    assert embed_run("--concurrency", "1") == 1

    out = capsys.readouterr().out
    assert "check that the API key" in out
    assert "provider may be down" in out
    assert len(embed_env.calls) == 5  # fatal_error_threshold, then stop
    assert "Incorrect API key provided" in log_text(env)


def test_failed_chunks_are_reported_and_the_run_still_exits_zero(
    env, embed_env, fast_failures, capsys
):
    embed_env.respond = lambda n, texts: llm_errors.Timeout(
        "timed out", model=MODEL, llm_provider="openai"
    )
    assert embed_run() == 0
    out = capsys.readouterr().out
    assert "5 chunk(s) failed" in out
    assert "retries them automatically" in out
    assert stored_statuses(env)["failed"] == 5

    # ...and the next plain run picks them up with no flag
    embed_env.respond = lambda n, texts: None
    assert embed_run() == 0
    assert "this run: 5/5 chunk(s) (5 ok" in capsys.readouterr().out


def test_rejected_chunks_are_reported_and_not_retried(env, embed_env, capsys):
    embed_env.respond = lambda n, texts: (
        llm_errors.ContentPolicyViolationError(
            "flagged", model=MODEL, llm_provider="openai", response=http_response(400)
        )
        if any(t.endswith("text 3") for t in texts)
        else None
    )
    assert embed_run() == 0
    out = capsys.readouterr().out
    assert "1 chunk(s) rejected by the provider's content policy" in out
    assert stored_statuses(env) == {"ok": 4, "failed": 0, "rejected": 1}

    embed_env.calls.clear()
    embed_run()
    assert embed_env.calls == []


def test_a_model_name_conflict_is_reported_with_the_fix(env, embed_env, capsys):
    embed_env.model = "text-embedding-3-large"
    assert embed_run() == 0
    out = capsys.readouterr().out
    assert (
        "model name conflict: configured 'text-embedding-3-small' but the provider "
        "reported openai/text-embedding-3-large"
    ) in out
    assert "databricks_docs_vectordb/embedding/config.py" in out


def test_ctrl_c_stops_cleanly_with_exit_130_and_a_rerun_resumes(
    env, embed_env, monkeypatch, capsys
):
    add_pages(env, 60)
    patch_settings(monkeypatch, batch_size=1)
    embed_env.delay = 0.05
    timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        code = main(["embed", "run", "--rate", "100000", "--concurrency", "1"])
    finally:
        timer.cancel()

    out = capsys.readouterr().out
    assert code == 130
    assert "interrupted:" in out and "run the same command again to resume" in out
    done = stored_statuses(env)["ok"]
    assert 0 < done < 60
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

    embed_env.delay = 0
    assert embed_run() == 0
    assert stored_statuses(env)["ok"] == 60
    sent = [t for call in embed_env.calls for t in call]
    assert len(sent) == len(set(sent))  # nothing embedded twice


def test_logging_setup_is_undone_after_a_run(env, embed_env):
    package = logging.getLogger("databricks_docs_vectordb")
    original = package.level
    package.setLevel(logging.ERROR)
    handlers = list(package.handlers)
    try:
        embed_run()
        assert list(package.handlers) == handlers
        assert package.level == logging.ERROR
    finally:
        package.setLevel(original)


# =========================================================================
# embed status
# =========================================================================


def test_status_before_any_run_points_at_embed_run(env, capsys):
    assert main(["embed", "status"]) == 0
    assert "embed run" in capsys.readouterr().out
    assert not (env / "embeddings.sqlite").exists()


def test_status_needs_no_api_key_or_network(env, embed_env, monkeypatch, capsys):
    embed_run("--limit", "3")
    capsys.readouterr()
    monkeypatch.delenv("OPENAI_API_KEY")
    embed_env.calls.clear()

    assert main(["embed", "status"]) == 0
    out = capsys.readouterr().out
    assert embed_env.calls == []
    assert "3 chunk(s) in 3 row(s)" in out
    assert "rows: 3 ok, 0 failed, 0 rejected" in out
    assert (
        "openai / text-embedding-3-small / - / 3: 3 chunk(s), 3 row(s) "
        "(3 ok, 0 failed, 0 rejected)"
    ) in out
    assert "2 chunk(s) still to embed (2 new): " in out


def test_status_lists_failed_and_rejected_chunks_with_reasons(
    env, embed_env, fast_failures, capsys
):
    # one chunk per batch (fast_failures): chunk 0 is refused, the rest time out
    embed_env.respond = lambda n, texts: (
        llm_errors.ContentPolicyViolationError(
            "flagged for policy",
            model=MODEL,
            llm_provider="openai",
            response=http_response(400),
        )
        if texts[0].endswith("text 0")
        else llm_errors.Timeout("timed out", model=MODEL, llm_provider="openai")
    )
    embed_run()
    capsys.readouterr()

    assert main(["embed", "status"]) == 0
    out = capsys.readouterr().out
    assert "rejected: 1 chunk(s)" in out
    assert f"{page_url(0)}#0: " in out and "flagged for policy" in out
    assert "failed: 4 chunk(s)" in out
    assert "timed out" in out
    assert "4 chunk(s) still to embed (4 failed)" in out


def test_status_truncates_long_problem_lists(env, embed_env, fast_failures, capsys):
    add_pages(env, 15)
    embed_env.respond = lambda n, texts: llm_errors.Timeout(
        "timed out", model=MODEL, llm_provider="openai"
    )
    embed_run()
    capsys.readouterr()
    main(["embed", "status"])
    out = capsys.readouterr().out
    assert "failed: 15 chunk(s), first 10" in out
    listed = [line for line in out.splitlines() if line.startswith("  https://")]
    assert len(listed) == 10


def test_status_when_everything_is_current(env, embed_env, capsys):
    embed_run()
    capsys.readouterr()
    main(["embed", "status"])
    assert "is embedded and current" in capsys.readouterr().out


def test_status_notices_a_model_change_as_work_to_do(
    env, embed_env, monkeypatch, capsys
):
    embed_run()
    capsys.readouterr()
    patch_settings(monkeypatch, model="text-embedding-3-large")
    main(["embed", "status"])
    assert "5 chunk(s) still to embed (5 model)" in capsys.readouterr().out


# =========================================================================
# embed verify
# =========================================================================


def test_verify_without_a_chunk_store_is_an_error(env, capsys):
    assert main(["embed", "verify"]) == 1
    assert "parse run" in capsys.readouterr().err


def test_verify_before_any_run_reports_coverage_and_creates_nothing(env, capsys):
    add_pages(env, 3)
    assert main(["embed", "verify"]) == 0
    out = capsys.readouterr().out
    assert "3 not-embedded" in out
    assert "not violations unless --complete" in out
    assert out.rstrip().endswith("OK")
    assert not (env / "embeddings.sqlite").exists()


def test_verify_complete_before_any_run_fails(env, capsys):
    add_pages(env, 3)
    assert main(["embed", "verify", "--complete"]) == 1
    assert "FAILED: 3 violations" in capsys.readouterr().out


def test_verify_passes_after_a_full_run_without_a_key_or_network(
    env, embed_env, monkeypatch, capsys
):
    embed_run()
    capsys.readouterr()
    monkeypatch.delenv("OPENAI_API_KEY")
    embed_env.calls.clear()

    assert main(["embed", "verify", "--complete"]) == 0
    out = capsys.readouterr().out
    assert "coverage (chunks): 5 ok" in out
    assert "openai/text-embedding-3-small/- (3)" in out
    assert out.rstrip().endswith("OK")
    assert embed_env.calls == []


def test_verify_flags_a_corrupt_blob_and_exits_1(env, embed_env, capsys):
    embed_run()
    capsys.readouterr()
    with EmbeddingsStore(env / "embeddings.sqlite") as store, store._conn:
        store._conn.execute(
            "UPDATE embeddings SET embedding = ? WHERE chunk_id = ?",
            (b"short", f"{page_url(2)}#0"),
        )
    assert main(["embed", "verify"]) == 1
    out = capsys.readouterr().out
    assert f"[bad-blob-length] {page_url(2)}#0" in out


def test_verify_flags_orphaned_rows_after_a_reparse(env, embed_env, capsys):
    embed_run()
    capsys.readouterr()
    with ChunkStore(env / "parsed.sqlite") as store:
        store.replace_page_chunks(page_url(4), [])
    assert main(["embed", "verify"]) == 1
    assert f"[orphaned-rows] {page_url(4)}#0" in capsys.readouterr().out


def dump(path: Path) -> list[str]:
    # Logical contents, not bytes: opening any store bumps SQLite's
    # file-change counter in the header even when no row changes.
    with sqlite3.connect(path) as conn:
        return list(conn.iterdump())


def test_verify_changes_nothing(env, embed_env):
    embed_run()
    before = {name: dump(env / name) for name in ("embeddings.sqlite", "parsed.sqlite")}
    main(["embed", "verify", "--complete"])
    after = {name: dump(env / name) for name in ("embeddings.sqlite", "parsed.sqlite")}
    assert after == before
