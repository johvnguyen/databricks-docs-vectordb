import asyncio
import logging
import os
import signal
import sqlite3
import time
from collections import Counter
from pathlib import Path

import pytest

from databricks_docs_vectordb.embedding.config import Settings
from databricks_docs_vectordb.embedding.provider import (
    EmbeddingBatch,
    EmbeddingError,
    RejectedError,
    RetryableError,
)
from databricks_docs_vectordb.embedding.runner import (
    Runner,
    candidate_reason,
    pack,
    text_hash,
)
from databricks_docs_vectordb.embedding.store import (
    EmbeddingsStore,
    ModelInfo,
    StoredChunk,
    unpack_embedding,
)
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore
from databricks_docs_vectordb.scraping.runner import TokenBucket

MODEL = "text-embedding-3-small"
CURRENT = ModelInfo("openai", MODEL, None)
AWS = "https://docs.databricks.com/aws/en/"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    fields = {
        "data_dir": tmp_path / "data",
        "model": MODEL,
        "rate_rpm": 1e6,
        "rate_tpm": 1e9,
        "concurrency": 2,
        "max_attempts": 1,
        "backoff_base": 0.001,
        "backoff_max": 0.01,
        "breaker_cooldown": 0.05,
    }
    return Settings(**{**fields, **overrides})


def make_chunk(url: str, index: int, text: str | None = None) -> Chunk:
    text = text if text is not None else f"chunk {index} of {url}"
    return Chunk(
        chunk_id=f"{url}#{index}",
        url=url,
        source="aws-en",
        cloud="aws",
        chunk_index=index,
        prev_chunk_id=None,
        next_chunk_id=None,
        heading_path="Heading",
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


def page_url(n: int) -> str:
    return f"{AWS}p{n:03d}"


@pytest.fixture
def stores(tmp_path):
    settings = make_settings(tmp_path)
    with (
        ChunkStore(settings.parsed_path) as chunks,
        EmbeddingsStore(settings.embeddings_path) as embeddings,
    ):
        yield chunks, embeddings


def add_pages(
    chunks: ChunkStore, pages: int, per_page: int = 1, text=None
) -> list[str]:
    """Add `pages` pages of `per_page` chunks each; returns chunk_ids in
    selection order."""
    ids = []
    for p in range(pages):
        url = page_url(p)
        page = [
            make_chunk(url, i, text(p, i) if text else None) for i in range(per_page)
        ]
        chunks.replace_page_chunks(url, page)
        ids += [c.chunk_id for c in page]
    return ids


def vector_for(text: str) -> list[float]:
    # Deterministic, and exact in float32, so a stored vector can be traced
    # back to the text it was computed from.
    return [float(len(text)), float(text.count(" ")), 1.0]


class FakeProvider:
    """Stands in for `provider.embed`. `respond(call_number, texts)` returns
    None (succeed), an exception to raise, or "hang"."""

    def __init__(
        self,
        respond=None,
        *,
        delay: float = 0.0,
        model: str = MODEL,
        provider: str = "openai",
    ):
        self.respond = respond or (lambda n, texts: None)
        self.delay = delay
        self.model = model
        self.provider = provider
        self.calls: list[list[str]] = []
        self.times: list[float] = []

    async def __call__(self, texts, *, model):
        self.calls.append(list(texts))
        self.times.append(time.monotonic())
        if self.delay:
            await asyncio.sleep(self.delay)
        outcome = self.respond(len(self.calls), texts)
        if outcome == "hang":
            await asyncio.sleep(60)
        if isinstance(outcome, BaseException):
            raise outcome
        return EmbeddingBatch(
            [vector_for(t) for t in texts], self.provider, self.model, None, 3
        )


def runner_with(settings, stores, provider, **kwargs) -> Runner:
    chunks, embeddings = stores
    return Runner(settings, chunks, embeddings, embed=provider, **kwargs)


def rows(embeddings: EmbeddingsStore) -> list[sqlite3.Row]:
    return embeddings._conn.execute(
        "SELECT * FROM embeddings ORDER BY chunk_id, sub_index"
    ).fetchall()


def statuses(embeddings: EmbeddingsStore) -> Counter[str]:
    return Counter(r["status"] for r in rows(embeddings))


def retryable(retry_after=None):
    return RetryableError("429 slow down", retry_after)


# =========================================================================
# pure pieces
# =========================================================================


def test_pack_respects_the_size_bound():
    groups = list(
        pack(range(7), size=lambda _: 1, tokens=lambda _: 1, max_size=3, max_tokens=99)
    )
    assert groups == [[0, 1, 2], [3, 4, 5], [6]]


def test_pack_respects_the_token_bound():
    items = [40, 40, 30, 50, 10]
    groups = list(
        pack(items, size=lambda _: 1, tokens=lambda t: t, max_size=99, max_tokens=80)
    )
    assert groups == [[40, 40], [30, 50], [10]]
    assert all(sum(g) <= 80 for g in groups)


def test_pack_puts_an_item_over_a_bound_on_its_own():
    items = [10, 500, 10, 10]
    groups = list(
        pack(items, size=lambda _: 1, tokens=lambda t: t, max_size=99, max_tokens=100)
    )
    assert groups == [[10], [500], [10, 10]]


def test_pack_of_nothing_is_nothing():
    assert list(pack([], size=len, tokens=len, max_size=1, max_tokens=1)) == []


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, "new"),
        (StoredChunk("other-hash", CURRENT, False), "changed"),
        (
            StoredChunk(
                "h", ModelInfo("openai", "text-embedding-3-large", None), False
            ),
            "model",
        ),
        (StoredChunk("h", ModelInfo("azure", MODEL, None), False), "model"),
        (StoredChunk("h", ModelInfo("openai", MODEL, "2027-01"), False), "model"),
        (StoredChunk("h", CURRENT, True), "failed"),
        (StoredChunk("h", CURRENT, False), None),
    ],
    ids=["new", "changed", "model", "provider", "version", "failed", "current"],
)
def test_candidate_reason(stored, expected):
    assert candidate_reason("h", stored, CURRENT) == expected


class FakeTime:
    def __init__(self):
        self.now = 100.0
        self.slept: list[float] = []

    def clock(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


async def test_bucket_amount_throttles_by_tokens_not_requests():
    t = FakeTime()
    bucket = TokenBucket(100.0, capacity=500, clock=t.clock, sleep=t.sleep)
    await bucket.acquire(500)  # a full bucket goes out at once
    assert t.slept == []
    await bucket.acquire(300)  # then 300 tokens at 100/s is 3s
    assert t.slept == [pytest.approx(3.0)]


@pytest.mark.parametrize("amount", [0, 501])
async def test_bucket_rejects_an_amount_it_could_never_serve(amount):
    bucket = TokenBucket(100.0, capacity=500)
    with pytest.raises(ValueError):
        await bucket.acquire(amount)


# =========================================================================
# the happy path and batching
# =========================================================================


async def test_every_chunk_is_embedded_and_stored_with_its_metadata(tmp_path, stores):
    chunks, embeddings = stores
    settings = make_settings(tmp_path)
    ids = add_pages(chunks, 3, per_page=2)
    provider = FakeProvider()

    summary = await runner_with(settings, stores, provider).run(handle_signals=False)

    assert summary.status == "completed"
    assert summary.selected == 6
    assert summary.reasons == Counter(new=6)
    assert summary.outcomes == Counter(ok=6)
    assert summary.unprocessed == 0
    stored = rows(embeddings)
    assert [r["chunk_id"] for r in stored] == sorted(ids)
    for r in stored:
        chunk = next(
            c for c in chunks.get_chunks(r["url"]) if c.chunk_id == r["chunk_id"]
        )
        assert r["status"] == "ok"
        assert (r["sub_index"], r["sub_count"]) == (0, 1)
        assert r["chunk_text_hash"] == text_hash(chunk.text)
        assert r["embedded_text"] == "Title\nHeading\nSection\n\n" + chunk.text
        assert r["token_count"] > 0
        assert (r["provider"], r["model"], r["model_version"]) == (
            "openai",
            MODEL,
            None,
        )
        assert r["dimensions"] == 3
        # each vector belongs to its own row's text, not a neighbour's
        assert list(unpack_embedding(r["embedding"])) == vector_for(r["embedded_text"])
        assert (r["attempts"], r["error"]) == (1, None)


async def test_batches_respect_the_size_bound(tmp_path, stores):
    chunks, _ = stores
    settings = make_settings(tmp_path, batch_size=4, concurrency=1)
    add_pages(chunks, 10)
    provider = FakeProvider()
    await runner_with(settings, stores, provider).run(handle_signals=False)
    assert [len(c) for c in provider.calls] == [4, 4, 2]


async def test_batches_respect_the_token_budget(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 10, text=lambda p, i: "word " * 40)
    # Budget for about two and a half inputs of ~50 tokens each.
    settings = make_settings(
        tmp_path, batch_token_budget=125, model_token_limit=8191, concurrency=1
    )
    provider = FakeProvider()
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert summary.outcomes == Counter(ok=10)
    assert [len(c) for c in provider.calls] == [2, 2, 2, 2, 2]


async def test_an_oversized_chunk_becomes_several_rows_across_several_requests(
    tmp_path, stores
):
    chunks, embeddings = stores
    add_pages(chunks, 1, text=lambda p, i: " ".join(f"w{n}" for n in range(3000)))
    settings = make_settings(
        tmp_path,
        model_token_limit=500,
        split_overlap_tokens=50,
        batch_token_budget=1000,
        batch_size=100,
        concurrency=1,
    )
    provider = FakeProvider()
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)

    assert summary.outcomes == Counter(ok=1)
    stored = rows(embeddings)
    n = len(stored)
    assert n > 2
    assert [r["sub_index"] for r in stored] == list(range(n))
    assert {r["sub_count"] for r in stored} == {n}
    assert all(r["token_count"] <= 500 for r in stored)
    # at most two 500-token slices fit a 1000-token request
    assert len(provider.calls) == -(-n // 2)
    assert all(len(c) <= 2 for c in provider.calls)
    assert [t for c in provider.calls for t in c] == [
        r["embedded_text"] for r in stored
    ]


async def test_an_oversized_chunk_is_batched_on_its_own(tmp_path, stores):
    chunks, _ = stores
    add_pages(
        chunks,
        3,
        text=lambda p, i: " ".join(f"w{n}" for n in range(2000)) if p == 1 else "short",
    )
    settings = make_settings(
        tmp_path, model_token_limit=500, batch_token_budget=600, concurrency=1
    )
    provider = FakeProvider()
    await runner_with(settings, stores, provider).run(handle_signals=False)
    # first short chunk; the oversized one's slices, one per request; last short one
    assert provider.calls[0][0].endswith("short")
    assert provider.calls[-1][0].endswith("short")
    assert all(len(c) == 1 for c in provider.calls)


async def test_limit_takes_the_first_candidates(tmp_path, stores):
    chunks, embeddings = stores
    ids = add_pages(chunks, 5)
    settings = make_settings(tmp_path)
    summary = await runner_with(settings, stores, FakeProvider()).run(
        limit=2, handle_signals=False
    )
    assert summary.selected == 2
    assert [r["chunk_id"] for r in rows(embeddings)] == ids[:2]


async def test_an_empty_run_completes_at_once(tmp_path, stores):
    settings = make_settings(tmp_path)
    provider = FakeProvider()
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert (summary.status, summary.selected, provider.calls) == ("completed", 0, [])


async def test_the_request_rate_limit_is_enforced(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 4)
    settings = make_settings(tmp_path, rate_rpm=600, batch_size=1, concurrency=4)
    provider = FakeProvider()
    await runner_with(settings, stores, provider).run(handle_signals=False)
    gaps = [b - a for a, b in zip(provider.times, provider.times[1:])]
    assert all(g >= 0.09 for g in gaps)  # 600/min = one per 0.1s


# =========================================================================
# errors: retry, rejection, non-retryable failures
# =========================================================================


async def test_a_retryable_error_is_retried_then_succeeds(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 1)
    settings = make_settings(tmp_path, max_attempts=3)
    provider = FakeProvider(lambda n, texts: retryable() if n < 3 else None)
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert summary.outcomes == Counter(ok=1)
    assert summary.requests == 3
    assert rows(embeddings)[0]["attempts"] == 3


async def test_retry_after_is_honored(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 1)
    settings = make_settings(tmp_path, max_attempts=2)
    provider = FakeProvider(
        lambda n, texts: retryable(retry_after=0.2) if n == 1 else None
    )
    await runner_with(settings, stores, provider).run(handle_signals=False)
    assert provider.times[1] - provider.times[0] >= 0.19


async def test_a_retry_after_beyond_the_limit_fails_without_waiting(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 1)
    settings = make_settings(tmp_path, max_attempts=5, retry_after_max=10)
    provider = FakeProvider(lambda n, texts: retryable(retry_after=3600))
    started = time.monotonic()
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert time.monotonic() - started < 1
    assert summary.outcomes == Counter(failed=1)
    assert "3600s" in rows(embeddings)[0]["error"]


async def test_exhausted_retries_store_failed_rows_with_no_vector(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 2)
    settings = make_settings(tmp_path, max_attempts=3, concurrency=1)
    provider = FakeProvider(lambda n, texts: retryable())
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)

    assert summary.outcomes == Counter(failed=2)
    assert len(provider.calls) == 3  # one batch, three attempts
    for r in rows(embeddings):
        assert r["status"] == "failed"
        assert (r["embedding"], r["dimensions"]) == (None, None)
        assert (r["provider"], r["model"]) == ("openai", MODEL)  # what was asked for
        assert r["attempts"] == 3
        assert "gave up after 3 attempts" in r["error"]


async def test_a_rejection_is_stored_without_retrying(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 1)
    settings = make_settings(tmp_path, max_attempts=5)
    provider = FakeProvider(lambda n, texts: RejectedError("flagged"))
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert summary.outcomes == Counter(rejected=1)
    assert len(provider.calls) == 1
    r = rows(embeddings)[0]
    assert (r["status"], r["error"], r["embedding"]) == ("rejected", "flagged", None)


async def test_a_rejected_batch_is_resent_chunk_by_chunk_to_isolate_the_culprit(
    tmp_path, stores
):
    chunks, embeddings = stores
    add_pages(chunks, 4, text=lambda p, i: "BAD" if p == 2 else f"fine {p}")
    settings = make_settings(tmp_path, concurrency=1)
    provider = FakeProvider(
        lambda n, texts: (
            RejectedError("flagged") if any(t.endswith("BAD") for t in texts) else None
        )
    )
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)

    assert summary.outcomes == Counter(ok=3, rejected=1)
    assert [len(c) for c in provider.calls] == [4, 1, 1, 1, 1]
    by_text = {
        r["embedded_text"].rsplit("\n", 1)[1]: r["status"] for r in rows(embeddings)
    }
    assert by_text == {
        "fine 0": "ok",
        "fine 1": "ok",
        "BAD": "rejected",
        "fine 3": "ok",
    }


async def test_a_non_retryable_error_fails_the_batch_without_retrying(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 3)
    settings = make_settings(tmp_path, max_attempts=5, concurrency=1)
    provider = FakeProvider(lambda n, texts: EmbeddingError("401 bad key"))
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert summary.outcomes == Counter(failed=3)
    assert len(provider.calls) == 1
    assert {r["error"] for r in rows(embeddings)} == {"401 bad key"}


async def test_a_provider_side_substitution_is_recorded_and_warned_about(
    tmp_path, stores, caplog
):
    chunks, embeddings = stores
    add_pages(chunks, 2)
    settings = make_settings(tmp_path)
    provider = FakeProvider(model="text-embedding-3-large")
    with caplog.at_level(logging.WARNING):
        summary = await runner_with(settings, stores, provider).run(
            handle_signals=False
        )
    assert {r["model"] for r in rows(embeddings)} == {"text-embedding-3-large"}
    assert summary.model_mismatches == {("openai", "text-embedding-3-large")}
    [warning] = [m for m in caplog.messages if "model name conflict" in m]
    assert "configured 'text-embedding-3-small'" in warning
    assert "reported openai/text-embedding-3-large" in warning
    assert "set `model = 'text-embedding-3-large'` in" in warning
    assert "embedding/config.py" in warning


async def test_a_substitution_to_another_provider_suggests_a_prefixed_name(
    tmp_path, stores, caplog
):
    chunks, _ = stores
    add_pages(chunks, 1)
    settings = make_settings(tmp_path)
    provider = FakeProvider(provider="azure", model="my-embedding-deployment")
    with caplog.at_level(logging.WARNING):
        await runner_with(settings, stores, provider).run(handle_signals=False)
    assert any(
        "set `model = 'azure/my-embedding-deployment'`" in m for m in caplog.messages
    )


async def test_no_warning_when_the_reported_model_matches(tmp_path, stores, caplog):
    chunks, _ = stores
    add_pages(chunks, 2)
    with caplog.at_level(logging.WARNING):
        summary = await runner_with(
            make_settings(tmp_path), stores, FakeProvider()
        ).run(handle_signals=False)
    assert summary.model_mismatches == frozenset()
    assert not [m for m in caplog.messages if "model name conflict" in m]


# -- consecutive non-retryable errors end the run -------------------------


async def test_consecutive_non_retryable_errors_end_the_run_with_a_hint(
    tmp_path, stores, caplog
):
    chunks, embeddings = stores
    add_pages(chunks, 20)
    settings = make_settings(
        tmp_path, batch_size=1, concurrency=1, max_attempts=5, fatal_error_threshold=3
    )
    provider = FakeProvider(lambda n, texts: EmbeddingError("401 invalid api key"))
    with caplog.at_level(logging.ERROR):
        summary = await runner_with(settings, stores, provider).run(
            handle_signals=False
        )

    assert summary.status == "provider_errors"
    assert len(provider.calls) == 3  # no retries, and nothing after the third
    assert summary.outcomes == Counter(failed=3)
    assert summary.unprocessed == 17
    assert statuses(embeddings) == Counter(failed=3)
    [error] = [m for m in caplog.messages if "consecutive non-retryable" in m]
    assert "401 invalid api key" in error
    assert "OPENAI_API_KEY" in error
    assert "provider may be down" in error


@pytest.mark.parametrize(
    "between", [None, RejectedError("flagged")], ids=["success", "rejection"]
)
async def test_a_success_or_rejection_resets_the_error_count(tmp_path, stores, between):
    chunks, _ = stores
    add_pages(chunks, 12)
    settings = make_settings(
        tmp_path, batch_size=1, concurrency=1, fatal_error_threshold=3
    )
    # two errors, then something that isn't one, over and over: never three
    provider = FakeProvider(
        lambda n, texts: EmbeddingError("400") if n % 3 else between
    )
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert summary.status == "completed"
    assert len(provider.calls) == 12


async def test_a_bug_in_the_embed_call_propagates(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 3)
    settings = make_settings(tmp_path)
    provider = FakeProvider(lambda n, texts: KeyError("bug"))
    with pytest.raises(KeyError):
        await runner_with(settings, stores, provider).run(handle_signals=False)
    assert rows(embeddings) == []


# =========================================================================
# re-runs: automatic candidate selection
# =========================================================================


async def test_a_second_run_reattempts_exactly_the_failed_chunks(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 6)
    settings = make_settings(tmp_path, batch_size=1, concurrency=1)
    failing_urls = {page_url(1), page_url(4)}
    first = FakeProvider(
        lambda n, texts: (
            retryable() if texts[0].split(" of ")[1] in failing_urls else None
        )
    )
    await runner_with(settings, stores, first).run(handle_signals=False)
    assert statuses(embeddings) == Counter(ok=4, failed=2)
    ok_before = {
        r["chunk_id"]: r["embedded_at"] for r in rows(embeddings) if r["status"] == "ok"
    }

    second = FakeProvider()
    summary = await runner_with(settings, stores, second).run(handle_signals=False)

    assert summary.reasons == Counter(failed=2)
    assert summary.outcomes == Counter(ok=2)
    assert sorted(t.split(" of ")[1] for c in second.calls for t in c) == sorted(
        failing_urls
    )
    assert statuses(embeddings) == Counter(ok=6)
    # rows that were already ok were not rewritten
    assert {
        r["chunk_id"]: r["embedded_at"]
        for r in rows(embeddings)
        if r["chunk_id"] in ok_before
    } == ok_before


async def test_a_rerun_with_nothing_changed_sends_nothing(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 3)
    settings = make_settings(tmp_path)
    await runner_with(settings, stores, FakeProvider()).run(handle_signals=False)
    provider = FakeProvider()
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert (summary.selected, provider.calls) == (0, [])


async def test_rejected_chunks_are_not_reselected(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 1)
    settings = make_settings(tmp_path)
    await runner_with(
        settings, stores, FakeProvider(lambda n, t: RejectedError("no"))
    ).run(handle_signals=False)
    provider = FakeProvider()
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert (summary.selected, provider.calls) == (0, [])


async def test_changed_text_is_reselected(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 2)
    settings = make_settings(tmp_path)
    await runner_with(settings, stores, FakeProvider()).run(handle_signals=False)
    chunks.replace_page_chunks(page_url(1), [make_chunk(page_url(1), 0, "new text")])
    summary = await runner_with(settings, stores, FakeProvider()).run(
        handle_signals=False
    )
    assert summary.reasons == Counter(changed=1)


async def test_a_model_change_reselects_everything(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 3)
    await runner_with(make_settings(tmp_path), stores, FakeProvider()).run(
        handle_signals=False
    )
    large = "text-embedding-3-large"
    settings = make_settings(tmp_path, model=large)
    summary = await runner_with(settings, stores, FakeProvider(model=large)).run(
        handle_signals=False
    )
    assert summary.reasons == Counter(model=3)
    assert {r["model"] for r in rows(embeddings)} == {large}


async def test_a_split_that_changes_size_leaves_no_orphaned_rows(tmp_path, stores):
    chunks, embeddings = stores
    long_text = " ".join(f"w{n}" for n in range(3000))
    add_pages(chunks, 1, text=lambda p, i: long_text)
    settings = make_settings(tmp_path, model_token_limit=500, batch_token_budget=1000)
    await runner_with(settings, stores, FakeProvider()).run(handle_signals=False)
    assert len(rows(embeddings)) > 1

    chunks.replace_page_chunks(page_url(0), [make_chunk(page_url(0), 0, "short now")])
    await runner_with(settings, stores, FakeProvider()).run(handle_signals=False)
    assert [(r["sub_index"], r["sub_count"]) for r in rows(embeddings)] == [(0, 1)]


# =========================================================================
# circuit breaker
# =========================================================================


async def test_consecutive_retryable_failures_trip_the_breaker_and_end_the_run(
    tmp_path, stores
):
    chunks, embeddings = stores
    add_pages(chunks, 20)
    settings = make_settings(
        tmp_path,
        batch_size=1,
        concurrency=1,
        max_attempts=5,
        breaker_threshold=10,
        breaker_max_trips=3,
    )
    provider = FakeProvider(lambda n, texts: retryable())
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)

    assert summary.status == "breaker_tripped"
    assert summary.breaker_trips == 3
    assert summary.requests == 30
    # 5 attempts per chunk, trips at failures 10, 20, 30: the third lands on
    # chunk 6's last attempt, so six chunks exhausted their retries
    assert summary.outcomes == Counter(failed=6)
    assert summary.unprocessed == 14
    assert statuses(embeddings) == Counter(failed=6)


async def test_a_trip_pauses_every_request_for_the_cooldown(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 10)
    settings = make_settings(
        tmp_path,
        batch_size=1,
        concurrency=1,
        breaker_threshold=2,
        breaker_cooldown=0.3,
        breaker_max_trips=3,
    )
    provider = FakeProvider(lambda n, texts: retryable())
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)

    assert summary.breaker_trips == 3
    assert len(provider.calls) == 6
    gaps = [b - a for a, b in zip(provider.times, provider.times[1:])]
    # a trip after calls 2 and 4 paused the next one
    assert gaps[1] >= 0.25 and gaps[3] >= 0.25


async def test_the_run_finishes_normally_if_the_provider_recovers(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 6)
    settings = make_settings(
        tmp_path, batch_size=1, concurrency=1, breaker_threshold=2, breaker_max_trips=3
    )
    provider = FakeProvider(lambda n, texts: retryable() if n <= 2 else None)
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert summary.status == "completed"
    assert summary.breaker_trips == 1
    assert statuses(embeddings) == Counter(ok=4, failed=2)


async def test_a_success_resets_the_failure_count(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 10)
    settings = make_settings(tmp_path, batch_size=1, concurrency=1, breaker_threshold=2)
    provider = FakeProvider(lambda n, texts: retryable() if n % 2 else None)
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert summary.breaker_trips == 0


@pytest.mark.parametrize(
    "answer",
    [RejectedError("flagged"), EmbeddingError("400")],
    ids=["rejected", "error"],
)
async def test_a_non_retryable_answer_resets_the_failure_count(
    tmp_path, stores, answer
):
    chunks, _ = stores
    add_pages(chunks, 10)
    settings = make_settings(tmp_path, batch_size=1, concurrency=1, breaker_threshold=2)
    provider = FakeProvider(lambda n, texts: retryable() if n % 2 else answer)
    summary = await runner_with(settings, stores, provider).run(handle_signals=False)
    assert summary.breaker_trips == 0


# =========================================================================
# shutdown
# =========================================================================


async def test_sigint_stops_cleanly_and_a_rerun_resumes_without_reembedding(
    tmp_path, stores
):
    chunks, embeddings = stores
    add_pages(chunks, 60)
    settings = make_settings(tmp_path, batch_size=2, concurrency=4)
    provider = FakeProvider(delay=0.03)

    async def interrupt():
        while len(provider.calls) < 8:
            await asyncio.sleep(0.005)
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = asyncio.create_task(interrupt())
    first = await runner_with(settings, stores, provider).run()
    await interrupter

    assert first.status == "interrupted"
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    done = statuses(embeddings)
    assert set(done) == {"ok"}
    assert 0 < done["ok"] < 60
    assert first.unprocessed == 60 - done["ok"]
    # every stored chunk is whole: one row, carrying a vector for its own text
    for r in rows(embeddings):
        assert list(unpack_embedding(r["embedding"])) == vector_for(r["embedded_text"])

    second_provider = FakeProvider()
    second = await runner_with(settings, stores, second_provider).run(
        handle_signals=False
    )
    assert second.status == "completed"
    assert second.selected == 60 - done["ok"]
    assert statuses(embeddings) == Counter(ok=60)
    # no chunk was ever embedded twice across the two runs
    sent = Counter(t for c in provider.calls + second_provider.calls for t in c)
    # (requests in flight at the interrupt finished and were stored, so
    # the rerun didn't select them again)
    assert len(sent) == 60
    assert max(sent.values()) == 1


async def test_a_second_sigint_aborts_immediately(tmp_path, stores):
    chunks, embeddings = stores
    add_pages(chunks, 10)
    settings = make_settings(tmp_path, batch_size=1)
    provider = FakeProvider(lambda n, texts: "hang")

    async def interrupt_twice():
        while len(provider.calls) < 2:
            await asyncio.sleep(0.005)
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = asyncio.create_task(interrupt_twice())
    started = time.monotonic()
    summary = await runner_with(settings, stores, provider).run()
    await interrupter

    assert time.monotonic() - started < 5  # not the 60s hang
    assert summary.status == "interrupted"
    assert rows(embeddings) == []
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


async def test_stop_cuts_a_backoff_sleep_short_and_leaves_the_chunk_untouched(
    tmp_path, stores
):
    chunks, embeddings = stores
    add_pages(chunks, 3)
    settings = make_settings(
        tmp_path, concurrency=1, max_attempts=3, backoff_base=60, backoff_max=60
    )
    provider = FakeProvider(lambda n, texts: retryable())
    runner = runner_with(settings, stores, provider)
    asyncio.get_running_loop().call_later(0.1, runner.request_stop)
    started = time.monotonic()
    summary = await runner.run(handle_signals=False)

    assert time.monotonic() - started < 3
    assert summary.status == "interrupted"
    assert summary.processed == 0
    assert rows(embeddings) == []  # untouched, not failed


async def test_stop_during_the_cooldown_does_not_wait_it_out(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 5)
    settings = make_settings(
        tmp_path,
        batch_size=1,
        concurrency=1,
        breaker_threshold=1,
        breaker_cooldown=60,
        breaker_max_trips=5,
    )
    provider = FakeProvider(lambda n, texts: retryable())
    runner = runner_with(settings, stores, provider)
    asyncio.get_running_loop().call_later(0.1, runner.request_stop)
    started = time.monotonic()
    summary = await runner.run(handle_signals=False)
    assert time.monotonic() - started < 3
    assert summary.status == "interrupted"


async def test_stop_ends_the_wait_for_a_rate_limit_token(tmp_path, stores):
    chunks, _ = stores
    add_pages(chunks, 3)
    settings = make_settings(tmp_path, rate_rpm=1, batch_size=1, concurrency=1)
    provider = FakeProvider()
    runner = runner_with(settings, stores, provider)
    asyncio.get_running_loop().call_later(0.1, runner.request_stop)
    started = time.monotonic()
    summary = await runner.run(handle_signals=False)
    assert time.monotonic() - started < 3  # not the 60s wait for a second token
    assert summary.outcomes == Counter(ok=1)


async def test_the_first_stop_reason_wins(tmp_path, stores):
    settings = make_settings(tmp_path)
    runner = runner_with(settings, stores, FakeProvider())
    runner.request_stop("breaker_tripped")
    runner.request_stop("interrupted")
    assert (await runner.run(handle_signals=False)).status == "breaker_tripped"
