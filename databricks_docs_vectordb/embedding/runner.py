"""Embed every chunk that needs it, mirroring `scraping/runner.py`'s shape
(chunking_and_embedding.md's "Failure handling" and "Batching,
concurrency, and rate limits").

Selection is automatic, with no flag (Resolved question 6): a chunk is a
candidate when it has no `embeddings` row, its stored `chunk_text_hash`
doesn't match its current text, its stored `(provider, model,
model_version)` doesn't match the configured model, or any of its rows is
`failed`. `ok` and `rejected` chunks that are still current are left alone.

A batch is a list of whole chunks, bounded by `batch_size` inputs and
`batch_token_budget` tokens, so every chunk's rows are written by one
`upsert`. A chunk that alone exceeds either bound (an oversized chunk split
into many slices) is its own batch and goes out as several requests, each
within the bounds.
"""

import asyncio
import hashlib
import logging
import random
import signal
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import litellm

from databricks_docs_vectordb.embedding.assemble import EmbeddingInput, assemble_inputs
from databricks_docs_vectordb.embedding.config import Settings
from databricks_docs_vectordb.embedding.provider import (
    EmbeddingBatch,
    EmbeddingError,
    RejectedError,
    RetryableError,
    configured_model_info,
    embed,
)
from databricks_docs_vectordb.embedding.store import (
    EmbeddingRow,
    EmbeddingsStore,
    ModelInfo,
    StoredChunk,
    pack_embedding,
)
from databricks_docs_vectordb.parsing.store import Chunk, ChunkStore
from databricks_docs_vectordb.scraping.runner import TokenBucket

logger = logging.getLogger(__name__)

RunStatus = Literal["completed", "interrupted", "breaker_tripped", "provider_errors"]
Outcome = Literal["ok", "failed", "rejected"]
Reason = Literal["new", "changed", "model", "failed"]
EmbedFn = Callable[..., Awaitable[EmbeddingBatch]]


class RunStopped(Exception):
    """Raised inside a batch when the run is stopping; its chunks are left
    as they were."""


class _RequestFailed(Exception):
    """A request ended without vectors and retrying it now won't help:
    retries exhausted, a Retry-After too long to wait out, or a
    non-retryable provider error."""


def text_hash(text: str) -> str:
    """`chunk_text_hash`: sha256 of `chunk.text` (Idempotency and
    staleness)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_reason(
    chunk_hash: str, stored: StoredChunk | None, current: ModelInfo
) -> Reason | None:
    """Why a chunk needs embedding, or None if what's stored is current."""
    if stored is None:
        return "new"
    if stored.chunk_text_hash != chunk_hash:
        return "changed"
    if stored.model_info != current:
        return "model"
    if stored.any_failed:
        return "failed"
    return None


def pack[T](
    items: Iterable[T],
    *,
    size: Callable[[T], int],
    tokens: Callable[[T], int],
    max_size: int,
    max_tokens: int,
) -> Iterator[list[T]]:
    """Group `items`, in order, into lists whose summed `size` and `tokens`
    stay within `max_size` and `max_tokens`. An item that alone exceeds a
    bound is yielded on its own rather than dropped."""
    group: list[T] = []
    group_size = group_tokens = 0
    for item in items:
        item_size, item_tokens = size(item), tokens(item)
        if group and (
            group_size + item_size > max_size or group_tokens + item_tokens > max_tokens
        ):
            yield group
            group, group_size, group_tokens = [], 0, 0
        group.append(item)
        group_size += item_size
        group_tokens += item_tokens
    if group:
        yield group


@dataclass
class _Job:
    """One chunk to embed, with its assembled input(s)."""

    chunk: Chunk
    chunk_hash: str
    inputs: list[EmbeddingInput]
    attempts: int = 0  # request attempts that carried this chunk

    @property
    def tokens(self) -> int:
        return sum(i.token_count for i in self.inputs)


@dataclass(frozen=True)
class RunSummary:
    status: RunStatus
    selected: int  # chunks the run set out to embed
    reasons: Counter[str]  # why they were selected: new / changed / model / failed
    outcomes: Counter[str]  # chunks finished: ok / failed / rejected
    requests: int  # provider calls, retries included
    tokens_sent: int  # input tokens across all of those calls
    breaker_trips: int
    elapsed: float
    # (provider, model) pairs the provider reported instead of the
    # configured one -- see Runner._check_substitution.
    model_mismatches: frozenset[tuple[str, str]] = frozenset()

    @property
    def processed(self) -> int:
        return sum(self.outcomes.values())

    @property
    def unprocessed(self) -> int:
        """Selected chunks left untouched (a stop, or the breaker); the next
        run selects them again."""
        return self.selected - self.processed


class Runner:
    """Embed a run's candidate chunks with a pool of workers.

    - Two shared token buckets limit every request start across all
      workers, retries included: one for requests per minute, one for input
      tokens per minute (a request takes as many tokens as it sends).
    - Retryable errors (rate limit, timeout, 5xx) back off with jitter,
      honoring Retry-After, up to `max_attempts`; then the batch's chunks
      are stored `failed` and the next run picks them up again.
    - A content-policy rejection is terminal: stored `rejected`, never
      retried. A rejected multi-chunk batch is re-sent one chunk at a time
      first, so only the chunk(s) actually refused are marked.
    - A circuit breaker counts consecutive retryable failures across all
      workers; any other answer from the provider resets it. When it trips,
      every request pauses for `breaker_cooldown`; the
      `breaker_max_trips`-th trip ends the run instead.
    - `fatal_error_threshold` consecutive non-retryable errors (bad key,
      unknown model, ...) end the run too, with a hint to check the API key
      and the provider's status, rather than failing every remaining batch
      the same way. A success or a rejection resets that count.
    - SIGINT/SIGTERM once: stop starting requests, let those on the wire
      finish and be recorded, cut backoff sleeps short, leave unfinished
      chunks as they were. Twice: cancel everything now. Either way the
      store is consistent (each chunk is one atomic `upsert`), so re-running
      resumes.

    A Runner does one run.
    """

    def __init__(
        self,
        settings: Settings,
        chunks: ChunkStore,
        embeddings: EmbeddingsStore,
        *,
        embed: EmbedFn = embed,
        clock: Callable[[], float] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._chunks = chunks
        self._embeddings = embeddings
        self._embed = embed
        self._clock = clock or time.monotonic
        self._rng = rng or random.random
        self._model = configured_model_info(settings.model)

        self._request_bucket = TokenBucket(
            settings.rate_rpm / 60, clock=self._clock, sleep=self._bucket_sleep
        )
        # Big enough for the largest single request `pack` can produce: a
        # full budget, or one input on its own when that's bigger.
        self._token_bucket = TokenBucket(
            settings.rate_tpm / 60,
            capacity=max(settings.batch_token_budget, settings.model_token_limit),
            clock=self._clock,
            sleep=self._bucket_sleep,
        )

        self._gate = asyncio.Event()  # set while requests may go out
        self._gate.set()
        self._stop = asyncio.Event()
        self._stop_reason: RunStatus | None = None
        self._signals_received = 0
        self._hard_stop = False
        self._workers: list[asyncio.Task[None]] = []
        self._cooldown_task: asyncio.Task[None] | None = None

        self._reasons: Counter[str] = Counter()
        self._outcomes: Counter[str] = Counter()
        self._requests = 0
        self._tokens_sent = 0
        self._consecutive_failures = 0
        self._consecutive_errors = 0
        self._trips = 0
        self._selected = 0
        self._started = 0.0
        self._substitutions_seen: set[tuple[str, str]] = set()

    # -- public ----------------------------------------------------------

    def request_stop(self, reason: RunStatus = "interrupted") -> None:
        """Ask the run to wind down. The first reason given is the one kept."""
        if self._stop_reason is None:
            self._stop_reason = reason
            self._stop.set()

    def select(self, limit: int | None = None) -> list[tuple[Chunk, str]]:
        """`(chunk, chunk_text_hash)` for every chunk that needs embedding,
        in `(url, chunk_index)` order, first `limit` only if given. Also
        tallies why each was picked, for the summary."""
        stored = self._embeddings.stored_chunks()
        selected: list[tuple[Chunk, str]] = []
        for chunk in self._chunks.iter_chunks():
            if limit is not None and len(selected) >= limit:
                break
            chunk_hash = text_hash(chunk.text)
            reason = candidate_reason(
                chunk_hash, stored.get(chunk.chunk_id), self._model
            )
            if reason is not None:
                self._reasons[reason] += 1
                selected.append((chunk, chunk_hash))
        return selected

    async def run(
        self, *, limit: int | None = None, handle_signals: bool = True
    ) -> RunSummary:
        """Embed every candidate chunk (or the first `limit`)."""
        candidates = self.select(limit)
        self._selected = len(candidates)
        self._started = self._clock()
        logger.info(
            "starting: %d chunks (%s) with %s/%s, %.0f requests/min, "
            "%.0f tokens/min, %d in flight",
            self._selected,
            _format_counts(self._reasons, ("new", "changed", "model", "failed")),
            self._model.provider,
            self._model.model,
            self._settings.rate_rpm,
            self._settings.rate_tpm,
            self._settings.concurrency,
        )

        # Assembled lazily as workers pull batches, so the first requests go
        # out without first tokenizing the whole corpus. `next()` on a shared
        # generator is safe here: workers only interleave at `await`s.
        batches = pack(
            self._jobs(candidates),
            size=lambda job: len(job.inputs),
            tokens=lambda job: job.tokens,
            max_size=self._settings.batch_size,
            max_tokens=self._settings.batch_token_budget,
        )
        self._workers = [
            asyncio.create_task(self._worker(batches))
            for _ in range(min(self._settings.concurrency, len(candidates)))
        ]
        installed = self._install_signal_handlers() if handle_signals else []
        try:
            await self._wait_for(self._workers)
        finally:
            self._remove_signal_handlers(installed)
            if self._cooldown_task is not None:
                self._cooldown_task.cancel()
                await asyncio.gather(self._cooldown_task, return_exceptions=True)

        summary = RunSummary(
            status=self._stop_reason or "completed",
            selected=self._selected,
            reasons=self._reasons,
            outcomes=self._outcomes,
            requests=self._requests,
            tokens_sent=self._tokens_sent,
            breaker_trips=self._trips,
            elapsed=self._clock() - self._started,
            model_mismatches=frozenset(self._substitutions_seen),
        )
        logger.info(
            "%s: %d/%d chunks (%s) in %.0fs, %d requests, %d tokens, "
            "%d breaker trips, %d left for the next run",
            summary.status,
            summary.processed,
            summary.selected,
            _format_counts(summary.outcomes, ("ok", "failed", "rejected")),
            summary.elapsed,
            summary.requests,
            summary.tokens_sent,
            summary.breaker_trips,
            summary.unprocessed,
        )
        return summary

    # -- workers ---------------------------------------------------------

    def _jobs(self, candidates: Sequence[tuple[Chunk, str]]) -> Iterator[_Job]:
        for chunk, chunk_hash in candidates:
            inputs = assemble_inputs(
                chunk,
                model=self._settings.model,
                token_limit=self._settings.model_token_limit,
                overlap_tokens=self._settings.split_overlap_tokens,
            )
            yield _Job(chunk, chunk_hash, inputs)

    async def _worker(self, batches: Iterator[list[_Job]]) -> None:
        while not self._stop.is_set():
            batch = next(batches, None)
            if batch is None:
                return
            try:
                await self._embed_batch(batch)
            except RunStopped:
                return  # unfinished chunks untouched: selected again next run

    async def _embed_batch(self, jobs: list[_Job]) -> None:
        if len(jobs) == 1:
            await self._embed_job(jobs[0])
            return

        inputs = [i for job in jobs for i in job.inputs]
        try:
            result = await self._request(jobs, inputs)
        except RejectedError as e:
            logger.warning(
                "a batch of %d chunks was rejected (%s); re-sending each chunk "
                "on its own to find which",
                len(jobs),
                e,
            )
            for job in jobs:
                await self._embed_job(job)
            return
        except _RequestFailed as e:
            for job in jobs:
                self._finish(job, "failed", error=str(e))
            return

        vectors = iter(result.vectors)
        for job in jobs:
            self._finish(job, "ok", [(result, next(vectors)) for _ in job.inputs])

    async def _embed_job(self, job: _Job) -> None:
        """One chunk alone, in as many requests as the bounds require."""
        results: list[tuple[EmbeddingBatch, list[float]]] = []
        requests = pack(
            job.inputs,
            size=lambda _: 1,
            tokens=lambda i: i.token_count,
            max_size=self._settings.batch_size,
            max_tokens=self._settings.batch_token_budget,
        )
        for inputs in requests:
            try:
                result = await self._request([job], inputs)
            except RejectedError as e:
                self._finish(job, "rejected", error=str(e))
                return
            except _RequestFailed as e:
                self._finish(job, "failed", error=str(e))
                return
            results.extend((result, vector) for vector in result.vectors)
        self._finish(job, "ok", results)

    async def _request(
        self, jobs: list[_Job], inputs: list[EmbeddingInput]
    ) -> EmbeddingBatch:
        """One provider call, retried on retryable errors.

        Raises RejectedError, _RequestFailed, or RunStopped.
        """
        texts = [i.text for i in inputs]
        tokens = sum(i.token_count for i in inputs)
        settings = self._settings
        last: RetryableError | None = None
        for attempt in range(1, settings.max_attempts + 1):
            await self._throttle(tokens)
            self._requests += 1
            self._tokens_sent += tokens
            for job in jobs:
                job.attempts += 1
            try:
                result = await self._embed(texts, model=settings.model)
            except RetryableError as e:
                last = e
                self._retryable_failure()
                if (
                    e.retry_after is not None
                    and e.retry_after > settings.retry_after_max
                ):
                    raise _RequestFailed(
                        f"{e}; provider asked us to wait {e.retry_after:.0f}s, "
                        f"more than the {settings.retry_after_max:.0f}s limit"
                    ) from e
                if attempt < settings.max_attempts:
                    delay = _backoff(attempt, settings, self._rng(), e.retry_after)
                    logger.warning(
                        "%d chunks: %s; retrying in %.1fs (attempt %d/%d)",
                        len(jobs),
                        e,
                        delay,
                        attempt,
                        settings.max_attempts,
                    )
                    await self._sleep(delay)
                continue
            except RejectedError:
                # The provider is answering, and about the content, not us.
                self._consecutive_failures = self._consecutive_errors = 0
                raise
            except EmbeddingError as e:
                self._consecutive_failures = 0
                logger.warning("%d chunks failed: %s", len(jobs), e)
                self._fatal_error(e)
                raise _RequestFailed(str(e)) from e

            self._consecutive_failures = self._consecutive_errors = 0
            self._check_substitution(result)
            return result

        raise _RequestFailed(f"gave up after {settings.max_attempts} attempts: {last}")

    async def _throttle(self, tokens: int) -> None:
        """Awaited before every request."""
        await self._gate.wait()  # closed while the breaker cools down
        if self._stop.is_set():
            raise RunStopped
        await self._request_bucket.acquire()
        await self._token_bucket.acquire(tokens)
        if self._stop.is_set():
            raise RunStopped

    async def _sleep(self, seconds: float) -> None:
        """Sleep that a stop request cuts short."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _bucket_sleep(self, seconds: float) -> None:
        """The rate limiters' wait: at a slow rate it can be many seconds, so
        a stop request must end it (by raising) rather than be waited out."""
        await self._sleep(seconds)
        if self._stop.is_set():
            raise RunStopped

    # -- results ---------------------------------------------------------

    def _finish(
        self,
        job: _Job,
        status: Outcome,
        results: Sequence[tuple[EmbeddingBatch, list[float]]] = (),
        *,
        error: str | None = None,
    ) -> None:
        """Replace everything stored for the job's chunk with one row per
        input, then count the outcome."""
        chunk = job.chunk
        rows = []
        for sub_index, embedding_input in enumerate(job.inputs):
            if status == "ok":
                batch, vector = results[sub_index]
                provider, model, model_version = (
                    batch.provider,
                    batch.model,
                    batch.model_version,
                )
                dimensions, blob = batch.dimensions, pack_embedding(vector)
            else:
                # Nothing came back: record what was asked for.
                provider, model, model_version = self._model
                dimensions, blob = None, None
            rows.append(
                EmbeddingRow(
                    chunk_id=chunk.chunk_id,
                    sub_index=sub_index,
                    sub_count=len(job.inputs),
                    url=chunk.url,
                    source=chunk.source,
                    cloud=chunk.cloud,
                    chunk_index=chunk.chunk_index,
                    prev_chunk_id=chunk.prev_chunk_id,
                    next_chunk_id=chunk.next_chunk_id,
                    heading_path=chunk.heading_path,
                    section_path=chunk.section_path,
                    doc_title=chunk.doc_title,
                    description=chunk.description,
                    last_updated=chunk.last_updated,
                    page_sha256=chunk.page_sha256,
                    chunk_text_hash=job.chunk_hash,
                    embedded_text=embedding_input.text,
                    token_count=embedding_input.token_count,
                    provider=provider,
                    model=model,
                    model_version=model_version,
                    dimensions=dimensions,
                    embedding=blob,
                    status=status,
                    attempts=job.attempts,
                    error=error,
                )
            )
        self._embeddings.upsert(chunk.chunk_id, rows)
        if status == "rejected":
            logger.warning("%s rejected: %s", chunk.chunk_id, error)
        self._record(status)

    def _check_substitution(self, result: EmbeddingBatch) -> None:
        """Warn (once per pair) when the provider reports a different
        provider/model than configured.

        The rows record what was reported, so until the config matches,
        every run re-selects them as a model mismatch and re-embeds them --
        hence the suggestion to update the config rather than just a note.
        """
        reported = (result.provider, result.model)
        if reported == (self._model.provider, self._model.model):
            return
        if reported not in self._substitutions_seen:
            self._substitutions_seen.add(reported)
            logger.warning(
                "model name conflict: configured %r (%s/%s) but the provider "
                "reported %s/%s. Vectors are stored under the reported name, "
                "so every run will re-embed them until the config matches: "
                "if the new model is expected, set `model = %r` in "
                "databricks_docs_vectordb/embedding/config.py",
                self._settings.model,
                self._model.provider,
                self._model.model,
                *reported,
                _config_name(*reported),
            )

    # -- bookkeeping and the circuit breaker ----------------------------

    def _record(self, outcome: Outcome) -> None:
        self._outcomes[outcome] += 1
        done = sum(self._outcomes.values())
        if done % self._settings.progress_every == 0:
            elapsed = self._clock() - self._started
            logger.info(
                "progress: %d/%d chunks (%s) | %.0fs elapsed | %.1f chunks/s",
                done,
                self._selected,
                _format_counts(self._outcomes, ("ok", "failed", "rejected")),
                elapsed,
                done / elapsed if elapsed else 0.0,
            )

    def _fatal_error(self, error: EmbeddingError) -> None:
        self._consecutive_errors += 1
        if self._consecutive_errors < self._settings.fatal_error_threshold:
            return
        logger.error(
            "%d consecutive non-retryable errors from %s (last: %s): stopping "
            "the run. Check that the API key for %s is set and valid (for "
            "OpenAI, OPENAI_API_KEY in .env) and that the model name is "
            "right, or the provider may be down -- check its status page. "
            "Chunks already marked failed are retried on the next run.",
            self._consecutive_errors,
            self._model.provider,
            error,
            self._model.provider,
        )
        self.request_stop("provider_errors")

    def _retryable_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._settings.breaker_threshold:
            self._trip()

    def _trip(self) -> None:
        self._consecutive_failures = 0
        self._trips += 1
        if self._trips >= self._settings.breaker_max_trips:
            logger.error(
                "circuit breaker tripped %d times: stopping the run", self._trips
            )
            self.request_stop("breaker_tripped")
            return
        logger.warning(
            "circuit breaker tripped (%d consecutive failures, trip %d of %d): "
            "pausing all requests for %.0fs",
            self._settings.breaker_threshold,
            self._trips,
            self._settings.breaker_max_trips,
            self._settings.breaker_cooldown,
        )
        self._gate.clear()
        self._cooldown_task = asyncio.create_task(self._cooldown())

    async def _cooldown(self) -> None:
        try:
            await self._sleep(self._settings.breaker_cooldown)
        finally:
            self._gate.set()

    # -- shutdown --------------------------------------------------------

    async def _wait_for(self, workers: list[asyncio.Task[None]]) -> None:
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            await self._cancel(workers)
            if not self._hard_stop:  # someone else cancelled us: let it propagate
                raise
        except BaseException:
            await self._cancel(workers)
            raise

    @staticmethod
    async def _cancel(workers: list[asyncio.Task[None]]) -> None:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    def _install_signal_handlers(self) -> list[signal.Signals]:
        loop = asyncio.get_running_loop()
        installed = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._on_signal, sig)
            except (NotImplementedError, ValueError, RuntimeError):
                continue  # not the main thread, or not supported here
            installed.append(sig)
        return installed

    @staticmethod
    def _remove_signal_handlers(installed: list[signal.Signals]) -> None:
        loop = asyncio.get_running_loop()
        for sig in installed:
            loop.remove_signal_handler(sig)

    def _on_signal(self, sig: signal.Signals) -> None:
        self._signals_received += 1
        if self._signals_received == 1:
            logger.warning(
                "%s received: finishing requests already in flight, then stopping "
                "(send it again to abort immediately)",
                sig.name,
            )
            self.request_stop("interrupted")
        else:
            logger.warning("%s received again: aborting now", sig.name)
            self.request_stop("interrupted")
            self._hard_stop = True
            for worker in self._workers:
                worker.cancel()


def _backoff(
    attempt: int, settings: Settings, unit: float, retry_after: float | None
) -> float:
    """Exponential backoff with jitter (50-100% of the nominal delay), never
    shorter than what the provider asked for. Same curve as
    `scraping/fetcher.py`'s."""
    nominal = min(settings.backoff_max, settings.backoff_base * 2 ** (attempt - 1))
    delay = nominal * (0.5 + unit / 2)
    return max(delay, retry_after) if retry_after is not None else delay


def _config_name(provider: str, model: str) -> str:
    """The `Settings.model` string that resolves to `(provider, model)`:
    the bare name when it's in LiteLLM's model table under that provider,
    otherwise `provider/model`. (A table lookup rather than
    `configured_model_info`, which raises -- and prints a banner to stdout
    -- for a name LiteLLM doesn't know.)"""
    known = litellm.model_cost.get(model) or {}
    if known.get("litellm_provider") == provider:
        return model
    return f"{provider}/{model}"


def _format_counts(counts: Counter[str], order: Sequence[str]) -> str:
    return ", ".join(f"{counts[k]} {k}" for k in order if counts[k]) or "nothing yet"
