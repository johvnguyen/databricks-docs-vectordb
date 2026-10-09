"""Four sanity checks on the configured model's collection, for insight while
building Phase 5 (vector_db_integration.md's "Verification", Q21, Q28).
They report rather than pass or fail: only the recall warning has a
threshold.

1. Self-retrieval: a seeded random sample of stored points, each searched
   for its own vector's top 10; how many come back first.
2. Index recall: the same searches, default (HNSW over the quantized
   vectors, at Qdrant's default search settings) against exact (a full scan
   of the original vectors, quantization ignored). Recall@10, with a warning
   below `recall_warn_below`, and both median latencies. Phase 5 picks its
   own search settings.
3. Real text queries: `QUESTIONS`, embedded with the configured model, each
   hit read back from embeddings.sqlite by its key, the way Phase 5 will
   (Q5), and whether the expected page came back. A question carries no
   title or heading prefix; the stored inputs do.
4. Filtered queries: the first `FILTERED` questions again, once with a
   `section` filter and once with a `url` filter, both their expected
   page's; whether every hit matches.

3 and 4 need the embedding model's API key, and are skipped with a note
without one, or if embedding the questions fails.
"""

import asyncio
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from databricks_docs_vectordb.embedding import provider
from databricks_docs_vectordb.embedding.provider import EmbeddingError
from databricks_docs_vectordb.embedding.store import EmbeddingsStore, ModelInfo
from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.runner import (
    RunFailed,
    configured_dimensions,
    unit_vector,
)
from databricks_docs_vectordb.indexing.store import Hit, QdrantStore, collection_name
from databricks_docs_vectordb.parsing.report import section_of
from databricks_docs_vectordb.scraping.config import SOURCES

TOP_K = 10
SAMPLE_SEED = 0  # the same points are sampled every time


class Question(NamedTuple):
    text: str
    expected_url: str  # a page that should be in the top `TOP_K`


# Questions a user might ask, each with a page that answers it. Checked
# 2026-10-09 to be `ok` pages in the manifest.
QUESTIONS = (
    Question("What is Delta Lake?", "https://docs.databricks.com/aws/en/delta/"),
    Question(
        "How do I upsert into a Delta table with MERGE?",
        "https://docs.databricks.com/aws/en/delta/merge",
    ),
    Question(
        "How does Auto Loader incrementally ingest new files from cloud storage?",
        "https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/auto-loader/",
    ),
    Question(
        "How do I schedule and orchestrate jobs?",
        "https://docs.databricks.com/aws/en/jobs/",
    ),
    Question(
        "What is a SQL warehouse?",
        "https://docs.databricks.com/aws/en/compute/sql-warehouse/",
    ),
    Question(
        "What is Unity Catalog?",
        "https://docs.databricks.com/aws/en/data-governance/unity-catalog/",
    ),
    Question(
        "How do I remove old data files from a Delta table with VACUUM?",
        "https://docs.databricks.com/aws/en/tables/operations/vacuum",
    ),
    Question(
        # The corpus documents Databricks' own vector search product.
        "What is Databricks AI Search?",
        "https://docs.databricks.com/aws/en/ai-search/ai-search",
    ),
    Question(
        "Which REST API call returns the details of a job run?",
        "https://docs.databricks.com/api/jobs/v2/run",
    ),
)
FILTERED = 3  # how many of QUESTIONS also run with each filter

# Embeds texts with the configured model; one vector per text, in order.
EmbedQueries = Callable[[Sequence[str]], list[list[float]]]


def embed_with(model: str) -> EmbedQueries:
    """Embed questions the way the corpus was (`provider.embed`), in one
    request."""

    def embed_queries(texts: Sequence[str]) -> list[list[float]]:
        return asyncio.run(provider.embed(texts, model=model)).vectors

    return embed_queries


class QueryHit(NamedTuple):
    score: float
    chunk_id: str
    sub_index: int
    url: str  # these two read back from embeddings.sqlite
    heading_path: str


@dataclass
class QueryResult:
    question: Question
    hits: list[QueryHit]

    @property
    def expected_rank(self) -> int | None:
        """The 1-based rank of the first hit on the expected page."""
        urls = [hit.url for hit in self.hits]
        if self.question.expected_url not in urls:
            return None
        return urls.index(self.question.expected_url) + 1


@dataclass
class FilteredResult:
    question: Question
    field: str  # "section" or "url"
    value: str
    hits: int
    mismatched: list[str]  # the field's value on each hit that doesn't match

    @property
    def ok(self) -> bool:
        return not self.mismatched


@dataclass
class CheckReport:
    collection: str
    points: int
    recall_warn_below: float
    sample_ids: tuple[str, ...] = ()
    self_top1: int = 0
    self_misses: list[str] = field(default_factory=list)
    recall: float | None = None  # mean recall@10; None with no points
    default_ms: float | None = None  # median latency per search
    exact_ms: float | None = None
    queries: list[QueryResult] = field(default_factory=list)
    filtered: list[FilteredResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def recall_warning(self) -> bool:
        return self.recall is not None and self.recall < self.recall_warn_below


def check_collection(
    qdrant: QdrantStore,
    settings: Settings,
    model: ModelInfo,
    embeddings: EmbeddingsStore,
    *,
    embed: EmbedQueries | None,
    clock: Callable[[], float] = time.perf_counter,
) -> CheckReport:
    """Run the four checks on `model`'s collection. With `embed` None (no
    API key), 3 and 4 are skipped with a note. Raises `RunFailed` when the
    collection does not exist, or query embeddings don't fit it."""
    dimensions = configured_dimensions(embeddings, model)
    collection = collection_name(settings.collection_prefix, model, dimensions)
    if not qdrant.collection_exists(collection):
        raise RunFailed(f"{collection} does not exist yet: run `index run` first")

    ids: list[str] = []
    offset = None
    while True:
        points, offset = qdrant.scroll(collection, offset=offset, with_vectors=False)
        ids.extend(point.id for point in points)
        if offset is None:
            break
    report = CheckReport(collection, len(ids), settings.recall_warn_below)
    sample = random.Random(SAMPLE_SEED).sample(
        sorted(ids), min(settings.recall_sample_size, len(ids))
    )
    report.sample_ids = tuple(sample)
    if sample:
        _self_retrieval_and_recall(qdrant, collection, sample, report, clock)
    else:
        report.notes.append("the collection is empty: nothing to sample")

    if embed is None:
        report.notes.append(
            "real text and filtered queries skipped: no API key for the "
            "embedding model (for OpenAI, OPENAI_API_KEY in .env)"
        )
        return report
    try:
        vectors = embed([q.text for q in QUESTIONS])
    except EmbeddingError as e:
        report.notes.append(
            f"real text and filtered queries skipped: embedding the questions failed: {e}"
        )
        return report
    if any(len(v) != dimensions for v in vectors):
        raise RunFailed(
            f"the query embeddings have {sorted({len(v) for v in vectors})} "
            f"dimensions, but {collection} has {dimensions}"
        )
    # Normalized as the stored vectors are; under `Dot`, Qdrant would not.
    if settings.normalize:
        vectors = [unit_vector(v) for v in vectors]
    for question, vector in zip(QUESTIONS, vectors, strict=True):
        hits = qdrant.search(collection, vector, limit=TOP_K)
        report.queries.append(
            QueryResult(question, [_read_back(hit, embeddings) for hit in hits])
        )
    for question, vector in list(zip(QUESTIONS, vectors, strict=True))[:FILTERED]:
        for field_name, value in (
            ("section", _section_of_url(question.expected_url)),
            ("url", question.expected_url),
        ):
            hits = qdrant.search(
                collection, vector, limit=TOP_K, match=(field_name, value)
            )
            mismatched = [
                str(hit.payload.get(field_name))
                for hit in hits
                if hit.payload.get(field_name) != value
            ]
            report.filtered.append(
                FilteredResult(question, field_name, value, len(hits), mismatched)
            )
    return report


def _self_retrieval_and_recall(
    qdrant: QdrantStore,
    collection: str,
    sample: list[str],
    report: CheckReport,
    clock: Callable[[], float],
) -> None:
    points = {point.id: point for point in qdrant.retrieve(collection, sample)}
    default_ms, exact_ms, recalls = [], [], []
    for pid in sample:
        point = points[pid]
        assert point.vector is not None
        started = clock()
        default = qdrant.search(collection, point.vector, limit=TOP_K)
        default_ms.append((clock() - started) * 1000)
        started = clock()
        exact = qdrant.search(collection, point.vector, limit=TOP_K, exact=True)
        exact_ms.append((clock() - started) * 1000)

        if default and default[0].id == pid:
            report.self_top1 += 1
        else:
            top = (
                f"{_key(default[0].payload)}, score {default[0].score:.4f}"
                if default
                else "nothing"
            )
            report.self_misses.append(f"{_key(point.payload)}: top-1 was {top}")
        expected = {hit.id for hit in exact}
        recalls.append(len(expected & {hit.id for hit in default}) / len(expected))
    report.recall = statistics.fmean(recalls)
    report.default_ms = statistics.median(default_ms)
    report.exact_ms = statistics.median(exact_ms)


def _read_back(hit: Hit, embeddings: EmbeddingsStore) -> QueryHit:
    """A hit as Phase 5 will see it: its key, then the row it names."""
    chunk_id = str(hit.payload.get("chunk_id", ""))
    sub_index = int(hit.payload.get("sub_index", 0))
    row = embeddings.lookup(chunk_id, sub_index)
    if row is None:
        return QueryHit(
            hit.score, chunk_id, sub_index, "", "(not in embeddings.sqlite)"
        )
    return QueryHit(hit.score, chunk_id, sub_index, row.url, row.heading_path)


def _key(payload: dict) -> str:
    return f"{payload.get('chunk_id')} (sub_index {payload.get('sub_index')})"


def _section_of_url(url: str) -> str:
    """The `section` payload value of a page, from its URL alone."""
    source = next(s for s in SOURCES if url.startswith(s.url_prefix))
    return f"{source.name}/{section_of(url, source.name, SOURCES)}"


def format_report(report: CheckReport) -> str:
    lines = [f"checked {report.collection} ({report.points} points)", ""]
    sampled = len(report.sample_ids)
    if report.recall is not None:
        lines.append(
            f"1. self-retrieval: {report.self_top1} of {sampled} sampled points "
            "are their own top-1"
        )
        lines.extend(f"   {miss}" for miss in report.self_misses)
        lines.append(
            f"2. index recall@{TOP_K}: {report.recall:.4f} over {sampled} searches, "
            "default search against exact (quantization ignored); median "
            f"{report.default_ms:.1f} ms against {report.exact_ms:.1f} ms"
        )
        if report.recall_warning:
            lines.append(
                f"   warning: recall@{TOP_K} is below {report.recall_warn_below}; "
                "Phase 5 may want rescoring or oversampling"
            )
    if report.queries:
        lines.append(
            f"3. real text queries (top {TOP_K}; the questions carry no title or "
            "heading prefix, the stored inputs do):"
        )
        for result in report.queries:
            rank = result.expected_rank
            found = (
                f"expected page at rank {rank}" if rank else "expected page not found"
            )
            lines.append(f'   "{result.question.text}": {found}')
            lines.append(f"     expected: {result.question.expected_url}")
            for n, hit in enumerate(result.hits, 1):
                lines.append(
                    f"     {n:>2}. {hit.score:.4f}  {hit.url}  {hit.heading_path or '-'}"
                )
        found = sum(1 for r in report.queries if r.expected_rank)
        lines.append(
            f"   the expected page came back for {found} of {len(report.queries)} "
            "questions"
        )
    if report.filtered:
        lines.append("4. filtered queries:")
        for f in report.filtered:
            outcome = (
                "every hit matches"
                if f.ok
                else f"{len(f.mismatched)} hit(s) do not match: {', '.join(f.mismatched)}"
            )
            lines.append(
                f'   "{f.question.text}" with {f.field} = {f.value}: {f.hits} hit(s), '
                f"{outcome}"
            )
    lines.extend(f"note: {note}" for note in report.notes)
    return "\n".join(lines)
