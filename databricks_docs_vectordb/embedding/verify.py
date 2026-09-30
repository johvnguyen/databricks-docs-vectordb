"""Check the embeddings store against `parsed.sqlite` and its own
invariants. No network; changes nothing.

Reports every problem found, not just the first, mirroring
`parsing/verify.py`'s shape. Two kinds of check:

- **Integrity** -- always a violation, since nothing the pipeline does on
  purpose produces them: rows whose `chunk_id` is no longer in
  `parsed.sqlite` (orphans: the runner never deletes them); a chunk's
  `sub_index` values that aren't exactly `0..sub_count-1`; rows of one
  chunk that disagree on hash, model, status, or dimensions; an `ok` row
  whose BLOB length isn't `dimensions` float32s; a `failed`/`rejected` row
  carrying a vector anyway.
- **Coverage** -- whether every chunk has a current row set: embedded at
  all, under its current text (`chunk_text_hash`), under the configured
  `(provider, model, model_version)`. A gap is only a violation with
  `complete`, since a pilot run (Step 9) embeds a sample on purpose and a
  re-parse legitimately leaves chunks stale until the next `embed run` --
  mirrors `parse verify`'s `--complete`. A `failed` or `rejected` row set
  counts as covered: it's the explicit outcome the design allows.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from databricks_docs_vectordb.embedding.runner import text_hash
from databricks_docs_vectordb.embedding.store import (
    EmbeddingCounts,
    EmbeddingsStore,
    EmbeddingSummary,
    ModelInfo,
    ModelKey,
)
from databricks_docs_vectordb.parsing.store import ChunkStore
from databricks_docs_vectordb.scraping.verify import Finding

FLOAT32_BYTES = 4

# Per-chunk coverage outcomes, in report order. The first three are
# "covered"; the last three are gaps (violations only with `complete`).
COVERAGE = ("ok", "rejected", "failed", "not-embedded", "stale-hash", "model-mismatch")
GAPS = frozenset({"not-embedded", "stale-hash", "model-mismatch"})

# How many findings of one kind `format_report` lists before summarising.
LIST_LIMIT = 20


@dataclass
class VerifyReport:
    model: ModelInfo  # the configured model, what "current" is measured against
    chunks: int = 0  # in parsed.sqlite
    coverage: Counter[str] = field(default_factory=Counter)  # COVERAGE keys
    counts: EmbeddingCounts = field(
        default_factory=lambda: EmbeddingCounts(0, 0, 0, 0, 0)
    )
    by_model: dict[ModelKey, EmbeddingCounts] = field(default_factory=dict)
    violations: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def gaps(self) -> int:
        return sum(self.coverage[k] for k in GAPS)


def verify_embeddings(
    chunks: ChunkStore,
    embeddings: EmbeddingsStore | None,
    model: ModelInfo,
    *,
    complete: bool = False,
) -> VerifyReport:
    """Check `embeddings` against `chunks` and itself under the configured
    `model`. `embeddings` may be None when no store exists yet: every chunk
    is then simply not embedded."""
    report = VerifyReport(model=model)
    rows_by_chunk: dict[str, list[EmbeddingSummary]] = defaultdict(list)
    if embeddings is not None:
        for row in embeddings.all_summaries():
            rows_by_chunk[row.chunk_id].append(row)
        report.counts = embeddings.counts()
        report.by_model = embeddings.counts_by_model()

    # Integrity of every stored row set, orphans included.
    for chunk_id, rows in rows_by_chunk.items():
        _check_row_set(chunk_id, rows, report)

    # Coverage: walk parsed.sqlite, consuming matched row sets; whatever is
    # left over belongs to no current chunk.
    for chunk in chunks.iter_chunks():
        report.chunks += 1
        outcome, detail = _coverage(
            text_hash(chunk.text), rows_by_chunk.pop(chunk.chunk_id, None), model
        )
        report.coverage[outcome] += 1
        if complete and outcome in GAPS:
            report.violations.append(Finding(outcome, chunk.chunk_id, detail))

    for chunk_id, rows in sorted(rows_by_chunk.items()):
        report.violations.append(
            Finding(
                "orphaned-rows",
                chunk_id,
                f"{len(rows)} row(s) for a chunk_id no longer in parsed.sqlite "
                "(its page was re-parsed into fewer chunks, or removed)",
            )
        )
    return report


def _check_row_set(
    chunk_id: str, rows: list[EmbeddingSummary], report: VerifyReport
) -> None:
    def violation(kind: str, detail: str) -> None:
        report.violations.append(Finding(kind, chunk_id, detail))

    n = len(rows)
    sub_indices = sorted(r.sub_index for r in rows)
    sub_counts = sorted({r.sub_count for r in rows})
    if sub_indices != list(range(n)) or sub_counts != [n]:
        violation(
            "bad-split",
            f"{n} row(s) with sub_index {sub_indices} and sub_count "
            f"{sub_counts}; expected sub_index 0..{n - 1}, each with sub_count {n}",
        )

    shared = {
        "chunk_text_hash": {r.chunk_text_hash for r in rows},
        "provider/model/version": {
            (r.provider, r.model, r.model_version) for r in rows
        },
        "status": {r.status for r in rows},
    }
    for name, values in shared.items():
        if len(values) > 1:
            violation(
                "inconsistent-rows",
                f"rows disagree on {name}: {sorted(map(str, values))}",
            )

    ok_dimensions = {r.dimensions for r in rows if r.status == "ok"}
    if len(ok_dimensions) > 1:
        violation(
            "inconsistent-rows",
            f"ok rows disagree on dimensions: {sorted(map(str, ok_dimensions))}",
        )

    for r in rows:
        where = f"sub_index {r.sub_index}"
        if r.status == "ok":
            expected = (r.dimensions or 0) * FLOAT32_BYTES
            if r.dimensions is None or r.blob_len != expected:
                violation(
                    "bad-blob-length",
                    f"{where}: embedding is {r.blob_len} bytes, expected "
                    f"{expected} ({r.dimensions} float32s)",
                )
        elif r.blob_len is not None:
            violation(
                "unexpected-vector",
                f"{where}: a {r.status} row carries a {r.blob_len}-byte embedding",
            )


def _coverage(
    chunk_hash: str, rows: list[EmbeddingSummary] | None, model: ModelInfo
) -> tuple[str, str]:
    """This chunk's COVERAGE outcome, and a detail line for a gap. Same
    precedence as `runner.candidate_reason`: text first, then model, then
    the stored status."""
    if not rows:
        return "not-embedded", "no embeddings rows (`embed run` would embed it)"
    first = rows[0]
    if first.chunk_text_hash != chunk_hash:
        return "stale-hash", (
            "stored chunk_text_hash doesn't match the chunk's current text "
            "(re-parsed since it was embedded)"
        )
    stored = ModelInfo(first.provider, first.model, first.model_version)
    if stored != model:
        return "model-mismatch", (
            f"embedded with {_model_name(stored)}, configured model is "
            f"{_model_name(model)}"
        )
    statuses = {r.status for r in rows}
    if "failed" in statuses:
        return "failed", ""
    if "rejected" in statuses:
        return "rejected", ""
    return "ok", ""


def _model_name(info: ModelInfo | ModelKey) -> str:
    return f"{info.provider}/{info.model}/{info.model_version or '-'}"


def format_report(report: VerifyReport, embeddings_path: str) -> str:
    lines = [
        f"checked {report.chunks} chunk(s) in parsed.sqlite against {embeddings_path}",
        f"configured model: {_model_name(report.model)}",
        "coverage (chunks): "
        + ", ".join(f"{report.coverage[k]} {k}" for k in COVERAGE),
    ]
    c = report.counts
    lines.append(
        f"rows by status: {c.ok} ok, {c.failed} failed, {c.rejected} rejected "
        f"({c.rows} row(s), {c.chunk_ids} chunk_id(s))"
    )
    if report.by_model:
        lines.append("rows by provider/model/version (dimensions):")
        for key, m in sorted(
            report.by_model.items(), key=lambda kv: tuple(map(str, kv[0]))
        ):
            current = "  <- configured" if key[:3] == tuple(report.model) else ""
            lines.append(
                f"  {_model_name(key)} ({key.dimensions or '-'}): {m.chunk_ids} "
                f"chunk(s), {m.rows} row(s): {m.ok} ok, {m.failed} failed, "
                f"{m.rejected} rejected{current}"
            )
    has_gap_violations = any(f.kind in GAPS for f in report.violations)
    if report.gaps and not has_gap_violations:
        lines.append(
            f"note: {report.gaps} chunk(s) not embedded or not current "
            "(not violations unless --complete); `embed run` handles them"
        )
    if report.violations:
        lines.append(f"violations ({len(report.violations)}):")
        by_kind: dict[str, list[Finding]] = defaultdict(list)
        for f in report.violations:
            by_kind[f.kind].append(f)
        for kind, findings in by_kind.items():
            for f in findings[:LIST_LIMIT]:
                lines.append(f"  [{f.kind}] {f.where}: {f.detail}")
            if len(findings) > LIST_LIMIT:
                lines.append(f"  [{kind}] ... and {len(findings) - LIST_LIMIT} more")
    lines.append("")
    lines.append("OK" if report.ok else f"FAILED: {len(report.violations)} violations")
    return "\n".join(lines)
