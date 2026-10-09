"""Check every point of the configured model's collection against the
embeddings store, `parsed.sqlite` and the manifest (vector_db_integration.md's
"Verification", Q20, Q38). Needs Qdrant; changes nothing.

Reports every problem found, not just the first, in the earlier phases' two
tiers:

- **Integrity**, always a violation: a stored vector that is not its row's
  BLOB after normalization; a payload whose `chunk_id`, `sub_index`, `url`
  or `section` is not its row's; a point with no current row (no `ok` row of
  the configured model has its ID, its row is an orphan, or its page is not
  `ok` in the manifest), noting when its ID is not the UUID5 of its payload's
  natural key either; a collection whose settings differ from what
  `create_collection` makes (the vector size, the distance, and the rest);
  and a `red` collection.
- **Coverage**, a violation only with `complete`: a current row with no
  point, which `index run` would load. Mirrors `embed verify --complete`.

A `yellow` or `grey` collection is still optimizing, which is a note.
`indexed_vectors_count` below the point count is normal and not checked.
Vectors are compared as float32, the precision Qdrant stores.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from databricks_docs_vectordb.embedding.store import EmbeddingsStore, ModelInfo
from databricks_docs_vectordb.indexing.config import Settings
from databricks_docs_vectordb.indexing.runner import (
    configured_dimensions,
    point_id,
    point_section,
    to_point,
    vector_hash,
)
from databricks_docs_vectordb.indexing.store import (
    PAYLOAD_FIELDS,
    QdrantStore,
    collection_name,
)
from databricks_docs_vectordb.parsing.store import ChunkStore
from databricks_docs_vectordb.scraping.manifest import Manifest
from databricks_docs_vectordb.scraping.verify import Finding

GAP = "missing-point"  # the one coverage finding

# How many findings of one kind `format_report` lists before summarising.
LIST_LIMIT = 20


@dataclass
class SectionCounts:
    rows: int = 0  # current rows: the points the section should have
    points: int = 0
    missing: int = 0  # current rows with no point
    violations: int = 0


@dataclass
class VerifyReport:
    collection: str
    exists: bool = False
    status: str | None = None  # green, yellow, grey or red; None if missing
    points: int = 0  # points checked
    rows: int = 0  # current rows
    missing: int = 0  # current rows with no point
    notes: list[str] = field(default_factory=list)
    violations: list[Finding] = field(default_factory=list)
    sections: dict[str, SectionCounts] = field(
        default_factory=lambda: defaultdict(SectionCounts)
    )

    @property
    def ok(self) -> bool:
        return not self.violations

    def violation(
        self, kind: str, where: str, detail: str, section: str | None = None
    ) -> None:
        self.violations.append(Finding(kind, where, detail))
        if section is not None:
            self.sections[section].violations += 1


def verify_collection(
    qdrant: QdrantStore,
    settings: Settings,
    model: ModelInfo,
    embeddings: EmbeddingsStore,
    chunks: ChunkStore,
    manifest: Manifest,
    *,
    complete: bool = False,
) -> VerifyReport:
    """Check `model`'s collection, every point of it (Q38).

    The current rows are what `index run` loads: the `ok` rows of `model`
    whose chunk is still in `parsed.sqlite` and whose page is `ok` in the
    manifest. Raises `RunFailed` when `model` has no `ok` rows, or they
    disagree on dimensions, since the collection's name needs them.
    """
    dimensions = configured_dimensions(embeddings, model)
    report = VerifyReport(
        collection_name(settings.collection_prefix, model, dimensions)
    )
    collection = report.collection

    # Every point: its payload fields and a hash of its vector, not the
    # vector itself (about 2.6 GB as Python floats for the whole corpus).
    stored: dict[str, tuple[dict, bytes]] = {}
    report.exists = qdrant.collection_exists(collection)
    if not report.exists:
        report.notes.append(f"{collection} does not exist yet; `index run` creates it")
    else:
        for mismatch in qdrant.settings_mismatches(collection, dimensions=dimensions):
            report.violation("collection", collection, mismatch)
        report.status = qdrant.collection_status(collection)
        if report.status == "red":
            report.violation(
                "index-red", collection, "Qdrant reports it red: an optimization failed"
            )
        elif report.status in ("yellow", "grey"):
            report.notes.append(
                f"the collection is {report.status}: still optimizing, or about "
                "to; searches work, but check again once it is green"
            )
        offset = None
        while True:
            points, offset = qdrant.scroll(collection, offset=offset, with_vectors=True)
            for point in points:
                assert point.vector is not None
                stored[point.id] = (point.payload, vector_hash(point.vector))
            if offset is None:
                break
    report.points = len(stored)

    chunk_ids = chunks.chunk_ids()
    ok_pages = {page.url for page in manifest.pages_with_status("ok")}
    left_out: dict[str, str] = {}  # point ID -> why its row is not current
    for row in embeddings.ok_vectors(model):
        pid = point_id(row.chunk_id, row.sub_index)
        if row.chunk_id not in chunk_ids:
            left_out[pid] = "its row is an orphan: the chunk is gone from parsed.sqlite"
            continue
        if row.url not in ok_pages:
            left_out[pid] = "its page is not ok in the manifest"
            continue
        section = point_section(row)
        counts = report.sections[section]
        counts.rows += 1
        report.rows += 1
        where = f"{row.chunk_id} (sub_index {row.sub_index})"
        found = stored.pop(pid, None)
        if found is None:
            counts.missing += 1
            report.missing += 1
            if complete:
                report.violation(
                    GAP, where, "no point for this row (`index run` loads it)", section
                )
            continue
        counts.points += 1
        payload, digest = found
        expected = to_point(row, normalize=settings.normalize)
        wrong = [f for f in PAYLOAD_FIELDS if payload.get(f) != expected.payload[f]]
        if wrong:
            report.violation(
                "payload-mismatch",
                where,
                f"point {pid}: "
                + "; ".join(
                    f"{f} is {payload.get(f)!r}, expected {expected.payload[f]!r}"
                    for f in wrong
                ),
                section,
            )
        if digest != vector_hash(expected.vector):
            report.violation(
                "vector-mismatch",
                where,
                f"point {pid}: the stored vector is not the row's BLOB"
                + (" after normalization" if settings.normalize else ""),
                section,
            )

    # Whatever is left matched no current row.
    for pid, (payload, _) in sorted(stored.items()):
        chunk_id, sub_index = payload.get("chunk_id"), payload.get("sub_index")
        keyed = isinstance(chunk_id, str) and isinstance(sub_index, int)
        where = f"{chunk_id} (sub_index {sub_index})" if keyed else f"point {pid}"
        detail = f"point {pid}: " + left_out.get(
            pid, "no ok row of the configured model has this ID"
        )
        if not keyed or point_id(chunk_id, sub_index) != pid:
            detail += "; its ID is not the UUID5 of its payload's chunk_id|sub_index"
        section = str(payload.get("section", "(none)"))
        report.sections[section].points += 1
        report.violation("extra-point", where, detail, section)
    return report


def format_report(report: VerifyReport, *, by_section: bool = False) -> str:
    lines = [
        f"checked {report.points} point(s) in {report.collection} against "
        f"{report.rows} current row(s)",
        f"collection: {report.status or 'missing'}",
        f"coverage (current rows): {report.rows - report.missing} with a point, "
        f"{report.missing} without",
    ]
    lines.extend(f"note: {note}" for note in report.notes)
    if report.missing and not any(f.kind == GAP for f in report.violations):
        lines.append(
            f"note: {report.missing} current row(s) have no point (not violations "
            "unless --complete); `index run` loads them"
        )
    if by_section and report.sections:
        width = max(len("source/section"), *map(len, report.sections))
        lines.append("per-section (current rows, points, without a point, violations):")
        header = ("rows", "points", "missing", "violations")
        lines.append(
            f"  {'source/section':<{width}}" + "".join(f"{h:>12}" for h in header)
        )
        for section, c in sorted(report.sections.items()):
            values = (c.rows, c.points, c.missing, c.violations)
            lines.append(f"  {section:<{width}}" + "".join(f"{v:>12}" for v in values))
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
