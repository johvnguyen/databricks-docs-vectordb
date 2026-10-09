"""A section-stratified sample of candidate chunks, for Step 9's pilot run
(`embed run --sample N`).

Reuses `scraping/sampling.py`'s allocation -- every (source, top-level
section) gets at least two chunks, the rest in proportion to size, ranked
by a hash so the same candidates always give the same sample -- ranking by
chunk_id rather than URL, so one page's chunks aren't taken together.

On top of that, the longest and shortest candidates are always included:
they're the edge cases a pilot exists to exercise (the oversized chunk that
must split into several stored rows, and the near-empty one that must
still embed), and a random draw of a few hundred from ~52k would almost
never hit either.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from databricks_docs_vectordb.scraping.config import SOURCES, Source
from databricks_docs_vectordb.scraping.sampling import sample_pages

if TYPE_CHECKING:
    from databricks_docs_vectordb.embedding.runner import Candidate
    from databricks_docs_vectordb.parsing.store import Chunk


@dataclass(frozen=True)
class ChunkSample:
    candidates: list["Candidate"]  # in the order they should be embedded
    sections: int
    minimum: int  # smallest N that covers every section with 2 chunks
    # ("longest" | "shortest", chunk) for each extreme that the stratified
    # draw missed and was added on top of it (so len(candidates) <= n + 2)
    extremes_added: list[tuple[str, "Chunk"]]


def sample_chunks(
    candidates: Sequence["Candidate"], n: int, sources: tuple[Source, ...] = SOURCES
) -> ChunkSample:
    """About `n` of `candidates`, spread across every section, plus the
    longest and shortest (by characters) if the draw didn't already take
    them. The extremes go first, so even an early stop covers them.

    Raises `SampleError` if `n` can't give every section two chunks.
    """
    by_id = {c.chunk.chunk_id: c for c in candidates}
    drawn = sample_pages(
        [c.chunk for c in candidates],
        sources,
        n,
        identity=lambda chunk: chunk.chunk_id,
        noun="chunks",
    )
    chosen = [by_id[chunk.chunk_id] for chunk in drawn.pages]

    extremes_added: list[tuple[str, Chunk]] = []
    if candidates:
        taken = {c.chunk.chunk_id for c in chosen}
        by_size = sorted(
            candidates, key=lambda c: (c.chunk.char_count, c.chunk.chunk_id)
        )
        extremes: list[Candidate] = []
        for label, extreme in (("longest", by_size[-1]), ("shortest", by_size[0])):
            if extreme in extremes:
                continue  # a single candidate is both
            if extreme.chunk.chunk_id not in taken:
                extremes_added.append((label, extreme.chunk))
            extremes.append(extreme)
        # Extremes first whether the draw took them or not, so an early stop
        # still covers the edge cases.
        chosen = extremes + [c for c in chosen if c not in extremes]

    return ChunkSample(chosen, drawn.sections, drawn.minimum, extremes_added)
