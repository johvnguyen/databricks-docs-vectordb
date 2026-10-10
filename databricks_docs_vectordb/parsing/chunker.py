"""Split an `html_walker` node tree into ordered, linked chunks.

Per parsing.md's chunking algorithm (Resolved question 6):

- Unit: heading-based. Split at `<h2>` first; if a resulting section still
  exceeds the configured target size, split further at its `<h3>`s.
- Never split a table or code block, regardless of size. One that alone
  exceeds the target size becomes its own chunk, flagged `is_oversized`.
- Target size and overlap are both configurable, not hard-coded.
- Overlap repeats a small amount of trailing content (or, in "heading"
  mode, restates the enclosing heading) at the start of the next chunk.
- Every chunk records `prev_chunk_id`/`next_chunk_id` for retrieval-time
  adjacency.

The heading split (h2, then h3) is the *primary* unit, exactly as
specified. But a section can still be too big after that -- either because
it has no `<h3>`s to split further, or because one single table or code
block inside it is alone bigger than the target. Neither case can be fixed
by picking a different heading boundary, so both fall through to a
size-based packing pass: nodes are greedily packed into chunks up to the
target size, except a single node that alone exceeds the target is never
merged with anything else and is isolated as its own oversized chunk. This
is what actually delivers "a table or code block that alone exceeds the
target size becomes its own chunk" when that block shares a section with
other, normal-sized content (seen for real on the ML tutorial pages and
the error-messages index -- see parsing.md's page-size-outliers finding).
All chunks produced this way from the same heading section share that
section's `heading_path`, since they're still logically one section, just
physically split for size.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from databricks_docs_vectordb.parsing.config import Settings
from databricks_docs_vectordb.parsing.html_walker import Heading, Node
from databricks_docs_vectordb.parsing.markdown import render
from databricks_docs_vectordb.parsing.metadata import PageMetadata
from databricks_docs_vectordb.parsing.store import Chunk


@dataclass(frozen=True)
class _ChunkPlan:
    heading_path: str
    text: str
    is_oversized: bool


def chunk_page(
    nodes: Sequence[Node], metadata: PageMetadata, settings: Settings
) -> list[Chunk]:
    """Turn one page's node tree into its ordered, linked chunks.

    Empty `nodes` (a page with no extractable content) produces an empty
    list -- it's `verify`'s job (Step 8) to flag that as a violation, not
    this function's job to prevent it.
    """
    plans: list[_ChunkPlan] = []
    for group in _group_by_heading_level(nodes, level=2):
        plans.extend(_emit_for_group(group, settings.target_chunk_size))
    if not plans:
        return []

    texts = _apply_overlap(plans, settings)
    chunk_ids = [f"{metadata.url}#{i}" for i in range(len(plans))]
    return [
        Chunk(
            chunk_id=chunk_ids[i],
            url=metadata.url,
            source=metadata.source,
            cloud=metadata.cloud,
            chunk_index=i,
            prev_chunk_id=chunk_ids[i - 1] if i > 0 else None,
            next_chunk_id=chunk_ids[i + 1] if i < len(plans) - 1 else None,
            heading_path=plan.heading_path,
            section_path=metadata.section_path,
            doc_title=metadata.doc_title,
            description=metadata.description,
            text=text,
            char_count=len(text),
            token_count=None,
            token_count_model=None,
            is_oversized=plan.is_oversized,
            last_updated=metadata.last_updated,
            page_sha256=metadata.page_sha256,
            fetched_at=metadata.fetched_at,
        )
        for i, (plan, text) in enumerate(zip(plans, texts, strict=True))
    ]


# -- heading-based grouping -----------------------------------------------


def _group_by_heading_level(nodes: Sequence[Node], level: int) -> list[list[Node]]:
    """Split a node list at each `Heading` of exactly `level`.

    Nodes before the first such heading form a leading, headless group
    (omitted entirely if there are none). Reused for both the top-level
    `<h2>` split and the recursive `<h3>` split inside an oversized section.
    """
    groups: list[list[Node]] = []
    current: list[Node] = []
    for node in nodes:
        if isinstance(node, Heading) and node.level == level:
            if current:
                groups.append(current)
            current = [node]
        else:
            current.append(node)
    if current:
        groups.append(current)
    return groups


def _group_heading_path(
    group: Sequence[Node], parent_h2_text: str | None = None
) -> str:
    """This group's own heading breadcrumb, prefixed with its parent `<h2>`'s
    text unless the group already contains that heading itself (it does
    when a heading-only leading `<h3>` group got folded into it -- see
    `_emit_for_group`)."""
    own = " > ".join(n.text for n in group if isinstance(n, Heading))
    has_own_h2 = any(isinstance(n, Heading) and n.level == 2 for n in group)
    if parent_h2_text and not has_own_h2:
        return f"{parent_h2_text} > {own}" if own else parent_h2_text
    return own


# -- per-section emission ---------------------------------------------------


def _emit_for_group(group: list[Node], target: int) -> list[_ChunkPlan]:
    heading_path = _group_heading_path(group)
    text = render(group)
    if len(text) <= target:
        return [_ChunkPlan(heading_path, text, is_oversized=False)]

    subgroups = _group_by_heading_level(group, level=3)
    if len(subgroups) <= 1:
        # No <h3>s to split on: any remaining excess goes through the
        # size-based packing fallback directly.
        return _pack_group(group, target, heading_path)

    # A heading-only leading group (just the parent <h2>, no intro content
    # before the first <h3>) would otherwise become its own near-empty
    # chunk; fold it into the first real <h3> group instead.
    if len(subgroups[0]) == 1:
        subgroups[1] = subgroups[0] + subgroups[1]
        subgroups = subgroups[1:]

    parent_h2_text = group[0].text if isinstance(group[0], Heading) else None

    plans = []
    for subgroup in subgroups:
        subgroup_heading_path = _group_heading_path(subgroup, parent_h2_text)
        subgroup_text = render(subgroup)
        if len(subgroup_text) <= target:
            plans.append(_ChunkPlan(subgroup_heading_path, subgroup_text, False))
        else:
            # <h3> is the deepest level parsing.md specifies splitting at;
            # anything still oversized here falls through to packing too.
            plans.extend(_pack_group(subgroup, target, subgroup_heading_path))
    return plans


def _pack_group(group: list[Node], target: int, heading_path: str) -> list[_ChunkPlan]:
    plans = []
    for node_group in _pack_nodes(group, target):
        text = render(node_group)
        plans.append(_ChunkPlan(heading_path, text, is_oversized=len(text) > target))
    return plans


def _pack_nodes(nodes: Sequence[Node], target: int) -> list[list[Node]]:
    """Greedily pack nodes into groups that render to at most `target` chars.

    A single node whose own rendered size already exceeds `target` (a
    table or code block too big to fit, most commonly) is never merged
    with neighbors -- it becomes its own group, isolated rather than cut.
    """
    groups: list[list[Node]] = []
    current: list[Node] = []
    current_size = 0
    for node in nodes:
        node_size = len(render([node]))
        if node_size > target:
            if current:
                groups.append(current)
                current, current_size = [], 0
            groups.append([node])
            continue

        # +2 for the blank-line join `render()` adds between blocks.
        joined_size = current_size + node_size + (2 if current else 0)
        if current and joined_size > target:
            groups.append(current)
            current, current_size = [node], node_size
        else:
            current.append(node)
            current_size = joined_size
    if current:
        groups.append(current)
    return groups


# -- overlap -----------------------------------------------------------

_OVERLAP_JOIN = "\n\n"  # separator between prepended overlap and a chunk's own text


def overlap_ceiling(settings: Settings) -> int:
    """The largest a chunk's final stored text can legitimately be once
    overlap is applied -- what `verify.py` checks `char_count` against
    before flagging a chunk `chunk-too-large` that isn't marked
    `is_oversized`.

    Exact for the default "chars" mode: a chunk's own content is at most
    `target_chunk_size` (or it would already be `is_oversized`), plus at
    most `overlap_size` characters of the previous chunk's tail, plus the
    fixed join separator. Only approximate for "heading" mode: restating
    the previous chunk's `heading_path` has no fixed length, so this uses
    a generous fixed margin instead of a precise bound -- a known
    limitation of that mode, not a bug (see parsing.md's revisit triggers).
    """
    if settings.overlap_mode == "heading":
        return settings.target_chunk_size + 200
    return settings.target_chunk_size + settings.overlap_size + len(_OVERLAP_JOIN)


def _apply_overlap(plans: Sequence[_ChunkPlan], settings: Settings) -> list[str]:
    """Each chunk's final text, with configured overlap prepended from the
    previous chunk. The first chunk has no previous chunk to draw from."""
    if not plans:
        return []
    final = [plans[0].text]
    for i in range(1, len(plans)):
        previous, current = plans[i - 1], plans[i]
        if settings.overlap_mode == "heading":
            final.append(_prefix_with_heading(previous.heading_path, current.text))
        else:
            final.append(
                _prefix_with_tail(previous.text, current.text, settings.overlap_size)
            )
    return final


def own_texts(chunks: Sequence[Chunk], settings: Settings) -> list[str]:
    """Each chunk's own text, without the overlap `chunk_page` prepended:
    the inverse of `_apply_overlap`, for showing what a chunk itself says.

    `chunks` is a whole page, in order, as stored (`ChunkStore.get_chunks`).
    Each prefix is rebuilt the way it was added, so a text that doesn't start
    with it (one stored under other overlap settings, say) comes back whole.
    """
    owns: list[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            owns.append(chunk.text)
            continue
        if settings.overlap_mode == "heading":
            prefix = _prefix_with_heading(chunks[i - 1].heading_path, "")
        else:
            prefix = _prefix_with_tail(owns[-1], "", settings.overlap_size)
        owns.append(chunk.text.removeprefix(prefix))
    return owns


def _prefix_with_tail(previous_text: str, text: str, overlap_size: int) -> str:
    tail = previous_text[-overlap_size:] if overlap_size else ""
    return f"{tail}{_OVERLAP_JOIN}{text}" if tail else text


def _prefix_with_heading(heading_path: str, text: str) -> str:
    if not heading_path:
        return text
    return f'*(continued from "{heading_path}")*{_OVERLAP_JOIN}{text}'
