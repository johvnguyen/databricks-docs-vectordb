"""Build the embedding input(s) for one chunk (chunking_and_embedding.md's
"What gets embedded" and "Oversized chunks").

Each input is `chunk.text` prefixed with its page/section context:

    {doc_title}
    {heading_path}
    {section_path}

    {chunk.text}

assembled fresh at embed time and never written back to `parsed.sqlite`.
When that doesn't fit the model's token limit, `chunk.text` is sliced into
consecutive, overlapping, token-bounded pieces and the same context is
re-prepended to each -- several vectors per chunk, not an average.

Slices are cut at token boundaries but taken as substrings of `chunk.text`
(via tiktoken's per-token character offsets), so a slice never ends in half
a multi-byte character. Every assembled input is re-counted as the exact
string that will be sent, and a slice that re-tokenizes over the limit is
shortened before it's returned -- nothing is left for the provider to
silently truncate.
"""

from typing import NamedTuple

from databricks_docs_vectordb.embedding.tokens import count_tokens, get_encoding
from databricks_docs_vectordb.parsing.store import Chunk


class EmbeddingInput(NamedTuple):
    text: str
    token_count: int


def build_context(chunk: Chunk) -> str:
    """The context prepended to every input for `chunk`, including the
    blank line that separates it from the chunk's own text."""
    return f"{chunk.doc_title}\n{chunk.heading_path}\n{chunk.section_path}\n\n"


def assemble_inputs(
    chunk: Chunk, *, model: str, token_limit: int, overlap_tokens: int
) -> list[EmbeddingInput]:
    """The embedding input(s) for `chunk`: one when context + text fits in
    `token_limit` tokens, otherwise one per overlapping slice of
    `chunk.text`, in order (list position is the row's `sub_index`).

    Each slice after the first starts with the last `overlap_tokens` tokens
    of the previous one. Raises ValueError if the context alone leaves no
    room for text beyond the overlap.
    """
    if overlap_tokens < 0:
        raise ValueError(f"overlap_tokens must be >= 0, got {overlap_tokens}")

    context = build_context(chunk)
    whole = context + chunk.text
    whole_count = count_tokens(whole, model)
    if whole_count <= token_limit:
        return [EmbeddingInput(whole, whole_count)]

    # Per-slice text budget. Joining context and slice can tokenize
    # slightly differently than either alone, so this is a starting point
    # that the re-count below corrects, not a guarantee.
    window = token_limit - count_tokens(context, model)
    if window - overlap_tokens <= 0:
        raise ValueError(
            f"{chunk.chunk_id}: context uses {token_limit - window} of "
            f"{token_limit} tokens, leaving no room for text beyond the "
            f"{overlap_tokens}-token overlap"
        )

    encoding = get_encoding(model)
    # Special-token text is ordinary text here, as in `count_tokens`.
    tokens = encoding.encode(chunk.text, disallowed_special=())
    _, offsets = encoding.decode_with_offsets(tokens)

    def char_offset(token_index: int) -> int:
        return offsets[token_index] if token_index < len(tokens) else len(chunk.text)

    inputs: list[EmbeddingInput] = []
    start = 0
    while True:
        end = min(start + window, len(tokens))
        while True:
            text = context + chunk.text[char_offset(start) : char_offset(end)]
            count = count_tokens(text, model)
            if count <= token_limit:
                break
            end -= count - token_limit
        if end - start <= overlap_tokens and end < len(tokens):
            # Shrinking ate the whole window: the next slice would start at
            # or before this one, so splitting could never finish.
            raise ValueError(
                f"{chunk.chunk_id}: slice at token {start} can't fit more "
                f"than the {overlap_tokens}-token overlap within {token_limit} tokens"
            )
        inputs.append(EmbeddingInput(text, count))
        if end >= len(tokens):
            return inputs
        start = end - overlap_tokens
