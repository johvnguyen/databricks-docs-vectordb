"""Real token counts, via `tiktoken`, for whatever model this phase is
configured to embed with (`embedding/config.py`'s `Settings.model`).

Phase 2 sized chunks in characters because no embedding model -- and so no
tokenizer -- existed yet (`parsing.md`'s Resolved question 6). Now that one
does, this is the natural place to compute a real count: for the
`parsed.sqlite.chunks.token_count` backfill, and for Step 4's splitting
logic to size against the model's actual token limit instead of a
char-based guess.
"""

from functools import lru_cache

import tiktoken

from databricks_docs_vectordb.parsing.store import ChunkStore

# Used for a model tiktoken doesn't recognize by name (e.g. a non-OpenAI
# model routed through LiteLLM, per Resolved question 1's "configurable to
# different models") -- an approximation, not an exact count, since we
# don't ship a tokenizer per provider. Every OpenAI embedding model
# tiktoken does know resolves to this same encoding anyway.
_FALLBACK_ENCODING = "cl100k_base"


@lru_cache(maxsize=None)
def get_encoding(model: str) -> tiktoken.Encoding:
    """The tiktoken encoding for `model`, cached.

    Cached because loading an encoding's BPE ranks isn't free, and this
    resolves once per model but `count_tokens` is called once per chunk
    (tens of thousands of times per run).
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def count_tokens(text: str, model: str) -> int:
    """How many tokens `model`'s tokenizer would bill `text` as."""
    return len(get_encoding(model).encode(text))


def backfill_token_counts(store: ChunkStore, model: str) -> int:
    """Fill in a real `token_count` for every chunk that doesn't have one
    yet, using `model`'s tokenizer.

    `parsed.sqlite.chunks.token_count` is `NULL` on every row until this
    runs -- Phase 2 never picks a tokenizer (`parsing.md`'s Resolved
    question 6), since no embedding model existed yet. A metadata fill-in
    only: `parsed.sqlite`'s `text` and chunk boundaries are untouched, and
    a chunk that already has a `token_count` is left alone -- so a second
    run with nothing changed touches nothing, and a run after a later
    re-parse only touches the pages that were actually re-parsed (`chunker.py`
    still doesn't set `token_count` on the chunks it produces).

    Returns how many chunks were updated.
    """
    missing = store.chunks_missing_token_count()
    if not missing:
        return 0
    counts = {chunk_id: count_tokens(text, model) for chunk_id, text in missing}
    return store.set_token_counts(counts)
