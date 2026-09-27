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
