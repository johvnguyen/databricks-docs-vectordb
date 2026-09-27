# TODO(chunking_and_embedding.md Step 3): a tiktoken encoder for the
# configured model; count_tokens(text) -> int. Also backfills
# parsed.sqlite's token_count column (NULL on every row today) using this --
# a metadata fill-in, not a re-chunk; parsed.sqlite's text and chunk
# boundaries stay untouched.
