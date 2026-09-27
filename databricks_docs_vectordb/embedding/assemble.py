# TODO(chunking_and_embedding.md Step 4): build the embedding input(s) for
# one chunk -- doc_title/heading_path/section_path/blank-line/chunk.text
# when it fits the model's token limit; otherwise slice chunk.text into
# token-bounded, overlapping pieces (limit minus the prepended context's
# token cost, minus the configured overlap) and re-prepend context to each,
# returning 1 input for a normal chunk or N for an oversized one.
