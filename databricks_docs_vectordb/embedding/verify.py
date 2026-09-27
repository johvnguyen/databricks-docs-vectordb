# TODO(chunking_and_embedding.md Step 8): check the embeddings store
# against its invariants, no network. Every parsed.sqlite chunk has a
# current, matching-hash, matching-model embeddings row set (or an
# explicit failed/rejected one); sub_index/sub_count are internally
# consistent; every embedding BLOB's length matches its dimensions.
# Reports counts by (provider, model, model_version) and by status.
