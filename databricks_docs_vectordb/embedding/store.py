# TODO(chunking_and_embedding.md Step 2): the SQLite embeddings store
# (data/embeddings.sqlite, table `embeddings`), following the pattern in
# parsing/store.py. upsert(chunk_id, rows) deletes then inserts all of a
# chunk_id's sub_index rows in one transaction (so a chunk that used to
# split into several pieces and now needs only one doesn't leave orphaned
# rows); get_hash(chunk_id); get_model_info(chunk_id); iteration for
# status/verify.
