# TODO(parsing.md Step 8): check the chunk store against its invariants, no
# network. Every ok manifest row has >=1 chunk; no chunk's text is empty;
# no chunk exceeds the target size ceiling unless is_oversized; every
# prev_chunk_id/next_chunk_id resolves within the same url; required
# metadata fields are non-null. --by-section reports per source/section:
# page count, chunk count, avg/median chunk size, oversized-chunk count.
