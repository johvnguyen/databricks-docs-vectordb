# TODO(parsing.md Step 6): split a node tree + metadata into ordered chunks.
# Heading-based (split at <h2>, further at <h3> if still oversized); never
# split a table or code block (an oversized one becomes its own chunk,
# flagged is_oversized); configurable target size and overlap; assign
# chunk_index and prev_chunk_id/next_chunk_id for adjacency-based retrieval.
