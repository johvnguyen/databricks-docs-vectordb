# TODO(parsing.md Step 7): per ok manifest row, skip if the store already
# has chunks for that url with a matching page_sha256 (idempotent,
# incremental re-parse); otherwise walk -> extract metadata -> chunk ->
# store.replace_page_chunks. Wired up from cli.py's `parse run`.
