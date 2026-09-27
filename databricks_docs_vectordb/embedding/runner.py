# TODO(chunking_and_embedding.md Step 6): candidate selection (no
# embeddings row, a chunk_text_hash mismatch, a (provider, model,
# model_version) mismatch, or status='failed' -- all automatic, no flag),
# batching bounded by batch_size and batch_token_budget, a shared
# token-bucket limiter throttling both RPM and TPM, retry with
# backoff+jitter on retryable errors, a circuit breaker that pauses after
# breaker_threshold consecutive failures and ends the run after
# breaker_max_trips. Follows scraping/runner.py's shape, including SIGINT
# handling.
