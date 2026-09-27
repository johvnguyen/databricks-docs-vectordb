# TODO(chunking_and_embedding.md Step 1): model (default
# "text-embedding-3-small", passed straight to litellm.embedding()),
# batch_size and batch_token_budget, rate_rpm/rate_tpm, concurrency,
# max_attempts/backoff_base/backoff_max/retry_after_max, breaker_threshold/
# breaker_cooldown/breaker_max_trips, the token overlap used when splitting
# oversized chunks, and embeddings_path (default ./data/embeddings.sqlite).
# Follows the pattern in scraping/config.py: a frozen Settings dataclass
# with defaults overridable via env var and CLI flag.
