# TODO(chunking_and_embedding.md Step 5): a thin wrapper around
# litellm.embedding(model=, input=), returning vectors plus the
# (provider, model, dimensions) actually reported back in the response
# (not just echoed from config, so a provider-side substitution is
# caught). Raises a typed error distinguishing retryable (rate limit,
# timeout, 5xx) from rejected (content policy).
