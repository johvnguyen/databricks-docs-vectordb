"""A thin wrapper around LiteLLM's embedding call (chunking_and_embedding.md's
"Provider and model", "Failure handling", and "Versioning").

Two jobs beyond making the call:

- Report what *actually* produced the vectors -- the provider LiteLLM
  routed to, the model name the provider echoed back, and the vector length
  -- rather than echoing config, so a provider-side substitution shows up
  in the stored rows (Resolved question 8).
- Turn LiteLLM's many exception types into three this phase acts on:
  `RetryableError` (rate limit, timeout, connection, 5xx -- back off and
  retry), `RejectedError` (content policy -- terminal, stored as
  `status='rejected'`), and plain `EmbeddingError` for everything else
  (auth, bad request, a malformed response -- retrying the same call won't
  help, but the chunk is left `failed` for the next run).

Async (`litellm.aembedding`, the same call shape as `litellm.embedding`)
because the Step 6 runner mirrors `scraping/runner.py`'s asyncio shape.
LiteLLM/OpenAI-SDK-level retries are turned off so the runner's backoff,
rate limiter, and circuit breaker are the only retry policy in play.
"""

import time
from collections.abc import Mapping, Sequence
from email.utils import parsedate_to_datetime
from typing import Any, NamedTuple

import litellm
import openai
from litellm import exceptions as llm_errors

from databricks_docs_vectordb.embedding.store import ModelInfo


class EmbeddingError(Exception):
    """An embedding call failed in a way retrying it now won't fix."""


class RetryableError(EmbeddingError):
    """Transient: rate limit, timeout, connection failure, or 5xx."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        # Seconds the provider asked us to wait, when it said (Retry-After).
        self.retry_after = retry_after


class RejectedError(EmbeddingError):
    """The provider refused the input on content-policy grounds. Terminal:
    re-sending the same text gets the same answer."""


class EmbeddingBatch(NamedTuple):
    vectors: list[list[float]]  # one per input, in input order
    provider: str
    model: str
    model_version: str | None
    dimensions: int


_RETRYABLE = (
    llm_errors.RateLimitError,
    llm_errors.Timeout,
    llm_errors.APIConnectionError,
    llm_errors.InternalServerError,
    llm_errors.ServiceUnavailableError,
    llm_errors.BadGatewayError,
)


def configured_model_info(model: str) -> ModelInfo:
    """The `(provider, model, model_version)` a config `model` string should
    produce, as LiteLLM resolves it (`"azure/my-deployment"` -> provider
    `azure`, model `my-deployment`) -- what the runner compares stored rows
    against, and what a failed or rejected row records, since no response
    came back to report anything. `model_version` is None for the same
    reason `embed` reports None (see Versioning)."""
    name, provider, _, _ = litellm.get_llm_provider(model)
    return ModelInfo(provider, name, None)


async def embed(
    texts: Sequence[str], *, model: str, timeout: float = 60.0
) -> EmbeddingBatch:
    """Embed `texts` with `model` in one request.

    Raises `RetryableError`, `RejectedError`, or `EmbeddingError`; any other
    exception is a bug here, not a provider failure, and propagates as-is.
    """
    if not texts:
        raise ValueError("texts must not be empty")
    try:
        response = await litellm.aembedding(
            model=model, input=list(texts), timeout=timeout, max_retries=0
        )
    except llm_errors.ContentPolicyViolationError as e:
        raise RejectedError(str(e)) from e
    except _RETRYABLE as e:
        raise RetryableError(str(e), _retry_after(e)) from e
    except openai.OpenAIError as e:
        # LiteLLM's exception classes all subclass the OpenAI SDK's, whatever
        # the provider. Anything left here is a non-retryable 4xx (auth, bad
        # request, context window, ...) -- or a 5xx LiteLLM didn't map to
        # one of the specific classes above.
        if (getattr(e, "status_code", None) or 0) >= 500:
            raise RetryableError(str(e), _retry_after(e)) from e
        raise EmbeddingError(str(e)) from e
    return _read_response(response, requested_model=model, count=len(texts))


def _read_response(
    response: Any, *, requested_model: str, count: int
) -> EmbeddingBatch:
    data = sorted(response.data, key=lambda item: _field(item, "index"))
    if len(data) != count:
        raise EmbeddingError(f"sent {count} inputs, got {len(data)} embeddings back")
    vectors = [list(_field(item, "embedding")) for item in data]
    dimensions = {len(v) for v in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise EmbeddingError(
            f"inconsistent vector lengths in one response: {sorted(dimensions)}"
        )

    # What LiteLLM routed to and what the provider says served the request.
    # Each falls back to how LiteLLM resolves the configured name only when
    # the response doesn't say.
    hidden = getattr(response, "_hidden_params", None) or {}
    fallback = configured_model_info(requested_model)
    return EmbeddingBatch(
        vectors=vectors,
        provider=hidden.get("custom_llm_provider") or fallback.provider,
        model=response.model or fallback.model,
        # No provider we use today reports a snapshot distinct from the
        # model name (OpenAI's text-embedding-3-* don't) -- see Versioning.
        model_version=None,
        dimensions=dimensions.pop(),
    )


def _field(item: Any, name: str) -> Any:
    # LiteLLM hands back `Embedding` objects from some providers and plain
    # dicts from others.
    return item[name] if isinstance(item, Mapping) else getattr(item, name)


def _retry_after(error: Exception) -> float | None:
    """Seconds from `retry-after-ms` / `Retry-After` on the failed response,
    if the provider sent either."""
    headers = getattr(error, "litellm_response_headers", None)
    if headers is None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
    if not headers:
        return None
    headers = {str(k).lower(): str(v) for k, v in dict(headers).items()}

    if (ms := headers.get("retry-after-ms")) is not None:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None
