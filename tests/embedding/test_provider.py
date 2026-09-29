import httpx
import litellm
import pytest
from litellm import exceptions as llm_errors
from litellm.types.utils import Embedding, EmbeddingResponse

from databricks_docs_vectordb.embedding.provider import (
    EmbeddingBatch,
    EmbeddingError,
    RejectedError,
    RetryableError,
    embed,
)

MODEL = "text-embedding-3-small"
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def make_response(
    vectors, *, model=MODEL, provider="openai", shuffle=False
) -> EmbeddingResponse:
    data = [
        Embedding(embedding=v, index=i, object="embedding")
        for i, v in enumerate(vectors)
    ]
    if shuffle:
        data.reverse()
    response = EmbeddingResponse(model=model, data=data)
    if provider is not None:
        response._hidden_params["custom_llm_provider"] = provider
    return response


def http_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers, request=REQUEST)


@pytest.fixture
def fake_litellm(monkeypatch):
    """Replace `litellm.aembedding`; set `.result` to a response or an
    exception to raise. Records every call's kwargs."""

    class Fake:
        def __init__(self) -> None:
            self.result: object = None
            self.calls: list[dict] = []

        async def __call__(self, **kwargs):
            self.calls.append(kwargs)
            if isinstance(self.result, BaseException):
                raise self.result
            return self.result

    fake = Fake()
    monkeypatch.setattr(litellm, "aembedding", fake)
    return fake


# -- success -----------------------------------------------------------------


async def test_normal_batch_returns_vectors_and_reported_model(fake_litellm):
    fake_litellm.result = make_response([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    batch = await embed(["a", "b"], model=MODEL)
    assert batch == EmbeddingBatch(
        vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        provider="openai",
        model=MODEL,
        model_version=None,
        dimensions=3,
    )


async def test_call_passes_model_input_and_disables_sdk_retries(fake_litellm):
    fake_litellm.result = make_response([[0.1]])
    await embed(("a",), model=MODEL, timeout=12.5)
    assert fake_litellm.calls == [
        {"model": MODEL, "input": ["a"], "timeout": 12.5, "max_retries": 0}
    ]


async def test_provider_side_model_substitution_is_reported_not_masked(fake_litellm):
    fake_litellm.result = make_response(
        [[0.1, 0.2]], model="text-embedding-3-small-2027", provider="azure"
    )
    batch = await embed(["a"], model=MODEL)
    assert (batch.provider, batch.model) == ("azure", "text-embedding-3-small-2027")


async def test_vectors_are_returned_in_input_order(fake_litellm):
    fake_litellm.result = make_response([[1.0], [2.0], [3.0]], shuffle=True)
    batch = await embed(["a", "b", "c"], model=MODEL)
    assert batch.vectors == [[1.0], [2.0], [3.0]]


async def test_dict_shaped_embeddings_are_accepted(fake_litellm):
    response = make_response([])
    response.data = [{"embedding": [0.5, 0.5], "index": 0, "object": "embedding"}]
    fake_litellm.result = response
    assert (await embed(["a"], model=MODEL)).vectors == [[0.5, 0.5]]


async def test_missing_provider_and_model_fall_back_to_litellm_resolution(fake_litellm):
    response = make_response([[0.1]], provider=None)
    response.model = None
    fake_litellm.result = response
    batch = await embed(["a"], model=MODEL)
    assert (batch.provider, batch.model) == ("openai", MODEL)


# -- malformed responses -----------------------------------------------------


async def test_wrong_vector_count_raises(fake_litellm):
    fake_litellm.result = make_response([[0.1]])
    with pytest.raises(EmbeddingError, match="sent 2 inputs, got 1"):
        await embed(["a", "b"], model=MODEL)


async def test_inconsistent_dimensions_raise(fake_litellm):
    fake_litellm.result = make_response([[0.1, 0.2], [0.3]])
    with pytest.raises(EmbeddingError, match="inconsistent"):
        await embed(["a", "b"], model=MODEL)


async def test_empty_input_is_a_caller_bug(fake_litellm):
    with pytest.raises(ValueError):
        await embed([], model=MODEL)
    assert fake_litellm.calls == []


# -- error classification ----------------------------------------------------


async def test_rate_limit_is_retryable_with_retry_after(fake_litellm):
    fake_litellm.result = llm_errors.RateLimitError(
        "slow down",
        llm_provider="openai",
        model=MODEL,
        response=http_response(429, {"Retry-After": "7"}),
    )
    with pytest.raises(RetryableError) as info:
        await embed(["a"], model=MODEL)
    assert info.value.retry_after == 7.0


async def test_retry_after_ms_wins_over_retry_after(fake_litellm):
    fake_litellm.result = llm_errors.RateLimitError(
        "slow down",
        llm_provider="openai",
        model=MODEL,
        response=http_response(429, {"retry-after-ms": "1500", "retry-after": "9"}),
    )
    with pytest.raises(RetryableError) as info:
        await embed(["a"], model=MODEL)
    assert info.value.retry_after == 1.5


async def test_timeout_is_retryable(fake_litellm):
    fake_litellm.result = llm_errors.Timeout(
        "timed out", model=MODEL, llm_provider="openai"
    )
    with pytest.raises(RetryableError) as info:
        await embed(["a"], model=MODEL)
    assert info.value.retry_after is None


async def test_connection_error_is_retryable(fake_litellm):
    fake_litellm.result = llm_errors.APIConnectionError(
        "reset", llm_provider="openai", model=MODEL, request=REQUEST
    )
    with pytest.raises(RetryableError):
        await embed(["a"], model=MODEL)


@pytest.mark.parametrize(
    "error",
    [
        llm_errors.InternalServerError(
            "boom", llm_provider="openai", model=MODEL, response=http_response(500)
        ),
        llm_errors.ServiceUnavailableError(
            "down", llm_provider="openai", model=MODEL, response=http_response(503)
        ),
        llm_errors.BadGatewayError(
            "gateway", llm_provider="openai", model=MODEL, response=http_response(502)
        ),
        llm_errors.APIError(
            504, "generic 5xx", llm_provider="openai", model=MODEL, request=REQUEST
        ),
    ],
    ids=["500", "503", "502", "unmapped-504"],
)
async def test_5xx_is_retryable(fake_litellm, error):
    fake_litellm.result = error
    with pytest.raises(RetryableError):
        await embed(["a"], model=MODEL)


async def test_content_policy_is_rejected_not_retryable(fake_litellm):
    fake_litellm.result = llm_errors.ContentPolicyViolationError(
        "flagged", model=MODEL, llm_provider="openai", response=http_response(400)
    )
    with pytest.raises(RejectedError) as info:
        await embed(["a"], model=MODEL)
    assert not isinstance(info.value, RetryableError)
    assert isinstance(info.value.__cause__, llm_errors.ContentPolicyViolationError)


@pytest.mark.parametrize(
    "error",
    [
        llm_errors.AuthenticationError(
            "bad key", llm_provider="openai", model=MODEL, response=http_response(401)
        ),
        llm_errors.BadRequestError(
            "bad", model=MODEL, llm_provider="openai", response=http_response(400)
        ),
        llm_errors.ContextWindowExceededError(
            "too long", model=MODEL, llm_provider="openai", response=http_response(400)
        ),
        llm_errors.NotFoundError(
            "no such model",
            model=MODEL,
            llm_provider="openai",
            response=http_response(404),
        ),
    ],
    ids=["auth", "bad-request", "context-window", "not-found"],
)
async def test_other_provider_errors_are_neither_retryable_nor_rejected(
    fake_litellm, error
):
    fake_litellm.result = error
    with pytest.raises(EmbeddingError) as info:
        await embed(["a"], model=MODEL)
    assert type(info.value) is EmbeddingError


async def test_non_provider_exceptions_propagate_unwrapped(fake_litellm):
    fake_litellm.result = KeyError("bug")
    with pytest.raises(KeyError):
        await embed(["a"], model=MODEL)
