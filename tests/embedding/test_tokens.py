import tiktoken

from databricks_docs_vectordb.embedding.tokens import count_tokens, get_encoding

OPENAI_SMALL = "text-embedding-3-small"
OPENAI_LARGE = "text-embedding-3-large"


def test_count_tokens_matches_known_tiktoken_counts():
    # cl100k_base, confirmed against the real tiktoken package.
    assert count_tokens("hello world", OPENAI_SMALL) == 2
    assert count_tokens("Delta Lake", OPENAI_SMALL) == 2
    assert count_tokens("tokenization", OPENAI_SMALL) == 2


def test_empty_string_has_zero_tokens():
    assert count_tokens("", OPENAI_SMALL) == 0


def test_get_encoding_resolves_openai_embedding_models_to_cl100k_base():
    assert get_encoding(OPENAI_SMALL).name == "cl100k_base"
    assert get_encoding(OPENAI_LARGE).name == "cl100k_base"


def test_unknown_model_falls_back_to_cl100k_base_instead_of_raising():
    # A non-OpenAI model routed through LiteLLM -- tiktoken doesn't know it
    # by name, but count_tokens must still return an approximation, not
    # raise (Resolved question 1's "configurable to different models").
    assert get_encoding("cohere/embed-english-v3.0").name == "cl100k_base"
    assert count_tokens("hello world", "cohere/embed-english-v3.0") == 2


def test_get_encoding_is_cached():
    assert get_encoding(OPENAI_SMALL) is get_encoding(OPENAI_SMALL)


def test_count_tokens_agrees_with_encoding_the_text_directly():
    text = "The quick brown fox jumps over the lazy dog."
    encoding = tiktoken.encoding_for_model(OPENAI_SMALL)
    assert count_tokens(text, OPENAI_SMALL) == len(encoding.encode(text))
