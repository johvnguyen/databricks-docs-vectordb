from collections import Counter

import pytest

from databricks_docs_vectordb.embedding.runner import Candidate, text_hash
from databricks_docs_vectordb.embedding.sampling import sample_chunks
from databricks_docs_vectordb.parsing.store import Chunk
from databricks_docs_vectordb.scraping.config import SOURCES
from databricks_docs_vectordb.scraping.sampling import SampleError
from databricks_docs_vectordb.scraping.verify import section_of

AWS = "https://docs.databricks.com/aws/en/"


def make_chunk(url: str, index: int, text: str | None = None) -> Chunk:
    text = text if text is not None else f"{url}#{index} " + "x" * 50
    return Chunk(
        chunk_id=f"{url}#{index}",
        url=url,
        source="aws-en",
        cloud="aws",
        chunk_index=index,
        prev_chunk_id=None,
        next_chunk_id=None,
        heading_path="H",
        section_path="S",
        doc_title="T",
        description=None,
        text=text,
        char_count=len(text),
        token_count=None,
        token_count_model=None,
        is_oversized=False,
        last_updated=None,
        page_sha256="ab" * 32,
        fetched_at=None,
    )


def candidates(sections: dict[str, int], per_page: int = 3) -> list[Candidate]:
    """`sections` maps a top-level section to how many pages it has."""
    out = []
    for section, pages in sections.items():
        for p in range(pages):
            url = f"{AWS}{section}/page{p:03d}"
            for i in range(per_page):
                chunk = make_chunk(url, i)
                out.append(Candidate(chunk, text_hash(chunk.text), "new"))
    return out


def ids(sample) -> list[str]:
    return [c.chunk.chunk_id for c in sample.candidates]


def sections_of(sample) -> Counter:
    return Counter(section_of(c.chunk, SOURCES)[1] for c in sample.candidates)


def test_every_section_gets_at_least_two_chunks():
    pool = candidates({"delta": 30, "sql": 20, "tiny": 1})
    sample = sample_chunks(pool, 20)
    counts = sections_of(sample)
    assert set(counts) == {"delta", "sql", "tiny"}
    assert all(n >= 2 for n in counts.values())


def test_the_rest_goes_to_the_bigger_sections():
    sample = sample_chunks(candidates({"big": 100, "small": 5}), 40)
    counts = sections_of(sample)
    assert counts["big"] > counts["small"]


def test_chunks_of_one_page_are_not_taken_together():
    # One section, one page of 60 chunks: a URL-keyed ranking would tie them
    # all and take chunks 0, 1, 2, ... in order.
    pool = candidates({"one": 1}, per_page=60)
    sample = sample_chunks(pool, 10)
    indices = sorted(c.chunk.chunk_index for c in sample.candidates)
    assert indices != list(range(len(indices)))


def test_the_longest_and_shortest_are_always_included_and_go_first():
    pool = candidates({"delta": 40, "sql": 40})
    longest = make_chunk(f"{AWS}release-notes/big", 0, "y" * 378_873)
    shortest = make_chunk(f"{AWS}delta/empty", 0, "- ")
    pool += [
        Candidate(longest, text_hash(longest.text), "new"),
        Candidate(shortest, text_hash(shortest.text), "new"),
    ]
    sample = sample_chunks(pool, 10)
    assert ids(sample)[:2] == [longest.chunk_id, shortest.chunk_id]
    # the longest is its section's only chunk, so the draw took it anyway;
    # the shortest shares a big section and had to be added
    assert [label for label, _ in sample.extremes_added] == ["shortest"]
    assert len(ids(sample)) == len(set(ids(sample)))


def test_extremes_the_draw_missed_are_added_on_top():
    pool = candidates({"delta": 40, "sql": 40})
    longest = make_chunk(f"{AWS}delta/big", 0, "y" * 5000)
    shortest = make_chunk(f"{AWS}sql/empty", 0, "- ")
    pool += [
        Candidate(longest, text_hash(longest.text), "new"),
        Candidate(shortest, text_hash(shortest.text), "new"),
    ]
    sample = sample_chunks(pool, 10)
    assert ids(sample)[:2] == [longest.chunk_id, shortest.chunk_id]
    assert [label for label, _ in sample.extremes_added] == ["longest", "shortest"]
    assert len(sample.candidates) == 12


def test_a_single_candidate_is_both_extremes_once():
    pool = candidates({"delta": 1}, per_page=1)
    assert len(sample_chunks(pool, 5).candidates) == 1


def test_an_extreme_the_draw_already_took_is_not_added_twice():
    pool = candidates({"delta": 2}, per_page=1)  # 2 chunks, n=2 takes both
    sample = sample_chunks(pool, 2)
    assert sample.extremes_added == []
    assert len(sample.candidates) == 2


def test_a_sample_bigger_than_the_pool_takes_everything_once():
    pool = candidates({"delta": 3, "sql": 2})
    sample = sample_chunks(pool, 1000)
    assert sorted(ids(sample)) == sorted(c.chunk.chunk_id for c in pool)


def test_the_same_candidates_always_give_the_same_sample_in_any_order():
    pool = candidates({"delta": 30, "sql": 30, "ml": 10})
    first = ids(sample_chunks(pool, 25))
    assert ids(sample_chunks(list(reversed(pool)), 25)) == first


def test_a_sample_too_small_is_refused_with_the_number_needed_in_chunks():
    pool = candidates({"a": 5, "b": 5, "c": 5})
    with pytest.raises(SampleError, match=r"--sample 5 .* takes 6 chunks"):
        sample_chunks(pool, 5)


def test_an_empty_pool_is_an_empty_sample():
    sample = sample_chunks([], 10)
    assert sample.candidates == [] and sample.extremes_added == []


def test_reasons_travel_with_the_sampled_candidates():
    pool = candidates({"delta": 5})
    pool = [c._replace(reason="failed") if i % 2 else c for i, c in enumerate(pool)]
    sample = sample_chunks(pool, 1000)
    assert Counter(c.reason for c in sample.candidates) == Counter(
        c.reason for c in pool
    )
