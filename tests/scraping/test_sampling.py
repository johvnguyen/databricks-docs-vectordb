import hashlib
import random
from collections import Counter

import pytest

from databricks_docs_vectordb.scraping.config import SOURCES
from databricks_docs_vectordb.scraping.manifest import Page
from databricks_docs_vectordb.scraping.sampling import SampleError, sample_pages
from databricks_docs_vectordb.scraping.verify import TOP_LEVEL, section_of

AWS = "https://docs.databricks.com/aws/en/"
API = "https://docs.databricks.com/api/"


def page(url: str, source: str | None = None) -> Page:
    return Page(
        url=url,
        source=source or ("api" if url.startswith(API) else "aws-en"),
        rel_path="x",
        status="pending",
        http_status=None,
        final_url=None,
        etag=None,
        last_modified=None,
        sha256=None,
        size_bytes=None,
        first_seen_at="2026-09-21T00:00:00Z",
        fetched_at=None,
        checked_at=None,
        attempts=0,
        error=None,
    )


def corpus(**sections: int) -> list[Page]:
    """Pages for aws-en sections with the given sizes, e.g. corpus(sql=100, delta=3)."""
    return [
        page(f"{AWS}{name}/p{i:03d}")
        for name, size in sections.items()
        for i in range(size)
    ]


def per_section(sample) -> Counter:
    return Counter(section_of(p, SOURCES) for p in sample.pages)


# -- coverage ------------------------------------------------------------


def test_every_section_gets_at_least_two_pages():
    pages = corpus(sql=500, pyspark=400, delta=10, tiny=2, big=50)
    sample = sample_pages(pages, SOURCES, 30)
    counts = per_section(sample)
    assert set(counts) == {
        ("aws-en", s) for s in ("sql", "pyspark", "delta", "tiny", "big")
    }
    assert all(n >= 2 for n in counts.values())


def test_a_section_with_one_page_contributes_that_page():
    pages = corpus(sql=50, lonely=1)
    sample = sample_pages(pages, SOURCES, 10)
    assert per_section(sample)[("aws-en", "lonely")] == 1


def test_sections_of_every_source_are_covered_including_top_level_pages():
    pages = corpus(sql=20) + [
        page(API + "workspace/a"),
        page(API + "workspace/b"),
        page(API + "iam/v2/x"),
        page(AWS + "introduction"),
        page(AWS + "faq"),
        page(API),
    ]
    counts = per_section(sample_pages(pages, SOURCES, 12))
    assert counts[("api", "workspace")] == 2
    assert counts[("api", "iam")] == 1
    assert counts[("api", TOP_LEVEL)] == 1
    assert counts[("aws-en", TOP_LEVEL)] == 2
    assert counts[("aws-en", "sql")] >= 2


def test_the_floor_is_configurable():
    pages = corpus(a=20, b=20)
    assert per_section(sample_pages(pages, SOURCES, 6, floor=3)) == Counter(
        {("aws-en", "a"): 3, ("aws-en", "b"): 3}
    )
    assert sample_pages(pages, SOURCES, 2, floor=1).minimum == 2


# -- size ----------------------------------------------------------------


def test_the_sample_is_exactly_n_pages():
    pages = corpus(sql=300, pyspark=200, delta=40, tiny=2)
    for n in (8, 9, 25, 100, 400):
        assert len(sample_pages(pages, SOURCES, n).pages) == n


def test_at_the_minimum_every_section_gets_exactly_the_floor():
    pages = corpus(sql=300, delta=40, tiny=2, lonely=1)
    sample = sample_pages(pages, SOURCES, 7)
    assert sample.minimum == 7
    assert per_section(sample) == Counter(
        {
            ("aws-en", "sql"): 2,
            ("aws-en", "delta"): 2,
            ("aws-en", "tiny"): 2,
            ("aws-en", "lonely"): 1,
        }
    )


def test_a_sample_larger_than_the_candidates_returns_them_all():
    pages = corpus(sql=30, delta=5)
    sample = sample_pages(pages, SOURCES, 10_000)
    assert sorted(p.url for p in sample.pages) == sorted(p.url for p in pages)


def test_extra_pages_go_to_the_largest_sections():
    pages = corpus(huge=1000, medium=100, small=10, tiny=3)
    counts = per_section(sample_pages(pages, SOURCES, 100))
    huge, medium, small = (counts[("aws-en", s)] for s in ("huge", "medium", "small"))
    assert huge > medium > small
    # proportional, not merely "a bit more": 1000 pages vs 100 pages
    assert huge >= 4 * medium
    assert small >= 2 and counts[("aws-en", "tiny")] >= 2


def test_no_section_gives_more_pages_than_it_has():
    pages = corpus(a=3, b=5, c=200)
    for n in (10, 50, 150, 208):
        sample = sample_pages(pages, SOURCES, n)
        counts = per_section(sample)
        assert counts[("aws-en", "a")] <= 3 and counts[("aws-en", "b")] <= 5
        assert len(sample.pages) == min(n, 208)


def test_few_extras_go_one_each_to_the_sections_with_the_most_room():
    pages = corpus(a=100, b=90, c=80, d=2)
    counts = per_section(sample_pages(pages, SOURCES, 9))  # minimum 8, one extra
    assert counts[("aws-en", "a")] == 3
    assert counts[("aws-en", "b")] == 2 and counts[("aws-en", "c")] == 2


# -- too small -----------------------------------------------------------


def test_an_n_below_the_minimum_says_how_many_are_needed():
    pages = corpus(a=10, b=10, c=10, lonely=1)
    with pytest.raises(SampleError) as excinfo:
        sample_pages(pages, SOURCES, 5)
    message = str(excinfo.value)
    assert "--sample 5 is too small" in message
    assert "all 4 sections" in message
    assert "takes 7 pages" in message and "--sample 7" in message


def test_the_minimum_is_reported_on_every_sample():
    pages = corpus(a=10, b=10)
    assert sample_pages(pages, SOURCES, 20).minimum == 4


# -- determinism ---------------------------------------------------------


def test_the_same_candidates_give_the_same_sample_in_the_same_order():
    pages = corpus(sql=300, pyspark=200, delta=40)
    first = sample_pages(pages, SOURCES, 60)
    second = sample_pages(pages, SOURCES, 60)
    assert [p.url for p in first.pages] == [p.url for p in second.pages]
    assert first.allocation == second.allocation


def test_the_sample_does_not_depend_on_the_order_candidates_arrive_in():
    pages = corpus(sql=300, pyspark=200, delta=40)
    shuffled = pages[:]
    random.Random(7).shuffle(shuffled)
    assert [p.url for p in sample_pages(pages, SOURCES, 60).pages] == [
        p.url for p in sample_pages(shuffled, SOURCES, 60).pages
    ]


def test_within_a_section_pages_are_ranked_by_url_hash_not_alphabetically():
    pages = corpus(sql=50)
    picked = {p.url for p in sample_pages(pages, SOURCES, 2).pages}
    ranked = sorted(
        (p.url for p in pages), key=lambda u: hashlib.sha256(u.encode()).hexdigest()
    )
    assert picked == set(ranked[:2])
    assert picked != {f"{AWS}sql/p000", f"{AWS}sql/p001"}  # not just the first two


def test_the_fetch_order_mixes_sections_so_an_early_stop_still_spans_them():
    pages = corpus(a=40, b=40, c=40, d=40, e=40)
    first_ten = sample_pages(pages, SOURCES, 50).pages[:10]
    assert len({section_of(p, SOURCES) for p in first_ten}) >= 3


def test_a_sample_is_a_subset_without_duplicates():
    pages = corpus(sql=100, delta=20)
    sample = sample_pages(pages, SOURCES, 40)
    urls = [p.url for p in sample.pages]
    assert len(urls) == len(set(urls))
    assert set(urls) <= {p.url for p in pages}


def test_taking_the_sample_leaves_the_candidates_alone():
    pages = corpus(sql=30, delta=10)
    before = [p.url for p in pages]
    sample_pages(pages, SOURCES, 12)
    assert [p.url for p in pages] == before


# -- edges ---------------------------------------------------------------


def test_no_candidates_gives_an_empty_sample():
    sample = sample_pages([], SOURCES, 50)
    assert sample.pages == [] and sample.minimum == 0 and sample.sections == 0


def test_sample_reports_what_it_drew_from():
    pages = corpus(a=10, b=10, c=10)
    sample = sample_pages(pages, SOURCES, 9)
    assert (sample.candidates, sample.sections) == (30, 3)
    assert sum(sample.allocation.values()) == 9
