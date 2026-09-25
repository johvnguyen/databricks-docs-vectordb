from databricks_docs_vectordb.parsing.report import (
    format_sections,
    overall_stats,
    section_of,
    section_stats,
)
from databricks_docs_vectordb.parsing.store import ChunkSummary
from databricks_docs_vectordb.scraping.config import SOURCES

AWS = "https://docs.databricks.com/aws/en/"
API = "https://docs.databricks.com/api/"


def summary(
    url: str, source: str, char_count: int, is_oversized: bool = False
) -> ChunkSummary:
    return ChunkSummary(
        url=url, source=source, char_count=char_count, is_oversized=is_oversized
    )


# -- section_of ------------------------------------------------------------


def test_section_of_a_nested_path():
    assert section_of(AWS + "sql/language-manual/", "aws-en", SOURCES) == "sql"


def test_section_of_a_lone_top_level_page():
    assert section_of(AWS + "introduction", "aws-en", SOURCES) == "(top-level)"


def test_section_of_the_source_root():
    assert section_of(AWS, "aws-en", SOURCES) == "(top-level)"


def test_section_of_an_unrecognized_source():
    assert section_of(AWS + "sql/", "gcp-en", SOURCES) == "(unknown)"


def test_section_of_a_url_outside_its_sources_prefix():
    assert section_of("https://example.com/sql/", "aws-en", SOURCES) == "(unknown)"


# -- section_stats / overall_stats -----------------------------------------


def test_section_stats_groups_by_source_and_section():
    summaries = [
        summary(AWS + "sql/a", "aws-en", 100),
        summary(AWS + "sql/b", "aws-en", 300),
        summary(API + "jobs/run", "api", 500),
    ]
    stats = section_stats(summaries, SOURCES)
    assert set(stats) == {("aws-en", "sql"), ("api", "jobs")}
    assert stats[("aws-en", "sql")].chunks == 2
    assert stats[("aws-en", "sql")].pages == 2


def test_section_stats_counts_distinct_pages_not_chunks():
    summaries = [
        summary(AWS + "sql/a", "aws-en", 100),
        summary(AWS + "sql/a", "aws-en", 200),  # same page, second chunk
        summary(AWS + "sql/a", "aws-en", 300),  # same page, third chunk
    ]
    stats = section_stats(summaries, SOURCES)
    assert stats[("aws-en", "sql")].pages == 1
    assert stats[("aws-en", "sql")].chunks == 3


def test_section_stats_avg_and_median_chars():
    summaries = [
        summary(AWS + "sql/a", "aws-en", 100),
        summary(AWS + "sql/b", "aws-en", 200),
        summary(AWS + "sql/c", "aws-en", 300),
    ]
    stats = section_stats(summaries, SOURCES)["aws-en", "sql"]
    assert stats.avg_chars == 200
    assert stats.median_chars == 200


def test_section_stats_oversized_count():
    summaries = [
        summary(AWS + "sql/a", "aws-en", 100, is_oversized=False),
        summary(AWS + "sql/b", "aws-en", 5000, is_oversized=True),
    ]
    stats = section_stats(summaries, SOURCES)["aws-en", "sql"]
    assert stats.oversized == 1


def test_overall_stats_is_not_an_average_of_per_section_averages():
    # sql: two chunks averaging 100; pyspark: one chunk of 10,000. A naive
    # "sum the section averages" total would say avg=5050 (wrong); the true
    # weighted average across all 3 chunks is (100+100+10000)/3.
    summaries = [
        summary(AWS + "sql/a", "aws-en", 50),
        summary(AWS + "sql/b", "aws-en", 150),
        summary(AWS + "pyspark/a", "aws-en", 10_000),
    ]
    overall = overall_stats(summaries)
    assert overall.chunks == 3
    assert overall.pages == 3
    assert overall.avg_chars == round((50 + 150 + 10_000) / 3)


def test_overall_stats_median_is_the_true_median_not_a_median_of_medians():
    summaries = [
        summary(f"{AWS}sql/{i}", "aws-en", size)
        for i, size in enumerate([10, 20, 30, 1000])
    ]
    overall = overall_stats(summaries)
    assert overall.median_chars == 25  # median of [10, 20, 30, 1000]


def test_empty_summaries_produce_zeroed_stats_not_a_crash():
    overall = overall_stats([])
    assert overall.pages == 0
    assert overall.chunks == 0
    assert overall.avg_chars == 0
    assert overall.median_chars == 0
    assert overall.oversized == 0


# -- format_sections ---------------------------------------------------


def test_format_sections_with_no_chunks():
    assert format_sections({}, overall_stats([])) == ["per-section: no chunks yet"]


def test_format_sections_includes_every_section_and_a_total_row():
    summaries = [
        summary(AWS + "sql/a", "aws-en", 100),
        summary(API + "jobs/run", "api", 500),
    ]
    sections = section_stats(summaries, SOURCES)
    lines = format_sections(sections, overall_stats(summaries))
    joined = "\n".join(lines)
    assert "aws-en/sql" in joined
    assert "api/jobs" in joined
    assert "total" in joined


def test_format_sections_total_row_uses_the_true_overall_stats_not_a_sum():
    summaries = [
        summary(AWS + "sql/a", "aws-en", 50),
        summary(AWS + "sql/b", "aws-en", 150),
        summary(AWS + "pyspark/a", "aws-en", 10_000),
    ]
    sections = section_stats(summaries, SOURCES)
    overall = overall_stats(summaries)
    lines = format_sections(sections, overall)
    total_line = next(line for line in lines if line.strip().startswith("total"))
    assert str(overall.avg_chars) in total_line
    assert str(overall.median_chars) in total_line
    # A naive sum of the two sections' own avg_chars would NOT equal the
    # true overall average -- confirm the total row doesn't show that.
    naive_sum = sum(s.avg_chars for s in sections.values())
    assert naive_sum != overall.avg_chars
    assert str(naive_sum) not in total_line.split()
