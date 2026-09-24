from pathlib import Path

import pytest

from databricks_docs_vectordb.parsing.metadata import (
    MetadataError,
    PageMetadata,
    cloud_for_source,
    extract,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "parsing"

URL = "https://docs.databricks.com/aws/en/delta/"
SHA256 = "ab" * 32
FETCHED_AT = "2026-09-11T00:00:00Z"


def load(*parts: str) -> str:
    return (FIXTURES.joinpath(*parts)).read_text(encoding="utf-8")


def extract_fixture(source: str, *parts: str) -> PageMetadata:
    return extract(
        load(*parts),
        url=URL,
        source=source,
        sha256=SHA256,
        fetched_at=FETCHED_AT,
    )


def wrap(head_extra: str = "", article_extra: str = "") -> str:
    """A minimal page shell, for edge cases not worth a whole fixture file."""
    return (
        "<html><head>"
        "<title>A Page | Databricks on AWS</title>"
        f"{head_extra}"
        "</head><body><article>"
        '<div class="theme-doc-markdown markdown"><header><h1>A Page</h1></header>'
        f"{article_extra}"
        "</div></article></body></html>"
    )


# -- manifest fields pass through unchanged ----------------------------------


def test_manifest_fields_pass_through_unchanged():
    meta = extract(
        wrap(),
        url="https://docs.databricks.com/aws/en/x/",
        source="aws-en",
        sha256="deadbeef" * 8,
        fetched_at="2026-01-01T00:00:00Z",
    )
    assert meta.url == "https://docs.databricks.com/aws/en/x/"
    assert meta.source == "aws-en"
    assert meta.page_sha256 == "deadbeef" * 8
    assert meta.fetched_at == "2026-01-01T00:00:00Z"


def test_fetched_at_can_be_none():
    meta = extract(wrap(), url=URL, source="aws-en", sha256=SHA256, fetched_at=None)
    assert meta.fetched_at is None


# -- cloud, from the source's config -----------------------------------------


def test_cloud_for_known_sources_is_aws():
    assert cloud_for_source("aws-en") == "aws"
    assert cloud_for_source("api") == "aws"


def test_cloud_for_an_unknown_source_raises():
    with pytest.raises(MetadataError, match="unknown source"):
        cloud_for_source("gcp-en")


def test_extract_raises_for_an_unknown_source():
    with pytest.raises(MetadataError, match="unknown source"):
        extract(wrap(), url=URL, source="gcp-en", sha256=SHA256, fetched_at=FETCHED_AT)


# -- title -----------------------------------------------------------------


def test_title_strips_the_aws_site_name_suffix():
    meta = extract(
        wrap(), url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT
    )
    assert meta.doc_title == "A Page"


def test_title_strips_the_api_site_name_suffix():
    html = wrap().replace(
        "<title>A Page | Databricks on AWS</title>",
        "<title>Run | Databricks API Reference</title>",
    )
    meta = extract(html, url=URL, source="api", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.doc_title == "Run"


def test_title_with_no_suffix_is_kept_as_is():
    html = wrap().replace(
        "<title>A Page | Databricks on AWS</title>", "<title>Just A Title</title>"
    )
    meta = extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.doc_title == "Just A Title"


def test_missing_title_raises():
    html = wrap().replace("<title>A Page | Databricks on AWS</title>", "")
    with pytest.raises(MetadataError, match="title"):
        extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)


def test_empty_title_raises():
    html = wrap().replace(
        "<title>A Page | Databricks on AWS</title>", "<title></title>"
    )
    with pytest.raises(MetadataError, match="title"):
        extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)


# -- description: present, missing, and og: fallback --------------------


def test_a_page_missing_a_description_yields_none_not_an_error():
    meta = extract(
        wrap(), url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT
    )
    assert meta.description is None


def test_description_is_read_from_meta_name_description():
    html = wrap(head_extra='<meta name="description" content="Hello there.">')
    meta = extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.description == "Hello there."


def test_description_falls_back_to_og_description():
    html = wrap(head_extra='<meta property="og:description" content="OG summary.">')
    meta = extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.description == "OG summary."


def test_name_description_wins_over_og_description():
    html = wrap(
        head_extra=(
            '<meta name="description" content="Primary.">'
            '<meta property="og:description" content="Secondary.">'
        )
    )
    meta = extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.description == "Primary."


def test_a_blank_description_is_treated_as_missing():
    html = wrap(head_extra='<meta name="description" content="   ">')
    meta = extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.description is None


# -- section_path, from the JSON-LD BreadcrumbList ---------------------------


def test_a_page_with_no_breadcrumb_json_ld_gets_an_empty_section_path():
    meta = extract(
        wrap(), url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT
    )
    assert meta.section_path == ""


def test_breadcrumb_items_are_joined_in_position_order_even_if_listed_out_of_order():
    html = wrap(
        head_extra="""
        <script type="application/ld+json">
        {"@type": "BreadcrumbList", "itemListElement": [
            {"@type": "ListItem", "position": 2, "name": "Delta Lake", "item": "x"},
            {"@type": "ListItem", "position": 1, "name": "Tables", "item": "y"}
        ]}
        </script>
        """
    )
    meta = extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.section_path == "Tables > Delta Lake"


def test_a_non_breadcrumb_json_ld_script_is_ignored():
    html = wrap(
        head_extra="""
        <script type="application/ld+json">{"@type": "WebPage", "name": "irrelevant"}</script>
        """
    )
    meta = extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.section_path == ""


def test_malformed_json_ld_does_not_raise():
    html = wrap(
        head_extra='<script type="application/ld+json">{not valid json</script>'
    )
    meta = extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.section_path == ""


# -- last_updated, from the rendered "Last updated" row ----------------------


def test_a_page_with_no_last_updated_row_gets_none():
    meta = extract(
        wrap(), url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT
    )
    assert meta.last_updated is None


def test_last_updated_reads_the_time_elements_datetime_attribute():
    html = wrap(
        article_extra=(
            '<span class="theme-last-updated">Last updated on '
            '<b><time datetime="2026-09-11T00:00:00.000Z" itemprop="dateModified">'
            "Sep 11, 2026</time></b></span>"
        )
    )
    meta = extract(html, url=URL, source="aws-en", sha256=SHA256, fetched_at=FETCHED_AT)
    assert meta.last_updated == "2026-09-11T00:00:00.000Z"


# -- real fixtures: exact expected values -----------------------------------


def test_delta_fixture_exact_metadata():
    meta = extract_fixture("aws-en", "delta", "index.html")
    assert meta.doc_title == "What is Delta Lake in Databricks?"
    assert meta.description == (
        "Delta Lake is the default open-source storage format for all "
        "Databricks tables, providing ACID transactions, scalable metadata "
        "handling, and unified support for batch and streaming workloads."
    )
    assert meta.section_path == "Tables > Delta Lake"
    assert meta.last_updated == "2026-09-11T00:00:00.000Z"
    assert meta.cloud == "aws"


def test_oltp_fixture_exact_metadata():
    meta = extract_fixture("aws-en", "oltp", "update-to-autoscaling-terraform.html")
    assert meta.doc_title == (
        "Update your Terraform configuration to use Autoscaling resources"
    )
    assert meta.description == (
        "Update an existing Terraform configuration from Lakebase "
        "Provisioned to Lakebase Autoscaling using import blocks. Covers "
        "database instances, catalogs, and synced tables."
    )
    assert meta.section_path == (
        "Lakebase Postgres > Provisioned > Upgrade to Autoscaling > Update Terraform"
    )
    assert meta.last_updated == "2026-09-11T00:00:00.000Z"


def test_api_fixture_exact_metadata():
    meta = extract_fixture("api", "api", "jobs", "v2", "run.html")
    assert meta.doc_title == "Run"
    assert meta.description == "Databricks API reference for Run."
    assert meta.section_path == "Run"
    assert meta.last_updated is None  # trimmed out of this fixture
    assert meta.cloud == "aws"


def test_error_classes_fixture_exact_metadata():
    meta = extract_fixture("aws-en", "error-messages", "error-classes.html")
    assert meta.doc_title == "Error conditions in Databricks"
    assert meta.description == (
        "Documentation for the error conditions used in Databricks"
    )
    assert meta.section_path == "Error messages > Error classes in Databricks"
    assert meta.last_updated is None  # trimmed out of this fixture


def test_tutorial_fixture_exact_metadata():
    meta = extract_fixture(
        "aws-en", "machine-learning", "sgc-yolo11n-detect-coco128.html"
    )
    assert meta.doc_title == "YOLO11n object detection on Databricks AI Runtime"
    assert meta.section_path == (
        "Machine learning > Train models > AI Runtime (Preview) > Examples "
        "> Computer vision > Object detection with YOLO11n"
    )
    assert meta.last_updated is None  # trimmed out of this fixture
