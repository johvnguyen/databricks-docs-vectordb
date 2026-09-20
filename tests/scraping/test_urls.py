from collections import Counter
from pathlib import Path

import httpx
import pytest

from databricks_docs_vectordb.scraping.config import SOURCES, Settings
from databricks_docs_vectordb.scraping.urls import parse_sitemap, url_to_rel_path

FIXTURES = Path(__file__).parent.parent / "fixtures"
AWS = "https://docs.databricks.com/aws/en/"
API = "https://docs.databricks.com/api/"


def test_parse_sitemap_returns_every_fixture_url_in_order():
    urls = parse_sitemap((FIXTURES / "sitemap-aws-en.xml").read_text())
    assert len(urls) == 29
    assert urls[0] == AWS
    assert urls[1] == AWS + "admin/account-settings/"
    assert AWS + "release-notes/runtime/13.2ml" in urls


def test_parse_sitemap_accepts_bytes():
    urls = parse_sitemap((FIXTURES / "sitemap-api.xml").read_bytes())
    assert len(urls) == 6
    assert urls[0] == API


def test_parse_sitemap_drops_duplicates_and_blank_locs():
    xml = (
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{AWS}a</loc></url>"
        f"<url><loc> {AWS}b </loc></url>"
        f"<url><loc>{AWS}a</loc></url>"
        "<url><loc>  </loc></url>"
        "<url></url>"
        "</urlset>"
    )
    assert parse_sitemap(xml) == [AWS + "a", AWS + "b"]


def test_parse_sitemap_empty_urlset_is_empty_list():
    xml = '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>'
    assert parse_sitemap(xml) == []


def test_parse_sitemap_rejects_invalid_xml():
    with pytest.raises(ValueError, match="not valid XML"):
        parse_sitemap("<urlset>")


@pytest.mark.parametrize(
    "xml",
    [
        # a sitemap index, not a page sitemap
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>',
        # an HTML error page, or a urlset without the sitemap namespace
        "<html><body>Not found</body></html>",
        "<urlset><url><loc>https://docs.databricks.com/aws/en/a</loc></url></urlset>",
    ],
)
def test_parse_sitemap_rejects_non_urlset_documents(xml):
    with pytest.raises(ValueError, match="urlset"):
        parse_sitemap(xml)


@pytest.mark.parametrize(
    ("url", "rel_path"),
    [
        (AWS + "delta/", "aws/en/delta/index.html"),
        (AWS + "delta/merge", "aws/en/delta/merge.html"),
        (
            AWS + "release-notes/runtime/16.2ml",
            "aws/en/release-notes/runtime/16.2ml.html",
        ),
        (AWS + "release-notes/runtime/13.2", "aws/en/release-notes/runtime/13.2.html"),
        (AWS, "aws/en/index.html"),
        (API, "api/index.html"),
        (
            API + "workspace/scopes",
            "api/workspace/scopes.html",
        ),
        (
            AWS + "sql/language-manual/functions/current_time",
            "aws/en/sql/language-manual/functions/current_time.html",
        ),
    ],
)
def test_url_to_rel_path(url, rel_path):
    assert url_to_rel_path(url) == rel_path


def test_dir_and_leaf_with_same_name_do_not_collide():
    assert url_to_rel_path(AWS + "delta/") != url_to_rel_path(AWS + "delta")


@pytest.mark.parametrize(
    "url",
    [
        "https://docs.databricks.com/gcp/en/delta/",  # another cloud
        "https://docs.databricks.com/aws/ja/delta/",  # another language
        "https://docs.databricks.com/aws/en",  # the prefix without its slash
        "https://docs.databricks.com/aws/enx/delta/",  # near-miss prefix
        "https://example.com/aws/en/delta/",  # another host
        "http://docs.databricks.com/aws/en/delta/",  # not https
        "https://docs.databricks.com/",
        "",
    ],
)
def test_url_outside_every_source_is_rejected(url):
    with pytest.raises(ValueError, match="outside every configured source"):
        url_to_rel_path(url)


@pytest.mark.parametrize(
    "url",
    [
        AWS + "../../etc/passwd",
        AWS + "delta/../../x",
        AWS + "./delta",
        AWS + "delta//merge",
        AWS + "delta/..",
        AWS + "..",
        API + "../aws/en/",
        AWS + "delta?s=foo",
        AWS + "delta?",
        AWS + "delta#section",
        AWS + "delta#",
        AWS + "%2e%2e/x",
        AWS + "delta%2Fmerge",
        AWS + "delta\\merge",
        AWS + "spaced page",
        AWS + "café",
        AWS + "a:b",
    ],
)
def test_unsafe_or_ambiguous_urls_are_rejected(url):
    with pytest.raises(ValueError):
        url_to_rel_path(url)


def test_sources_argument_restricts_what_is_accepted():
    only_api = [s for s in SOURCES if s.name == "api"]
    assert url_to_rel_path(API, only_api) == "api/index.html"
    with pytest.raises(ValueError, match="outside every configured source"):
        url_to_rel_path(AWS + "delta/", only_api)


def test_fixture_urls_map_to_unique_paths_and_stay_relative():
    urls = parse_sitemap((FIXTURES / "sitemap-aws-en.xml").read_text()) + parse_sitemap(
        (FIXTURES / "sitemap-api.xml").read_text()
    )
    paths = [url_to_rel_path(u) for u in urls]
    assert len(set(paths)) == len(paths)
    for path in paths:
        assert not path.startswith("/")
        assert ".." not in Path(path).parts
        assert path.endswith(".html")


@pytest.mark.network
def test_live_sitemaps_parse_and_map_without_collisions():
    settings = Settings.load()
    all_urls: list[str] = []
    with httpx.Client(
        headers={"User-Agent": settings.user_agent}, follow_redirects=True, timeout=30
    ) as client:
        for source in settings.sources:
            response = client.get(source.sitemap_url)
            response.raise_for_status()
            urls = parse_sitemap(response.content)
            assert len(urls) > 100, f"{source.name} sitemap looks truncated"
            all_urls += urls

    paths = [url_to_rel_path(u, settings.sources) for u in all_urls]
    collisions = [p for p, n in Counter(paths).items() if n > 1]
    assert collisions == []
    # ~5,933 for aws-en (including archive) and ~217 for api at time of writing
    assert 5_000 < len(all_urls) < 8_000
