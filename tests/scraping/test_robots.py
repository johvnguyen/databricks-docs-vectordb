import urllib.robotparser
from pathlib import Path

import httpx
import pytest

from databricks_docs_vectordb.scraping.config import Settings
from databricks_docs_vectordb.scraping.robots import ROBOTS_AGENT, RobotsRules
from databricks_docs_vectordb.scraping.urls import parse_sitemap

FIXTURES = Path(__file__).parent.parent / "fixtures"
ROBOTS_TXT = (FIXTURES / "robots.txt").read_text()
HOST = "https://docs.databricks.com"


@pytest.fixture(scope="module")
def rules() -> RobotsRules:
    return RobotsRules(ROBOTS_TXT)


@pytest.mark.parametrize(
    "path",
    [
        "/aws/en/",
        "/aws/en/delta/",
        "/aws/en/sql/language-manual/",
        "/aws/en/release-notes/runtime/16.2ml",
        "/api/",
        "/api/workspace/scopes",
        "/aws/ja/delta/",  # allowed by robots.txt, excluded by our source list
    ],
)
def test_allowed_urls(rules, path):
    assert rules.is_allowed(HOST + path)


@pytest.mark.parametrize(
    "path",
    [
        "/aws/en/archive/",
        "/aws/en/archive/admin-guide/create-workspace-aws",
        "/aws/en/search-for",
        "/aws/en/search-for/anything",
        "/aws/ja/archive/x",
        "/gcp/en/archive/x",
        "/gcp/pt/search-for",
    ],
)
def test_disallowed_urls(rules, path):
    assert not rules.is_allowed(HOST + path)


def test_search_query_rule_needs_a_path_without_a_longer_allow(rules):
    # `Disallow: *s=*` is shorter than `Allow: /aws/en/`, and longest match
    # wins, so it only takes effect where no longer Allow covers the path.
    # We never fetch query URLs anyway (url_to_rel_path rejects them).
    assert rules.is_allowed(HOST + "/aws/en/delta/?s=foo")
    assert not rules.is_allowed(HOST + "/api/workspace/scopes?s=foo")


def test_stdlib_robotparser_would_allow_archive_pages(rules):
    # Why we use protego: the stdlib parser takes the first matching rule, so
    # `Allow: /` at the top of this file makes everything look allowed.
    stdlib = urllib.robotparser.RobotFileParser()
    stdlib.parse(ROBOTS_TXT.splitlines())
    url = HOST + "/aws/en/archive/admin-guide/create-workspace-aws"
    assert stdlib.can_fetch(ROBOTS_AGENT, url)
    assert not rules.is_allowed(url)


def test_filtering_the_aws_en_fixture_removes_exactly_the_archive_urls(rules):
    urls = parse_sitemap((FIXTURES / "sitemap-aws-en.xml").read_text())
    disallowed = {u for u in urls if not rules.is_allowed(u)}
    assert disallowed == {u for u in urls if "/archive/" in u}
    assert len(disallowed) == 3
    assert len(urls) - len(disallowed) == 26


def test_api_fixture_urls_are_all_allowed(rules):
    urls = parse_sitemap((FIXTURES / "sitemap-api.xml").read_text())
    assert all(rules.is_allowed(u) for u in urls)


def test_sitemaps_and_crawl_delay_from_the_real_file(rules):
    assert len(rules.sitemaps) == 6
    assert "https://docs.databricks.com/aws/en/sitemap.xml" in rules.sitemaps
    assert rules.crawl_delay is None


def test_crawl_delay_is_reported_when_the_site_sets_one():
    text = "User-agent: *\nCrawl-delay: 5\nAllow: /\n"
    assert RobotsRules(text).crawl_delay == 5


def test_rules_for_our_agent_apply_only_to_our_agent():
    text = (
        f"User-agent: {ROBOTS_AGENT}\nDisallow: /private/\n\n"
        "User-agent: *\nAllow: /\n"
    )
    url = HOST + "/private/page"
    assert not RobotsRules(text).is_allowed(url)
    assert RobotsRules(text, user_agent="someone-else").is_allowed(url)


def test_empty_robots_txt_allows_everything():
    assert RobotsRules("").is_allowed(HOST + "/aws/en/archive/x")


def test_fixture_is_stored_as_the_server_sent_it():
    raw = (FIXTURES / "robots.txt").read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert raw.startswith(b"User-agent: *")
    assert len(raw) == 1002


@pytest.mark.network
def test_fixture_matches_the_live_robots_txt():
    settings = Settings.load()
    response = httpx.get(
        f"{HOST}/robots.txt",
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
        timeout=30,
    )
    response.raise_for_status()
    assert (
        response.content == (FIXTURES / "robots.txt").read_bytes()
    ), "robots.txt changed upstream; review the diff, then refresh the fixture"


@pytest.mark.network
def test_live_robots_filter_removes_exactly_the_archive_urls():
    settings = Settings.load()
    with httpx.Client(
        headers={"User-Agent": settings.user_agent}, follow_redirects=True, timeout=30
    ) as client:
        live_rules = RobotsRules(
            client.get(f"{HOST}/robots.txt").raise_for_status().text
        )
        counts = {}
        for source in settings.sources:
            urls = parse_sitemap(
                client.get(source.sitemap_url).raise_for_status().content
            )
            disallowed = [u for u in urls if not live_rules.is_allowed(u)]
            archive = [u for u in urls if "/archive/" in u]
            assert disallowed == archive, source.name
            counts[source.name] = len(urls) - len(disallowed)
    print(f"allowed pages per source: {counts}")
    assert counts["api"] > 100
    assert counts["aws-en"] > 5_000
