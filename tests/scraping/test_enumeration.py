from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

from databricks_docs_vectordb.scraping.config import Settings, Source
from databricks_docs_vectordb.scraping.enumeration import (
    EnumerateError,
    enumerate_sources,
    format_report,
)
from databricks_docs_vectordb.scraping.manifest import Manifest

FIXTURES = Path(__file__).parent.parent / "fixtures"
ROBOTS = "https://docs.databricks.com/robots.txt"
AWS_SITEMAP = "https://docs.databricks.com/aws/en/sitemap.xml"
API_SITEMAP = "https://docs.databricks.com/api/sitemap.xml"
AWS = "https://docs.databricks.com/aws/en/"
API = "https://docs.databricks.com/api/"
CONTACT = "me@example.com"


def sitemap(*urls: str) -> bytes:
    body = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return (
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{body}</urlset>"
    ).encode()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(contact=CONTACT, data_dir=tmp_path / "data", rate=100.0)


@pytest.fixture
def manifest(settings):
    with Manifest(settings.manifest_path) as m:
        yield m


@pytest.fixture
def site():
    """The real robots.txt and trimmed real sitemaps, served by a mock."""
    with respx.mock(assert_all_called=False) as router:
        yield SimpleNamespace(
            router=router,
            robots=router.get(ROBOTS).respond(
                200, content=(FIXTURES / "robots.txt").read_bytes()
            ),
            aws=router.get(AWS_SITEMAP).respond(
                200, content=(FIXTURES / "sitemap-aws-en.xml").read_bytes()
            ),
            api=router.get(API_SITEMAP).respond(
                200, content=(FIXTURES / "sitemap-api.xml").read_bytes()
            ),
        )


@pytest.fixture
def client(settings):
    with httpx.Client(headers={"User-Agent": settings.user_agent}) as c:
        yield c


def run(settings, client, manifest, sleeps=None):
    return enumerate_sources(
        settings,
        client,
        manifest,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def files_under(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []


# -- happy path ----------------------------------------------------------


def test_first_run_adds_allowed_urls_and_skips_archive(
    settings, client, manifest, site
):
    result = run(settings, client, manifest)

    aws, api = result.sources
    assert (aws.source, aws.listed, aws.disallowed, aws.allowed) == (
        "aws-en",
        29,
        3,
        26,
    )
    assert (aws.added, aws.revived, aws.removed) == (26, 0, 0)
    assert (api.source, api.listed, api.disallowed, api.allowed) == ("api", 6, 0, 6)
    assert (api.added, api.revived, api.removed) == (6, 0, 0)
    assert manifest.counts_by_source() == {
        "aws-en": Counter(pending=26),
        "api": Counter(pending=6),
    }
    assert result.totals == manifest.counts_by_source()
    assert manifest.get(AWS + "archive/") is None
    assert manifest.get(AWS + "delta/").rel_path == "aws/en/delta/index.html"
    assert manifest.get(API).rel_path == "api/index.html"


def test_robots_txt_is_saved_verbatim(settings, client, manifest, site):
    result = run(settings, client, manifest)
    expected = (FIXTURES / "robots.txt").read_bytes()
    assert settings.robots_path.read_bytes() == expected
    assert result.robots_bytes == len(expected) == 1002


def test_each_sitemap_is_saved_verbatim_under_data_sitemaps(
    settings, client, manifest, site
):
    result = run(settings, client, manifest)
    for source_result, fixture in zip(
        result.sources, ("sitemap-aws-en.xml", "sitemap-api.xml"), strict=True
    ):
        copy = source_result.sitemap_copy
        assert copy.parent == settings.sitemaps_dir
        assert copy.name.startswith(f"{source_result.source}-")
        assert copy.suffix == ".xml"
        assert copy.read_bytes() == (FIXTURES / fixture).read_bytes()


def test_running_again_reports_nothing_new_or_removed(settings, client, manifest, site):
    run(settings, client, manifest)
    before = manifest.counts_by_source()

    again = run(settings, client, manifest)

    assert [(r.added, r.revived, r.removed) for r in again.sources] == [(0, 0, 0)] * 2
    assert manifest.counts_by_source() == before


def test_a_rerun_keeps_fetch_progress(settings, client, manifest, site):
    run(settings, client, manifest)
    manifest.record_ok(
        AWS + "delta/", http_status=200, sha256="ab" * 32, size_bytes=1, etag="e"
    )
    run(settings, client, manifest)
    page = manifest.get(AWS + "delta/")
    assert (page.status, page.etag) == ("ok", "e")


def test_urls_that_leave_the_sitemap_are_removed_and_can_come_back(
    settings, client, manifest, site
):
    run(settings, client, manifest)
    dropped = AWS + "delta/merge"
    xml = (
        (FIXTURES / "sitemap-aws-en.xml")
        .read_text()
        .replace(
            f"<url><loc>{dropped}</loc><changefreq>weekly</changefreq><priority>0.5</priority></url>",
            "",
        )
    )
    assert dropped not in xml
    site.aws.respond(200, content=xml.encode())

    gone = run(settings, client, manifest)
    assert (gone.sources[0].removed, gone.sources[0].added) == (1, 0)
    assert manifest.get(dropped).status == "removed"
    assert manifest.get(API).status == "pending"  # other source untouched

    site.aws.respond(200, content=(FIXTURES / "sitemap-aws-en.xml").read_bytes())
    back = run(settings, client, manifest)
    assert (back.sources[0].revived, back.sources[0].removed) == (1, 0)
    assert manifest.get(dropped).status == "pending"


# -- filtering -----------------------------------------------------------


def test_urls_outside_the_source_or_invalid_are_skipped_not_fatal(
    settings, client, manifest, site
):
    site.aws.respond(
        200,
        content=sitemap(
            AWS + "delta/",
            "https://docs.databricks.com/gcp/en/delta/",  # another cloud
            AWS + "search?q=x",  # query strings can't be stored
            AWS + "archive/old",  # disallowed by robots.txt
        ),
    )
    result = run(settings, client, manifest)
    aws = result.sources[0]
    assert aws.outside_source == ["https://docs.databricks.com/gcp/en/delta/"]
    assert aws.invalid == [AWS + "search?q=x"]
    assert aws.disallowed == 1
    assert aws.allowed == aws.added == 1
    assert manifest.counts_by_source()["aws-en"] == Counter(pending=1)


def test_the_source_prefix_keeps_one_sitemaps_urls_out_of_the_other(
    settings, client, manifest, site
):
    site.api.respond(200, content=sitemap(API, AWS + "delta/"))
    result = run(settings, client, manifest)
    assert result.sources[1].outside_source == [AWS + "delta/"]
    assert manifest.get(AWS + "delta/").source == "aws-en"


# -- politeness ----------------------------------------------------------


def test_every_request_carries_our_user_agent(settings, client, manifest, site):
    run(settings, client, manifest)
    assert len(site.router.calls) == 3
    for call in site.router.calls:
        assert call.request.headers["User-Agent"] == settings.user_agent


def test_robots_txt_is_fetched_once_even_with_two_sources(
    settings, client, manifest, site
):
    run(settings, client, manifest)
    assert site.robots.call_count == 1
    assert site.aws.call_count == site.api.call_count == 1


def test_it_pauses_between_requests(client, manifest, site, tmp_path):
    settings = Settings(contact=CONTACT, data_dir=tmp_path / "data", rate=4.0)
    sleeps: list[float] = []
    run(settings, client, manifest, sleeps)
    assert sleeps == [0.25, 0.25]  # 3 requests, 2 gaps of 1/rate


# -- failures leave everything untouched --------------------------------


def assert_nothing_changed(settings: Settings, manifest: Manifest) -> None:
    assert manifest.counts() == Counter()
    # only the manifest database (and its WAL sidecars) may exist
    assert all(
        p.name.startswith("manifest.sqlite") for p in files_under(settings.data_dir)
    )


@pytest.mark.parametrize("status", [404, 500, 503])
def test_robots_txt_http_errors_abort(settings, client, manifest, site, status):
    site.robots.respond(status)
    with pytest.raises(EnumerateError, match=f"robots.txt returned HTTP {status}"):
        run(settings, client, manifest)
    assert_nothing_changed(settings, manifest)
    assert site.aws.call_count == 0


@pytest.mark.parametrize(
    "body",
    [b"", b"<html><body>Not found</body></html>", b"Sitemap: https://x/\n"],
    ids=["empty", "html-error-page", "no-user-agent-line"],
)
def test_a_200_that_is_not_a_robots_file_aborts(settings, client, manifest, site, body):
    # Parsed as "allow everything", it would silently disable the archive filter.
    site.robots.respond(200, content=body)
    with pytest.raises(EnumerateError, match="no User-agent line"):
        run(settings, client, manifest)
    assert_nothing_changed(settings, manifest)


def test_a_failing_second_sitemap_leaves_the_first_unapplied(
    settings, client, manifest, site
):
    site.api.respond(500)
    with pytest.raises(EnumerateError, match="sitemap.xml returned HTTP 500"):
        run(settings, client, manifest)
    assert_nothing_changed(settings, manifest)
    assert not settings.robots_path.exists()


@pytest.mark.parametrize(
    "body",
    [
        b"<html><body>Not found</body></html>",
        b"<urlset>",
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>',
    ],
    ids=["html", "malformed", "sitemap-index"],
)
def test_a_bad_sitemap_names_the_source_and_aborts(
    settings, client, manifest, site, body
):
    site.aws.respond(200, content=body)
    with pytest.raises(EnumerateError, match="aws-en: bad sitemap"):
        run(settings, client, manifest)
    assert_nothing_changed(settings, manifest)


def test_an_empty_sitemap_aborts_and_removes_nothing(settings, client, manifest, site):
    run(settings, client, manifest)
    before = manifest.counts()
    site.aws.respond(200, content=sitemap())
    with pytest.raises(EnumerateError, match="aws-en: .*lists no URLs"):
        run(settings, client, manifest)
    assert manifest.counts() == before


def test_a_sitemap_with_no_usable_urls_aborts(settings, client, manifest, site):
    site.aws.respond(200, content=sitemap(AWS + "archive/a", AWS + "archive/b"))
    with pytest.raises(EnumerateError, match="none of the 2 sitemap URLs are usable"):
        run(settings, client, manifest)
    assert_nothing_changed(settings, manifest)


def test_network_errors_become_enumerate_errors(settings, client, manifest, site):
    site.robots.mock(side_effect=httpx.ConnectTimeout("timed out"))
    with pytest.raises(EnumerateError, match="could not fetch .*robots.txt"):
        run(settings, client, manifest)
    assert_nothing_changed(settings, manifest)


def test_sources_on_different_hosts_are_refused(client, manifest, tmp_path):
    settings = Settings(
        contact=CONTACT,
        data_dir=tmp_path / "data",
        sources=(
            Source("a", "https://one.example/sitemap.xml", "https://one.example/"),
            Source("b", "https://two.example/sitemap.xml", "https://two.example/"),
        ),
    )
    with pytest.raises(EnumerateError, match="several hosts"):
        run(settings, client, manifest)


def test_a_failed_run_does_not_overwrite_the_previous_copies(
    settings, client, manifest, site
):
    run(settings, client, manifest)
    good = settings.robots_path.read_bytes()
    site.robots.respond(200, content=b"garbage")
    with pytest.raises(EnumerateError):
        run(settings, client, manifest)
    assert settings.robots_path.read_bytes() == good


def test_saved_copies_are_ordinary_readable_files(settings, client, manifest, site):
    run(settings, client, manifest)
    mode = settings.robots_path.stat().st_mode & 0o777
    assert mode & 0o600 == 0o600  # owner can read and write
    assert mode & 0o044 == 0o044  # group and others can read


def test_no_temp_files_are_left_behind(settings, client, manifest, site):
    run(settings, client, manifest)
    leftovers = [p for p in files_under(settings.data_dir) if p.name.startswith(".")]
    assert leftovers == []


# -- report --------------------------------------------------------------


def test_report_summarises_each_source_and_the_manifest(
    settings, client, manifest, site
):
    text = format_report(run(settings, client, manifest))
    assert "robots.txt: 1002 bytes saved to" in text
    assert "aws-en: 29 URLs in sitemap; skipped 3 by robots.txt" in text
    assert "26 new, 0 revived, 0 removed" in text
    assert "  aws-en: 26 pending" in text
    assert "  api: 6 pending" in text
    assert CONTACT not in text


def test_report_lists_a_few_skipped_urls(settings, client, manifest, site):
    weird = [AWS + f"q{i}?x=1" for i in range(7)]
    site.aws.respond(200, content=sitemap(AWS + "delta/", *weird))
    text = format_report(run(settings, client, manifest))
    assert f"skipped (invalid): {weird[0]}" in text
    assert "... and 2 more (invalid)" in text
