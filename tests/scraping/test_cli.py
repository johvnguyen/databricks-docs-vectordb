import time
from pathlib import Path

import pytest
import respx

from databricks_docs_vectordb.cli import main
from databricks_docs_vectordb.scraping.manifest import Manifest

FIXTURES = Path(__file__).parent.parent / "fixtures"
ROBOTS = "https://docs.databricks.com/robots.txt"
AWS_SITEMAP = "https://docs.databricks.com/aws/en/sitemap.xml"
API_SITEMAP = "https://docs.databricks.com/api/sitemap.xml"
CONTACT = "me@example.com"


@pytest.fixture
def env(monkeypatch, tmp_path):
    """No real .env, no real sleeping, data in a temp dir."""
    monkeypatch.chdir(tmp_path)  # so find_dotenv() can't see the repo's .env
    monkeypatch.setenv("DATABRICKS_DOCS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DATABRICKS_DOCS_CONTACT", CONTACT)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return tmp_path / "data"


@pytest.fixture
def site():
    with respx.mock(assert_all_called=False) as router:
        router.get(ROBOTS).respond(200, content=(FIXTURES / "robots.txt").read_bytes())
        router.get(AWS_SITEMAP).respond(
            200, content=(FIXTURES / "sitemap-aws-en.xml").read_bytes()
        )
        router.get(API_SITEMAP).respond(
            200, content=(FIXTURES / "sitemap-api.xml").read_bytes()
        )
        yield router


def test_scrape_enumerate_prints_a_summary_and_fills_the_manifest(env, site, capsys):
    assert main(["scrape", "enumerate"]) == 0

    out = capsys.readouterr().out
    assert "aws-en: 26 pending" in out
    assert "api: 6 pending" in out
    assert CONTACT not in out
    assert (env / "robots.txt").read_bytes() == (FIXTURES / "robots.txt").read_bytes()
    assert len(list((env / "sitemaps").glob("*.xml"))) == 2
    with Manifest(env / "manifest.sqlite") as m:
        assert m.counts()["pending"] == 32


def test_running_enumerate_twice_reports_nothing_new(env, site, capsys):
    main(["scrape", "enumerate"])
    capsys.readouterr()
    assert main(["scrape", "enumerate"]) == 0
    out = capsys.readouterr().out
    assert out.count("0 new, 0 revived, 0 removed") == 2


def test_requests_send_the_contact_in_the_user_agent(env, site):
    main(["scrape", "enumerate"])
    agents = {call.request.headers["User-Agent"] for call in site.calls}
    assert len(agents) == 1
    (agent,) = agents
    assert agent.startswith("databricks-docs-vectordb/")
    assert agent.endswith(f"(+{CONTACT})")


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_missing_contact_exits_before_any_request(
    env, site, monkeypatch, capsys, value
):
    if value is None:
        monkeypatch.delenv("DATABRICKS_DOCS_CONTACT")
    else:
        monkeypatch.setenv("DATABRICKS_DOCS_CONTACT", value)

    assert main(["scrape", "enumerate"]) == 1

    assert "DATABRICKS_DOCS_CONTACT" in capsys.readouterr().err
    assert len(site.calls) == 0
    assert not (env / "manifest.sqlite").exists()


def test_a_failed_fetch_exits_nonzero_with_the_reason_on_stderr(env, site, capsys):
    site.get(ROBOTS).respond(503)
    assert main(["scrape", "enumerate"]) == 1
    captured = capsys.readouterr()
    assert "robots.txt returned HTTP 503" in captured.err
    assert CONTACT not in captured.err
    assert captured.out == ""


def test_an_unknown_command_is_an_argparse_error(env, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["scrape", "bogus"])
    assert excinfo.value.code == 2


def test_a_bare_scrape_is_an_argparse_error(env, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["scrape"])
    assert excinfo.value.code == 2
