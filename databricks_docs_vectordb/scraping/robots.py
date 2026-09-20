from protego import Protego

ROBOTS_AGENT = "databricks-docs-vectordb"


class RobotsRules:
    """robots.txt rules for one user agent.

    Uses `protego` rather than `urllib.robotparser`: the stdlib parser applies
    the first matching rule and has no wildcard support, so with Databricks'
    robots.txt (`Allow: /` comes first) it allows everything, including the
    disallowed `/archive/` pages. `protego` implements longest-match with
    wildcards.
    """

    def __init__(self, robots_txt: str, user_agent: str = ROBOTS_AGENT) -> None:
        self._rules = Protego.parse(robots_txt)
        self._user_agent = user_agent

    def is_allowed(self, url: str) -> bool:
        return self._rules.can_fetch(url, self._user_agent)

    @property
    def crawl_delay(self) -> float | None:
        """Seconds the site asks us to wait between requests, if it says so."""
        return self._rules.crawl_delay(self._user_agent)

    @property
    def sitemaps(self) -> list[str]:
        return list(self._rules.sitemaps)
