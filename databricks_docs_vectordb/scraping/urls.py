import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from urllib.parse import urlsplit

from databricks_docs_vectordb.scraping.config import SOURCES, Source

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"

# Every sitemap URL uses only these characters. Anything else (percent-escapes,
# backslashes, colons, ...) is rejected rather than mapped to a file path.
_SAFE_PATH = re.compile(r"[A-Za-z0-9/_.\-]*")


def parse_sitemap(xml: str | bytes) -> list[str]:
    """Return the `<loc>` URLs of a sitemap `<urlset>`, in order, without duplicates."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ValueError(f"sitemap is not valid XML: {e}") from e
    if root.tag != f"{{{SITEMAP_NS}}}urlset":
        raise ValueError(f"expected a sitemap <urlset>, got <{root.tag}>")

    locs = (
        loc.text for loc in root.iterfind(f"{{{SITEMAP_NS}}}url/{{{SITEMAP_NS}}}loc")
    )
    return list(dict.fromkeys(text.strip() for text in locs if text and text.strip()))


def url_to_rel_path(url: str, sources: Iterable[Source] = SOURCES) -> str:
    """Map a page URL to its file path relative to `data/raw/`.

    A URL ending in `/` becomes `<path>index.html`. Any other URL gets `.html`
    appended, never replacing an existing dotted suffix, so `.../16.2ml`
    becomes `.../16.2ml.html`.

    Raises ValueError for URLs outside every source's prefix, and for anything
    that could escape `data/raw/` or map ambiguously (query strings, fragments,
    `.`/`..`/empty path segments, characters outside `[A-Za-z0-9/_.-]`).
    """
    if not any(url.startswith(source.url_prefix) for source in sources):
        raise ValueError(f"URL is outside every configured source: {url!r}")

    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.query
        or parts.fragment
        or "?" in url
        or "#" in url
    ):
        raise ValueError(
            f"unsupported URL (need plain https, no query or fragment): {url!r}"
        )

    path = parts.path.removeprefix("/")
    if not _SAFE_PATH.fullmatch(path):
        raise ValueError(f"URL path has unsupported characters: {url!r}")

    segments = path.split("/")
    if path.endswith("/"):
        segments.pop()  # the empty segment after the trailing slash
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError(f"URL path has an empty, '.' or '..' segment: {url!r}")

    return f"{path}index.html" if path.endswith("/") else f"{path}.html"
