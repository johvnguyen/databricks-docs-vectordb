"""Extract a page's metadata: `<title>`, meta description, the JSON-LD
`BreadcrumbList`, and the rendered "Last updated" date -- combined with the
identifiers the Phase 1 manifest already knows (`url`, `source`, `sha256`,
`fetched_at`) and the cloud the source belongs to.

This is a separate concern from `html_walker.py`: everything here lives in
`<head>`, or in `<article>` but outside `div.theme-doc-markdown.markdown`
(the breadcrumbs `<nav>`, the "Last updated" row) -- it never touches the
content node tree.
"""

import json
from dataclasses import dataclass

from bs4 import BeautifulSoup

# Every current source is AWS-only (see plan.md's scope decision: "AWS,
# English, current version only"). When a second cloud is scraped, this
# becomes a real per-source lookup -- see parsing.md's "GCP/Azure cloud
# add" revisit trigger.
SOURCE_CLOUDS: dict[str, str] = {
    "aws-en": "aws",
    "api": "aws",
}


class MetadataError(ValueError):
    """The page is missing something metadata.py can't do without."""


@dataclass(frozen=True)
class PageMetadata:
    url: str
    source: str
    cloud: str
    doc_title: str
    description: str | None
    section_path: str
    last_updated: str | None
    page_sha256: str
    fetched_at: str | None


def cloud_for_source(source: str) -> str:
    try:
        return SOURCE_CLOUDS[source]
    except KeyError:
        raise MetadataError(
            f"unknown source {source!r}; no cloud configured for it"
        ) from None


def extract(
    html: str, *, url: str, source: str, sha256: str, fetched_at: str | None
) -> PageMetadata:
    """Parse one raw page's metadata, combined with what the manifest knows.

    `url`, `source`, `sha256`, and `fetched_at` come from the Phase 1
    manifest row for this page; `cloud` is derived from `source`.
    """
    soup = BeautifulSoup(html, "lxml")
    return PageMetadata(
        url=url,
        source=source,
        cloud=cloud_for_source(source),
        doc_title=_title(soup),
        description=_description(soup),
        section_path=_section_path(soup),
        last_updated=_last_updated(soup),
        page_sha256=sha256,
        fetched_at=fetched_at,
    )


def _title(soup: BeautifulSoup) -> str:
    tag = soup.find("title")
    text = tag.get_text() if tag is not None else ""
    if not text.strip():
        raise MetadataError("no <title> element found")
    # Docusaurus appends a site-name suffix that differs by source (e.g.
    # " | Databricks on AWS" on aws-en, " | Databricks API Reference" on
    # api) -- strip whichever one is there rather than hardcoding one.
    title, separator, _suffix = text.rpartition(" | ")
    return title if separator else text


def _description(soup: BeautifulSoup) -> str | None:
    for attrs in ({"name": "description"}, {"property": "og:description"}):
        tag = soup.find("meta", attrs=attrs)
        if tag is None:
            continue
        content = (tag.get("content") or "").strip()
        if content:
            return content
    return None


def _section_path(soup: BeautifulSoup) -> str:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        if not script.string:
            continue
        try:
            data = json.loads(script.string)
        except json.JSONDecodeError:
            continue
        if data.get("@type") != "BreadcrumbList":
            continue
        items = sorted(
            data.get("itemListElement", []), key=lambda item: item.get("position", 0)
        )
        names = [item.get("name", "") for item in items]
        return " > ".join(name for name in names if name)
    return ""


def _last_updated(soup: BeautifulSoup) -> str | None:
    tag = soup.find("time", attrs={"itemprop": "dateModified"})
    if tag is None:
        return None
    value = (tag.get("datetime") or "").strip()
    return value or None
