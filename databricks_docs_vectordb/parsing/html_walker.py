"""Walk a raw Databricks docs page into an ordered tree of content nodes.

Reads `div.theme-doc-markdown.markdown` inside `<article>` and emits one
node per block-level element: `Heading`, `Paragraph`, `ListBlock`,
`QuoteBlock`, `Table`, `CodeBlock`, `Admonition`, `TabGroup`, and (on `api/`
pages) `ApiAttr`. Everything else described in parsing.md's "Strip" rule --
breadcrumbs, the mobile TOC toggle, the "Last updated" row, hash-link
permalink glyphs, and Docusaurus's `<!-- -->` i18n placeholder comments --
is dropped. The page's own `<h1>` is skipped too: that's metadata.py's job
(Step 5), not a content node.

Two DOM shapes turned up beyond what parsing.md's HTML survey called out,
both handled by unwrapping rather than a special node type: `api/`
*endpoint* pages (e.g. `jobs/v2/run.html`) wrap each section's content in
`div.apiLayout > div.apiMain` alongside a side panel (`aside.apiMethodsPanel`,
a duplicate example -- skipped, like the right-hand TOC); and a handful of
pages wrap block content in a plain, unclassed `<div>` for no semantic
reason -- unwrapped like `scrollable-table-container`.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


@dataclass(frozen=True)
class Heading:
    level: int  # 2 for <h2>, 3 for <h3>, etc.
    id: str | None
    text: str


@dataclass(frozen=True)
class Paragraph:
    text: str


@dataclass(frozen=True)
class ListItem:
    text: str
    children: tuple["ListItem", ...] = ()  # a nested <ul>/<ol> under this item


@dataclass(frozen=True)
class ListBlock:
    ordered: bool
    items: tuple[ListItem, ...]


@dataclass(frozen=True)
class QuoteBlock:
    text: str


@dataclass(frozen=True)
class Table:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class CodeBlock:
    language: str | None
    code: str


@dataclass(frozen=True)
class Admonition:
    type: str  # "tip", "note", "caution", ... -- read from the visible label
    children: tuple["Node", ...]


@dataclass(frozen=True)
class TabPanel:
    label: str
    children: tuple["Node", ...]


@dataclass(frozen=True)
class TabGroup:
    tabs: tuple[TabPanel, ...]


@dataclass(frozen=True)
class ApiAttr:
    name: str
    type: str | None
    facets: tuple[str, ...]
    description: str
    children: tuple["ApiAttr", ...] = ()  # nested object/array-of-object fields


Node = (
    Heading
    | Paragraph
    | ListBlock
    | QuoteBlock
    | Table
    | CodeBlock
    | Admonition
    | TabGroup
    | ApiAttr
)


class NoArticleError(ValueError):
    """The page has no `<article>` (or no markdown div inside it)."""


def walk(html: str) -> list[Node]:
    """Parse one raw page into its ordered content nodes."""
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article")
    if article is None:
        raise NoArticleError("no <article> element found")
    markdown_div = article.select_one("div.theme-doc-markdown.markdown")
    if markdown_div is None:
        raise NoArticleError("no div.theme-doc-markdown.markdown element found")
    return _walk_children(markdown_div)


# -- block-level dispatch --------------------------------------------------

_SKIP_TAGS = frozenset({"nav", "header", "script", "style"})
_SKIP_CLASS_MARKERS = ("theme-doc-toc-mobile", "theme-doc-footer-edit-meta-row")
_HEADING_TAGS = frozenset({"h2", "h3", "h4", "h5", "h6"})


def _walk_children(container: Tag) -> list[Node]:
    nodes: list[Node] = []
    for child in container.children:
        nodes.extend(_walk_one(child))
    return nodes


def _walk_one(el: object) -> list[Node]:
    if not isinstance(el, Tag):
        return []  # NavigableString (stray whitespace) or Comment

    name = el.name
    classes = _classes(el)

    if name in _SKIP_TAGS or _has_class(classes, *_SKIP_CLASS_MARKERS):
        return []
    if name in _HEADING_TAGS:
        return [_heading(el)]
    if name == "p":
        text = _inline_text(el)
        return [Paragraph(text)] if text else []
    if name in ("ul", "ol"):
        return [_list_block(el)]
    if name == "table":
        return [_table(el)]
    if name == "blockquote":
        text = _inline_text(el)
        return [QuoteBlock(text)] if text else []
    if _has_class(classes, "scrollable-table-container"):
        return _walk_children(el)  # a table wrapper, not content of its own
    if _has_class(classes, "theme-code-block"):
        return [_code_block(el)]
    if _has_class(classes, "theme-admonition"):
        return [_admonition(el)]
    if _has_class(classes, "tabs-container"):
        return [_tab_group(el)]
    if name == "dl" and _has_class(classes, "apiAttr"):
        return [_api_attr(el)]
    if _has_class(classes, "apiLayout"):
        api_main = el.select_one("div.apiMain") or el  # else skip the aside panel
        return _walk_children(api_main)
    if name == "div":
        # An unrecognized wrapper div (a tailwind-scoped widget, a layout
        # container, ...): recurse so nothing nested inside it is silently
        # dropped.
        return _walk_children(el)

    # A stray inline-only element at block level (rare): keep its text
    # rather than lose it.
    text = _inline_text(el)
    return [Paragraph(text)] if text else []


# -- individual node builders ------------------------------------------------


def _heading(el: Tag) -> Heading:
    return Heading(level=int(el.name[1]), id=el.get("id"), text=_inline_text(el))


def _list_block(el: Tag) -> ListBlock:
    items = tuple(_list_item(li) for li in el.find_all("li", recursive=False))
    return ListBlock(ordered=el.name == "ol", items=items)


def _list_item(li: Tag) -> ListItem:
    nested_lists = li.find_all(("ul", "ol"), recursive=False)
    own_text = _inline_text(
        li, skip=lambda child: isinstance(child, Tag) and child.name in ("ul", "ol")
    )
    children = tuple(
        _list_item(sub_li)
        for nested in nested_lists
        for sub_li in nested.find_all("li", recursive=False)
    )
    return ListItem(text=own_text, children=children)


def _table(el: Tag) -> Table:
    headers: tuple[str, ...] = ()
    thead = el.find("thead")
    if thead is not None:
        header_row = thead.find("tr")
        if header_row is not None:
            headers = tuple(
                _inline_text(cell)
                for cell in header_row.find_all(("th", "td"), recursive=False)
            )

    rows: list[tuple[str, ...]] = []
    tbodies = el.find_all("tbody") or [el]
    for tbody in tbodies:
        for tr in tbody.find_all("tr", recursive=False):
            rows.append(
                tuple(
                    _inline_text(cell)
                    for cell in tr.find_all(("td", "th"), recursive=False)
                )
            )
    return Table(headers=headers, rows=tuple(rows))


def _code_block(el: Tag) -> CodeBlock:
    pre = el.find("pre")
    language = None
    if pre is not None:
        for cls in _classes(pre):
            if cls.startswith("language-"):
                language = cls.removeprefix("language-")
                break
    return CodeBlock(language=language, code=_code_text(pre) if pre is not None else "")


def _code_text(pre: Tag) -> str:
    """Reassemble source text from Prism's per-line `<span class=token-line>`s.

    Each line is one span (its trailing `<br/>` contributes no text), so
    joining their text with "\\n" recovers the original code exactly --
    unlike a plain `.get_text()`, which would run every line together.
    """
    lines = pre.select("span.token-line")
    if lines:
        return "\n".join(line.get_text() for line in lines)
    return pre.get_text()


def _admonition(el: Tag) -> Admonition:
    heading = el.find(class_=_class_contains("admonitionHeading"))
    content = el.find(class_=_class_contains("admonitionContent"))
    admonition_type = _inline_text(heading) if heading is not None else ""
    children = tuple(_walk_children(content)) if content is not None else ()
    return Admonition(type=admonition_type, children=children)


def _tab_group(el: Tag) -> TabGroup:
    tablist = el.find(attrs={"role": "tablist"})
    labels = (
        [_inline_text(li) for li in tablist.find_all("li", recursive=False)]
        if tablist is not None
        else []
    )
    panels = el.find_all(attrs={"role": "tabpanel"})
    tabs = tuple(
        TabPanel(label=label, children=tuple(_walk_children(panel)))
        for label, panel in zip(labels, panels, strict=False)
    )
    return TabGroup(tabs=tabs)


def _api_attr(dl: Tag) -> ApiAttr:
    dt = dl.find("dt")
    name_el = dt.find(class_="apiAttr__name") if dt is not None else None
    type_el = dt.find(class_="apiAttr__type") if dt is not None else None
    facet_els = dt.find_all(class_="apiAttr__facet") if dt is not None else []

    dd = dl.find("dd")
    description = ""
    children: tuple[ApiAttr, ...] = ()
    if dd is not None:
        details = dd.find("details", recursive=False)
        description_parts = [
            _inline_text(child)
            for child in dd.find_all(recursive=False)
            if child is not details
        ]
        description = "\n\n".join(part for part in description_parts if part)
        if details is not None:
            children = tuple(
                _api_attr(nested)
                for nested in details.find_all("dl", class_="apiAttr")
                # only direct children of *this* details -- a nested dl's
                # own recursive call handles anything deeper than that.
                if nested.find_parent("details") is details
            )

    return ApiAttr(
        name=_inline_text(name_el) if name_el is not None else "",
        type=_inline_text(type_el) if type_el is not None else None,
        facets=tuple(_inline_text(f) for f in facet_els),
        description=description,
        children=children,
    )


# -- inline text -------------------------------------------------------------


def _inline_text(el: Tag, *, skip: Callable[[object], bool] | None = None) -> str:
    """An element's text, with inline formatting rendered as markdown.

    `<strong>`/`<b>` -> `**text**`, `<em>`/`<i>` -> `*text*`, `<code>` ->
    `` `text` ``, `<a href>` -> `[text](href)` (the href is kept exactly as
    found -- see parsing.md's "Link targets" resolution). Hash-link
    permalink glyphs, icons (`<svg>`), and `<!-- -->` comments contribute
    nothing. Whitespace (including the newlines Docusaurus's source HTML is
    wrapped at) is collapsed to single spaces, since this is prose, not code.
    """
    parts = [
        _inline_fragment(child)
        for child in el.children
        if skip is None or not skip(child)
    ]
    return _normalize_whitespace("".join(parts))


def _inline_fragment(node: object) -> str:
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""

    classes = _classes(node)
    if node.name == "a" and _has_class(classes, "hash-link"):
        return ""
    if node.name == "svg":
        return ""
    if node.name == "br":
        return "\n"
    if node.name == "a":
        inner = _inline_text(node)
        href = node.get("href", "")
        return f"[{inner}]({href})" if inner else ""
    if node.name in ("strong", "b"):
        inner = _inline_text(node)
        return f"**{inner}**" if inner else ""
    if node.name in ("em", "i"):
        inner = _inline_text(node)
        return f"*{inner}*" if inner else ""
    if node.name == "code":
        inner = node.get_text()
        return f"`{inner}`" if inner else ""
    return _inline_text(node)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


# -- small helpers -------------------------------------------------------


def _classes(el: Tag) -> tuple[str, ...]:
    classes = el.get("class") or []
    return tuple(classes) if isinstance(classes, list) else (classes,)


def _has_class(classes: Iterable[str], *markers: str) -> bool:
    return any(marker in cls for cls in classes for marker in markers)


def _class_contains(marker: str) -> Callable[[str | None], bool]:
    def predicate(value: str | None) -> bool:
        return value is not None and marker in value

    return predicate
