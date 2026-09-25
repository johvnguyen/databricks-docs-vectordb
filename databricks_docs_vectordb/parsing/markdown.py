"""Render an `html_walker` node tree to markdown text.

Inline formatting (bold, italic, inline code, links) is already resolved
into markdown syntax by `html_walker._inline_text` -- this module only
handles block-level layout: headings, lists, tables, code fences,
admonitions, tab groups, and the API reference's nested parameter trees.

Per parsing.md's rendering rules:
- Tabs: every panel rendered inline, each preceded by a bold label.
- Admonitions: a blockquote with a bold type label.
- Nested API parameter trees: an indented markdown list, one line per
  parameter, expanding what was inside a collapsed `<details>`.
- Code blocks: fenced with the recovered language.
- Links: already left as the absolute Databricks URL found in `href` by
  the walker; nothing left to do here.
"""

from collections.abc import Sequence

from databricks_docs_vectordb.parsing.html_walker import (
    Admonition,
    ApiAttr,
    CodeBlock,
    Heading,
    ListBlock,
    Node,
    Paragraph,
    QuoteBlock,
    TabGroup,
    Table,
)


def render(nodes: Sequence[Node]) -> str:
    """Render an ordered node list to markdown, blocks separated by blank lines.

    Consecutive top-level `ApiAttr` nodes (siblings in the raw HTML, e.g.
    every parameter of one API resource) are rendered as a single combined
    list rather than one one-item list per parameter.
    """
    blocks: list[str] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if isinstance(node, ApiAttr):
            group = []
            while i < len(nodes) and isinstance(nodes[i], ApiAttr):
                group.append(nodes[i])
                i += 1
            blocks.append(_render_api_attrs(group))
            continue
        block = _render_one(node)
        if block:
            blocks.append(block)
        i += 1
    return "\n\n".join(blocks)


def _render_one(node: Node) -> str:
    if isinstance(node, Heading):
        return f"{'#' * node.level} {node.text}"
    if isinstance(node, Paragraph):
        return node.text
    if isinstance(node, ListBlock):
        return _render_list(node)
    if isinstance(node, QuoteBlock):
        return _quote_wrap(node.text)
    if isinstance(node, Table):
        return _render_table(node)
    if isinstance(node, CodeBlock):
        return _render_code(node)
    if isinstance(node, Admonition):
        return _render_admonition(node)
    if isinstance(node, TabGroup):
        return _render_tab_group(node)
    if isinstance(node, ApiAttr):
        # A lone ApiAttr not part of a contiguous run (render() groups runs
        # before reaching here; this is a defensive fallback).
        return _render_api_attrs([node])
    raise TypeError(f"unknown node type: {type(node).__name__}")


# -- lists -------------------------------------------------------------------


def _render_list(node: ListBlock, indent: int = 0) -> str:
    lines = []
    for i, item in enumerate(node.items, start=1):
        bullet = f"{i}." if node.ordered else "-"
        lines.append(f"{'  ' * indent}{bullet} {item.text}")
        if item.children:
            # A nested <ul>/<ol>'s own ordered-ness isn't tracked separately
            # by html_walker (it flattens all of an item's nested lists into
            # one `children` tuple) -- nested sub-items always render as an
            # unordered list.
            nested = ListBlock(ordered=False, items=item.children)
            lines.append(_render_list(nested, indent + 1))
    return "\n".join(lines)


# -- quotes and admonitions ---------------------------------------------------


def _quote_wrap(text: str) -> str:
    if not text:
        return ">"
    return "\n".join(f"> {line}" if line else ">" for line in text.split("\n"))


def _render_admonition(node: Admonition) -> str:
    label = f"**{(node.type or 'note').capitalize()}:**"
    if node.children and isinstance(node.children[0], Paragraph):
        first_line = f"{label} {node.children[0].text}"
        rest = render(node.children[1:])
        body = f"{first_line}\n\n{rest}" if rest else first_line
    else:
        content = render(node.children)
        body = f"{label}\n\n{content}" if content else label
    return _quote_wrap(body)


# -- tables --------------------------------------------------------------


def _render_table(node: Table) -> str:
    if not node.headers and not node.rows:
        return ""
    ncols = len(node.headers) if node.headers else len(node.rows[0])
    headers = node.headers or tuple(f"Column {i + 1}" for i in range(ncols))

    lines = [_table_row(headers), _table_row(["---"] * len(headers))]
    for row in node.rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        lines.append(_table_row(padded[: len(headers)]))
    return "\n".join(lines)


def _table_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(_escape_table_cell(c) for c in cells) + " |"


def _escape_table_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


# -- code blocks -----------------------------------------------------------


def _render_code(node: CodeBlock) -> str:
    fence = _fence_for(node.code)
    return f"{fence}{node.language or ''}\n{node.code}\n{fence}"


def _fence_for(code: str) -> str:
    """A backtick fence longer than any backtick run already in `code`.

    So a code sample that itself shows a markdown fence (e.g. a snippet
    demonstrating a docstring) can't prematurely close the block.
    """
    longest = run = 0
    for ch in code:
        if ch == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return "`" * max(3, longest + 1)


# -- tabs --------------------------------------------------------------------


def _render_tab_group(node: TabGroup) -> str:
    blocks = []
    for tab in node.tabs:
        label = f"**{tab.label}:**"
        content = render(tab.children)
        blocks.append(f"{label}\n\n{content}" if content else label)
    return "\n\n".join(blocks)


# -- API parameter trees -------------------------------------------------


def _render_api_attrs(attrs: Sequence[ApiAttr], indent: int = 0) -> str:
    lines = []
    for attr in attrs:
        lines.append(_render_api_attr_line(attr, indent))
        if attr.children:
            lines.append(_render_api_attrs(attr.children, indent + 1))
    return "\n".join(lines)


def _render_api_attr_line(attr: ApiAttr, indent: int) -> str:
    head_parts = [f"**{attr.name}**"]
    if attr.type:
        head_parts.append(f"({attr.type})")
    if attr.facets:
        head_parts.append(" ".join(f"`{facet}`" for facet in attr.facets))
    head = " ".join(head_parts)
    prefix = f"{'  ' * indent}- "
    if attr.description:
        # A markdown list item can't hold a blank-line-separated paragraph
        # without extra indentation; flatten a multi-paragraph description
        # onto one line rather than break the list structure.
        description = attr.description.replace("\n\n", " ")
        return f"{prefix}{head}: {description}"
    return f"{prefix}{head}"
