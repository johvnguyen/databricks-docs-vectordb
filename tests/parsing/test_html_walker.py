from pathlib import Path

import pytest

from databricks_docs_vectordb.parsing.html_walker import (
    Admonition,
    ApiAttr,
    CodeBlock,
    Heading,
    ListBlock,
    NoArticleError,
    Paragraph,
    QuoteBlock,
    TabGroup,
    Table,
    walk,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "parsing"


def load(*parts: str) -> str:
    return (FIXTURES.joinpath(*parts)).read_text(encoding="utf-8")


def all_text(nodes) -> str:
    """Every bit of text anywhere in a node tree, for artifact-leak checks."""
    parts = []
    for node in nodes:
        if isinstance(node, Heading | Paragraph | QuoteBlock):
            parts.append(node.text)
        elif isinstance(node, ListBlock):
            for item in node.items:
                parts.append(item.text)
        elif isinstance(node, Table):
            parts.extend(node.headers)
            for row in node.rows:
                parts.extend(row)
        elif isinstance(node, Admonition):
            parts.append(node.type)
            parts.append(all_text(node.children))
        elif isinstance(node, TabGroup):
            for tab in node.tabs:
                parts.append(tab.label)
                parts.append(all_text(tab.children))
        elif isinstance(node, ApiAttr):
            parts.append(node.name)
            parts.append(node.description)
            parts.append(all_text(node.children))
        # CodeBlock is deliberately excluded: source code legitimately
        # contains "<!--" or other lookalike substrings.
    return "\n".join(parts)


def assert_no_artifacts(nodes) -> None:
    text = all_text(nodes)
    assert "<!--" not in text
    assert "-->" not in text
    assert "hash-link" not in text
    assert "\u200b" not in text  # the hash-link glyph's zero-width space


def find_all(nodes, node_type):
    return [n for n in nodes if isinstance(n, node_type)]


def find_one(nodes, node_type, **attrs):
    for n in nodes:
        if isinstance(n, node_type) and all(
            getattr(n, k) == v for k, v in attrs.items()
        ):
            return n
    raise AssertionError(f"no {node_type.__name__} with {attrs} in {nodes!r}")


# -- no <article> ----------------------------------------------------------


def test_a_page_with_no_article_element_is_rejected():
    with pytest.raises(NoArticleError):
        walk("<html><body><p>no article here</p></body></html>")


def test_an_article_with_no_markdown_div_is_rejected():
    with pytest.raises(NoArticleError):
        walk("<html><body><article><p>wrong shape</p></article></body></html>")


# -- api/ pages: the "View as Markdown" toolbar is not content ------------


def test_the_view_as_markdown_toolbar_is_stripped():
    html = (
        "<html><body><article>"
        '<div class="theme-doc-markdown markdown">'
        "<header><h1>Title</h1></header>"
        '<div class="apiMdActions">'
        '<a href="/api/x.md">View as Markdown</a>'
        '<button class="apiMdCopy">Copy as Markdown</button>'
        "</div>"
        '<h2 id="s">Section</h2><p>Real content.</p>'
        "</div></article></body></html>"
    )
    nodes = walk(html)
    assert nodes == [
        Heading(level=2, id="s", text="Section"),
        Paragraph("Real content."),
    ]


# -- delta/index.html: plain prose + headings + links -----------------------


@pytest.fixture(scope="module")
def delta_nodes():
    return walk(load("delta", "index.html"))


def test_delta_has_headings_and_paragraphs(delta_nodes):
    assert find_all(delta_nodes, Heading)
    assert find_all(delta_nodes, Paragraph)


def test_delta_the_page_h1_is_not_a_content_node(delta_nodes):
    # "What is Delta Lake in Databricks?" is the <h1>; metadata.py's job.
    assert all(h.level >= 2 for h in find_all(delta_nodes, Heading))


def test_delta_headings_keep_their_slug_id_and_drop_the_hash_link(delta_nodes):
    heading = find_one(delta_nodes, Heading, id="getting-started-with-delta-lake")
    assert heading.text == "Getting started with Delta Lake"


def test_delta_internal_links_become_markdown_links_with_the_href_unchanged(
    delta_nodes,
):
    assert "[ACID transactions](/aws/en/lakehouse/acid)" in all_text(delta_nodes)


def test_delta_external_links_keep_their_full_url(delta_nodes):
    assert "(https://delta.io)" in all_text(delta_nodes)


def test_delta_no_stripped_artifacts_leak_into_any_node(delta_nodes):
    assert_no_artifacts(delta_nodes)


# -- oltp/...: tabs + table + code + admonitions -----------------------------


@pytest.fixture(scope="module")
def oltp_nodes():
    return walk(load("oltp", "update-to-autoscaling-terraform.html"))


def test_oltp_has_a_table_and_code_blocks(oltp_nodes):
    assert find_all(oltp_nodes, Table)
    assert find_all(oltp_nodes, CodeBlock)


def test_oltp_table_has_headers_and_rows(oltp_nodes):
    table = find_all(oltp_nodes, Table)[0]
    assert table.headers == ("Provisioned", "Autoscaling")
    assert len(table.rows) > 1
    assert all(len(row) == 2 for row in table.rows)


def test_oltp_code_block_language_is_recovered_from_the_pre_class(oltp_nodes):
    code_blocks = find_all(oltp_nodes, CodeBlock)
    # Most of this page's blocks are Terraform (hcl); one is a plain-text
    # command-output preview -- both languages must be recovered correctly,
    # not just defaulted to one value.
    languages = {c.language for c in code_blocks}
    assert languages == {"hcl", "text"}
    assert any(
        'endpoint_type = "ENDPOINT_TYPE_READ_WRITE"' in c.code for c in code_blocks
    )


def test_oltp_both_tab_panels_are_kept_each_tagged_with_its_label(oltp_nodes):
    tab_group = find_all(oltp_nodes, TabGroup)[0]
    assert [t.label for t in tab_group.tabs] == [
        "Only import the Autoscaling resources",
        "Use Autoscaling feature during the import",
    ]
    # Docusaurus marks the inactive panel `hidden`; both must survive.
    for tab in tab_group.tabs:
        assert find_all(tab.children, CodeBlock)


def test_oltp_admonitions_carry_their_type_and_content(oltp_nodes):
    admonitions = find_all(oltp_nodes, Admonition)
    assert admonitions
    assert all(a.type == "note" for a in admonitions)
    assert all(find_all(a.children, Paragraph) for a in admonitions)


def test_oltp_no_stripped_artifacts_leak_into_any_node(oltp_nodes):
    assert_no_artifacts(oltp_nodes)


# -- api/jobs/v2/run.html: nested apiAttr / details --------------------------


@pytest.fixture(scope="module")
def cluster_spec():
    nodes = walk(load("api", "jobs", "v2", "run.html"))
    return find_one(nodes, ApiAttr, name="cluster_spec")


def test_api_top_level_attrs_are_recovered(cluster_spec):
    assert cluster_spec.type == "object"
    assert cluster_spec.description  # non-empty prose


def test_api_full_nesting_depth_is_recovered_past_a_collapsed_details(cluster_spec):
    # cluster_spec -> new_cluster -> autoscale -> min_workers/max_workers is
    # 3 levels deep, and each level lives inside a collapsed
    # <details data-collapsed="true">, not just the first.
    new_cluster = find_one(cluster_spec.children, ApiAttr, name="new_cluster")
    assert new_cluster.type == "object"

    autoscale = find_one(new_cluster.children, ApiAttr, name="autoscale")
    assert autoscale.type == "object"

    leaf_names = {c.name for c in autoscale.children}
    assert leaf_names == {"min_workers", "max_workers"}
    min_workers = find_one(autoscale.children, ApiAttr, name="min_workers")
    assert min_workers.type == "int32"
    assert min_workers.children == ()
    assert "minimum number of workers" in min_workers.description


def test_api_facets_are_captured(cluster_spec):
    job_cluster_key = find_one(cluster_spec.children, ApiAttr, name="job_cluster_key")
    assert job_cluster_key.facets == (
        "[ 1 .. 100 ] characters",
        "^[\\\\w\\\\-\\\\_]+$",
    )


def test_api_no_stripped_artifacts_leak_into_any_node(cluster_spec):
    assert_no_artifacts([cluster_spec])


# -- error-messages/error-classes.html: large, heading-heavy -----------------


@pytest.fixture(scope="module")
def error_classes_nodes():
    return walk(load("error-messages", "error-classes.html"))


def test_error_classes_many_headings_are_recovered(error_classes_nodes):
    assert len(find_all(error_classes_nodes, Heading)) > 50


def test_error_classes_heading_levels_are_h2_and_h3_only(error_classes_nodes):
    levels = {h.level for h in find_all(error_classes_nodes, Heading)}
    assert levels == {2, 3}


def test_error_classes_no_stripped_artifacts_leak_into_any_node(error_classes_nodes):
    assert_no_artifacts(error_classes_nodes)


# -- ML tutorial page: an oversized single code block ------------------------


@pytest.fixture(scope="module")
def tutorial_nodes():
    return walk(load("machine-learning", "sgc-yolo11n-detect-coco128.html"))


def test_tutorial_has_a_much_larger_code_block_than_its_prose(tutorial_nodes):
    code_blocks = find_all(tutorial_nodes, CodeBlock)
    paragraphs = find_all(tutorial_nodes, Paragraph)
    assert code_blocks
    biggest_code = max(len(c.code) for c in code_blocks)
    avg_paragraph = sum(len(p.text) for p in paragraphs) / len(paragraphs)
    assert biggest_code > avg_paragraph * 10


def test_tutorial_multiline_code_is_preserved_with_real_newlines(tutorial_nodes):
    code_blocks = find_all(tutorial_nodes, CodeBlock)
    biggest = max(code_blocks, key=lambda c: len(c.code))
    assert biggest.code.count("\n") > 10


def test_tutorial_a_genuine_blockquote_becomes_a_quote_block_not_an_admonition(
    tutorial_nodes,
):
    assert find_all(tutorial_nodes, QuoteBlock)


def test_tutorial_no_stripped_artifacts_leak_into_any_node(tutorial_nodes):
    assert_no_artifacts(tutorial_nodes)
