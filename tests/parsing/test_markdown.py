from pathlib import Path

from databricks_docs_vectordb.parsing.html_walker import (
    Admonition,
    ApiAttr,
    CodeBlock,
    Heading,
    ListBlock,
    ListItem,
    Paragraph,
    QuoteBlock,
    TabGroup,
    Table,
    TabPanel,
    walk,
)
from databricks_docs_vectordb.parsing.markdown import render

FIXTURES = Path(__file__).parent.parent / "fixtures" / "parsing"


def load(*parts: str) -> str:
    return (FIXTURES.joinpath(*parts)).read_text(encoding="utf-8")


# -- isolated node rendering (exact expected output) -------------------------


def test_heading_levels_map_to_hash_counts():
    assert render([Heading(level=2, id="x", text="Title")]) == "## Title"
    assert render([Heading(level=3, id="x", text="Sub")]) == "### Sub"


def test_paragraph_renders_as_is():
    assert render([Paragraph("Hello world.")]) == "Hello world."


def test_blocks_are_separated_by_a_blank_line():
    out = render([Paragraph("First."), Paragraph("Second.")])
    assert out == "First.\n\nSecond."


def test_unordered_list():
    block = ListBlock(ordered=False, items=(ListItem("one"), ListItem("two")))
    assert render([block]) == "- one\n- two"


def test_ordered_list():
    block = ListBlock(ordered=True, items=(ListItem("one"), ListItem("two")))
    assert render([block]) == "1. one\n2. two"


def test_nested_list_item_is_indented_two_spaces():
    inner = ListItem("child")
    outer = ListBlock(ordered=False, items=(ListItem("parent", children=(inner,)),))
    assert render([outer]) == "- parent\n  - child"


def test_quote_block_prefixes_every_line():
    assert render([QuoteBlock("line one\nline two")]) == "> line one\n> line two"


def test_table_renders_a_header_separator_and_rows():
    table = Table(headers=("A", "B"), rows=(("1", "2"), ("3", "4")))
    assert render([table]) == ("| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |")


def test_table_cell_pipes_and_newlines_are_escaped():
    table = Table(headers=("A",), rows=(("has | pipe\nand newline",),))
    assert render([table]) == "| A |\n| --- |\n| has \\| pipe and newline |"


def test_code_block_is_fenced_with_its_language():
    block = CodeBlock(language="python", code="print(1)")
    assert render([block]) == "```python\nprint(1)\n```"


def test_code_block_with_no_language_fences_with_an_empty_info_string():
    block = CodeBlock(language=None, code="plain text")
    assert render([block]) == "```\nplain text\n```"


def test_code_fence_is_widened_past_backtick_runs_already_in_the_code():
    code = "before\n```python\nx = 1\n```\nafter"
    block = CodeBlock(language="markdown", code=code)
    out = render([block])
    assert out.startswith("````markdown\n")
    assert out.endswith("\n````")
    assert code in out


def test_admonition_is_a_blockquote_with_a_bold_type_label():
    admonition = Admonition(type="tip", children=(Paragraph("Be careful."),))
    assert render([admonition]) == "> **Tip:** Be careful."


def test_admonition_with_multiple_children_keeps_the_rest_quoted():
    admonition = Admonition(
        type="warning",
        children=(Paragraph("First line."), Paragraph("Second block.")),
    )
    out = render([admonition])
    assert out.startswith("> **Warning:** First line.\n>\n> Second block.")
    assert all(line.startswith(">") for line in out.split("\n"))


def test_tab_group_renders_every_panel_with_a_bold_label():
    group = TabGroup(
        tabs=(
            TabPanel(label="Python", children=(CodeBlock("python", "x = 1"),)),
            TabPanel(label="SQL", children=(CodeBlock("sql", "SELECT 1"),)),
        )
    )
    out = render([group])
    assert out == (
        "**Python:**\n\n```python\nx = 1\n```\n\n**SQL:**\n\n```sql\nSELECT 1\n```"
    )


def test_a_single_api_attr_renders_as_a_one_line_bullet():
    attr = ApiAttr(name="job_id", type="int64", facets=(), description="The job ID.")
    assert render([attr]) == "- **job_id** (int64): The job ID."


def test_api_attr_facets_are_rendered_as_inline_code():
    attr = ApiAttr(
        name="job_cluster_key",
        type="string",
        facets=("[ 1 .. 100 ] characters",),
        description="",
    )
    assert render([attr]) == "- **job_cluster_key** (string) `[ 1 .. 100 ] characters`"


def test_nested_api_attrs_are_indented_two_spaces_per_level():
    leaf = ApiAttr(name="min_workers", type="int32", facets=(), description="Min.")
    mid = ApiAttr(
        name="autoscale", type="object", facets=(), description="", children=(leaf,)
    )
    top = ApiAttr(
        name="cluster_spec", type="object", facets=(), description="", children=(mid,)
    )
    out = render([top])
    assert out == (
        "- **cluster_spec** (object)\n"
        "  - **autoscale** (object)\n"
        "    - **min_workers** (int32): Min."
    )


def test_consecutive_top_level_api_attrs_render_as_one_combined_list():
    a = ApiAttr(name="a", type="string", facets=(), description="")
    b = ApiAttr(name="b", type="string", facets=(), description="")
    # One combined block (no blank line between them), unlike two Paragraphs.
    assert render([a, b]) == "- **a** (string)\n- **b** (string)"


def test_api_attrs_separated_by_another_node_are_not_merged():
    a = ApiAttr(name="a", type="string", facets=(), description="")
    heading = Heading(level=3, id="response", text="Response")
    b = ApiAttr(name="b", type="string", facets=(), description="")
    out = render([a, heading, b])
    assert out == "- **a** (string)\n\n### Response\n\n- **b** (string)"


# -- fixture-based checks for the trickier real elements --------------------


def test_oltp_fixture_tab_group_renders_both_labels_and_both_code_fences():
    nodes = walk(load("oltp", "update-to-autoscaling-terraform.html"))
    tab_group = next(n for n in nodes if isinstance(n, TabGroup))
    out = render([tab_group])
    assert "**Only import the Autoscaling resources:**" in out
    assert "**Use Autoscaling feature during the import:**" in out
    assert out.count("```hcl") == 2


def test_oltp_fixture_admonitions_render_as_quoted_bold_note_labels():
    nodes = walk(load("oltp", "update-to-autoscaling-terraform.html"))
    out = render(nodes)
    assert "> **Note:**" in out


def test_oltp_fixture_table_renders_as_a_markdown_table():
    nodes = walk(load("oltp", "update-to-autoscaling-terraform.html"))
    out = render(nodes)
    assert "| Provisioned | Autoscaling |" in out
    assert "| --- | --- |" in out


def test_api_fixture_nested_list_reaches_the_full_depth_with_indentation():
    nodes = walk(load("api", "jobs", "v2", "run.html"))
    out = render(nodes)
    # cluster_spec -> new_cluster -> autoscale -> min_workers/max_workers,
    # each one indentation level deeper than its parent.
    assert "- **cluster_spec** (object):" in out
    assert "  - **new_cluster** (object):" in out
    assert "    - **autoscale** (object):" in out
    assert "      - **min_workers** (int32):" in out
    assert "      - **max_workers** (int32):" in out


def test_api_fixture_top_level_attrs_form_one_combined_list():
    nodes = walk(load("api", "jobs", "v2", "run.html"))
    out = render(nodes)
    # No blank line should separate sibling top-level parameters.
    assert "- **job_id** (int64):" in out
    assert "\n- **run_id** (int64):" in out
    assert "- **job_id** (int64):\n\n" not in out


def test_error_classes_fixture_renders_many_headings():
    nodes = walk(load("error-messages", "error-classes.html"))
    out = render(nodes)
    assert out.count("\n## ") + out.startswith("## ") >= 1
    assert out.count("### ") > 50


def test_tutorial_fixture_code_block_is_fenced_with_python():
    nodes = walk(load("machine-learning", "sgc-yolo11n-detect-coco128.html"))
    out = render(nodes)
    assert "```python" in out
