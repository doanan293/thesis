import json
from pathlib import Path

from corpus_pipeline.corpus.crawling.integrate_ankhang import integrate_ankhang_corpus
from corpus_pipeline.corpus.crawling.parse_html import parse_html_tree


def test_parse_html_tree_writes_only_requested_output(tmp_path: Path) -> None:
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    (html_dir / "drug.html").write_text(
        "<html><h1>Thuốc A</h1><section><h2>Hướng dẫn sử dụng</h2>"
        "<div><div><p>Nội dung.</p></div></div></section></html>",
        encoding="utf-8",
    )
    output = tmp_path / "markdown"

    result = parse_html_tree(html_dir, output)

    assert result.total == 1
    assert result.success == 1
    assert list(output.rglob("*.md")) == [output / "drug.md"]


def test_integrate_ankhang_corpus_writes_requested_candidate(tmp_path: Path) -> None:
    markdown = tmp_path / "markdown"
    markdown.mkdir()
    (markdown / "drug-a.md").write_text("# Drug A\n\nNội dung thuốc.", encoding="utf-8")
    sections_in = tmp_path / "sections.in.jsonl"
    chunks_in = tmp_path / "chunks.in.jsonl"
    sections_in.write_text('{"id":"base","text":"base"}\n')
    chunks_in.write_text('{"id":"base:chunk-001","section_id":"base","text":"base"}\n')
    sections_out = tmp_path / "sections.out.jsonl"
    chunks_out = tmp_path / "chunks.out.jsonl"

    integrate_ankhang_corpus(
        markdown_dir=markdown,
        sections_path=sections_in,
        chunks_path=chunks_in,
        mappings_path=tmp_path / "missing-mappings.json",
        output_sections_path=sections_out,
        output_chunks_path=chunks_out,
    )

    assert sections_in.read_text() == '{"id":"base","text":"base"}\n'
    assert any(
        json.loads(line)["id"].startswith("brand:ankhang:")
        for line in sections_out.read_text().splitlines()
    )
