import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString


def clean_text(text):
    # Normalize whitespaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def node_to_markdown(node):
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        text = str(node)
        return text if text.strip() else ""

    markdown = []

    # Process the node itself
    if node.name in ["p", "span"]:
        text = node.get_text().strip()
        if text:
            return f"\n{text}\n"
    elif node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        level = int(node.name[1])
        return f"\n{'#' * level} {node.get_text().strip()}\n"
    elif node.name == "ul":
        items = []
        for li in node.find_all("li", recursive=False):
            items.append(f"- {li.get_text().strip()}\n")
        return "\n" + "".join(items) + "\n"
    elif node.name == "ol":
        items = []
        for i, li in enumerate(node.find_all("li", recursive=False), 1):
            items.append(f"{i}. {li.get_text().strip()}\n")
        return "\n" + "".join(items) + "\n"
    elif node.name in ["strong", "b"]:
        return f" **{node.get_text().strip()}** "
    elif node.name in ["em", "i"]:
        return f" *{node.get_text().strip()}* "
    elif node.name == "br":
        return "\n"
    elif node.name == "table":
        # Skip the intro note table
        if "biên soạn lại" in node.get_text():
            return ""
        # Basic table parsing if any other table exists
        rows = []
        for tr in node.find_all("tr"):
            cells = [td.get_text().strip() for td in tr.find_all(["td", "th"])]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            # Add a simple separator after the first row (header)
            num_cols = len(node.find("tr").find_all(["td", "th"]))
            separator = "| " + " | ".join(["---"] * num_cols) + " |"
            rows.insert(1, separator)
            return "\n" + "\n".join(rows) + "\n"

    # For other containers (div, etc.), recurse into children
    for child in node.children:
        child_md = node_to_markdown(child)
        if child_md:
            markdown.append(child_md)

    return "".join(markdown)


def parse_leaflet_html(file_path):
    with open(file_path, encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Get Title / Brand Name
    h1 = soup.find("h1")
    if h1:
        brand_name = h1.get_text().strip()
    else:
        title_tag = soup.find("title")
        brand_name = (
            title_tag.get_text().split("|")[0].strip()
            if title_tag
            else "Không rõ tên thuốc"
        )

    # Extract the Hướng dẫn sử dụng section
    huong_dan_h2 = None
    for h2 in soup.find_all("h2"):
        if "Hướng dẫn sử dụng" in h2.text:
            huong_dan_h2 = h2
            break

    if not huong_dan_h2:
        return None  # No user guide section found

    parent = huong_dan_h2.parent
    if parent is None:
        return None
    container_div = parent.find("div")
    if not container_div:
        return None

    sub_divs = container_div.find_all("div", recursive=False)
    content_node = container_div if not sub_divs else sub_divs[0]

    # Convert the content node to markdown
    markdown_content = []
    markdown_content.append(f"# {brand_name}\n\n")

    for child in content_node.children:
        child_md = node_to_markdown(child)
        if child_md.strip():
            markdown_content.append(child_md)

    full_md = "".join(markdown_content)

    # Post-processing cleanup
    full_md = re.sub(r"\n{3,}", "\n\n", full_md)
    full_md = clean_text(full_md)

    return full_md


@dataclass(frozen=True)
class ParseHtmlResult:
    total: int
    success: int
    skipped: int


def parse_html_tree(html_dir: Path, output_dir: Path) -> ParseHtmlResult:
    html_dir = Path(html_dir)
    output_dir = Path(output_dir)
    if not html_dir.is_dir():
        raise FileNotFoundError(f"HTML directory does not exist: {html_dir}")
    html_files = sorted(html_dir.rglob("*.html"))
    total_files = len(html_files)
    success_count = 0
    skipped_count = 0
    for file_path in html_files:
        md_content = parse_leaflet_html(file_path)
        if md_content:
            destination = output_dir / file_path.relative_to(html_dir).with_suffix(
                ".md"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(md_content, encoding="utf-8")
            success_count += 1
        else:
            skipped_count += 1
    return ParseHtmlResult(total_files, success_count, skipped_count)
