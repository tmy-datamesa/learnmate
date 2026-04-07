"""Parse Obsidian wiki markdown files and extract content with metadata.

Each wiki file becomes one document. Frontmatter (type, updated) is extracted
as metadata. The rest of the file becomes the document content.
"""

from pathlib import Path


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a markdown file into frontmatter metadata and body content.

    Args:
        text: Raw markdown file content.

    Returns:
        A tuple of (metadata dict, body string). If no frontmatter found,
        metadata is empty and body is the full text.
    """
    metadata: dict[str, str] = {}

    if not text.startswith("---"):
        return metadata, text

    # Find the closing --- of frontmatter
    end_index = text.find("---", 3)
    if end_index == -1:
        return metadata, text

    frontmatter_block = text[3:end_index].strip()
    body = text[end_index + 3:].strip()

    # Parse simple key: value pairs
    for line in frontmatter_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()

    return metadata, body


def load_wiki_documents(wiki_path: Path) -> list[dict]:
    """Load all markdown files from the wiki directory.

    Walks through wiki subfolders (concepts/, entities/, sources/) and
    creates one document per file with metadata.

    Args:
        wiki_path: Path to the Obsidian wiki directory.

    Returns:
        A list of dicts, each with keys: id, content, metadata.
        metadata contains: filename, type, updated, subfolder.
    """
    documents: list[dict] = []

    # Only process wiki content folders, skip index/log/bridge
    content_folders = ["concepts", "entities", "sources"]

    for folder_name in content_folders:
        folder = wiki_path / folder_name
        if not folder.exists():
            continue

        for md_file in sorted(folder.glob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            metadata, body = parse_frontmatter(text)

            if not body.strip():
                continue

            doc = {
                "id": f"{folder_name}/{md_file.stem}",
                "content": body,
                "metadata": {
                    "filename": md_file.stem,
                    "subfolder": folder_name,
                    "type": metadata.get("type", "unknown"),
                    "updated": metadata.get("updated", ""),
                    "source_url": metadata.get("source_url", ""),
                },
            }
            documents.append(doc)

    return documents
