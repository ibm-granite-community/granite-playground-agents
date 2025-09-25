import re

from markdown_it import MarkdownIt
from pydantic import BaseModel


class MarkdownText(BaseModel):
    content: str
    start_index: int
    end_index: int


class MarkdownToken(MarkdownText):
    type: str
    tag: str


class MarkdownSection(MarkdownText):
    pass


def get_markdown_tokens_with_content(markdown_text: str) -> list[MarkdownToken]:
    md = MarkdownIt("commonmark")
    tokens = md.parse(markdown_text)

    lines = markdown_text.splitlines(keepends=True)
    line_start_offsets = []
    offset = 0
    for line in lines:
        line_start_offsets.append(offset)
        offset += len(line)

    token_offsets = []

    for token in tokens:
        if token.map is None or not token.content:
            continue

        start_line, end_line = token.map

        # Calculate range covering the whole mapped lines
        start_char = line_start_offsets[start_line]
        # end_line is exclusive in token.map, so get start offset of end_line or text end
        end_char = line_start_offsets[end_line] if end_line < len(line_start_offsets) else len(markdown_text)

        # Extract the substring of the mapped lines
        block_text = markdown_text[start_char:end_char]

        # Try to locate token.content inside block_text
        relative_pos = block_text.find(token.content)

        if relative_pos == -1:
            # fallback to line-based offsets if not found (rare)
            token_start = start_char
            token_end = end_char
        else:
            token_start = start_char + relative_pos
            token_end = token_start + len(token.content)

        token_offsets.append(
            MarkdownToken(
                type=token.type,
                tag=token.tag,
                content=token.content,
                start_index=token_start,
                end_index=token_end,
            )
        )

    return token_offsets


def get_markdown_sections(markdown_text: str) -> list[MarkdownSection]:
    # Matches:
    #   - Markdown headings: # Heading
    #   - Bold headings: **Heading**
    heading_pattern = re.compile(
        r"^(?P<full>(?P<hashes>#{1,6})\s+(?P<title1>[^\n]+)|(?P<bold>\*\*(?P<title2>.+?)\*\*))", re.MULTILINE
    )

    matches = list(heading_pattern.finditer(markdown_text))
    sections = []

    # Handle text before first heading (if any)
    if matches:
        first_start = matches[0].start()
        if first_start > 0:
            content = markdown_text[:first_start]
            sections.extend(split_markdown_paragraphs(content, 0))
    else:
        # No headings found
        return split_markdown_paragraphs(markdown_text, 0)

    # Now process each heading and the content after it
    for i, match in enumerate(matches):
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        content = markdown_text[content_start:content_end]

        sections.extend(split_markdown_paragraphs(content, content_start))

    return sections


def split_markdown_paragraphs(text: str, offset: int) -> list[MarkdownSection]:
    paragraphs = []
    splits = re.split(r"\n\s*\n", text)
    search_start = 0

    for para in splits:
        para = para.strip()
        if not para:
            continue
        # Find paragraph position in original text starting from search_start
        start = text.find(para, search_start)
        if start == -1:
            continue
        end = start + len(para)
        paragraphs.append(MarkdownSection(content=para, start_index=offset + start, end_index=offset + end))
        search_start = end
    return paragraphs
