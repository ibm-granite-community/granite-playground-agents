from markdown_it import MarkdownIt
from pydantic import BaseModel


class MarkdownToken(BaseModel):
    type: str
    tag: str
    content: str
    start_index: int
    end_index: int


def get_markdown_tokens(markdown_text: str) -> list[MarkdownToken]:
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
