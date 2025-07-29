import re

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
        if token.map is None:
            continue

        start_line, end_line = token.map
        char_start = line_start_offsets[start_line]
        char_end = line_start_offsets[end_line] if end_line < len(line_start_offsets) else len(markdown_text)

        token_offsets.append(
            MarkdownToken(
                type=token.type, tag=token.tag, content=token.content, start_index=char_start, end_index=char_end
            )
        )

    return token_offsets


def split_on_last_special(sentence: str) -> tuple[str, str, int]:
    # Search for last occurrence of '**' or ':'
    match = re.search(r"(.*)(\*\*|:)([^:*]*)$", sentence)
    if match:
        before = match.group(1)
        sep = match.group(2)
        after = match.group(3)
        index = len(before) + len(sep)  # position right after the matched separator
        return before, after, index
    return sentence, "", -1  # -1 means no match found
