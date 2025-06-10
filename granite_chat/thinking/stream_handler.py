from collections import deque
from collections.abc import Generator

from pydantic import BaseModel


class TokenEvent(BaseModel):
    token: str | None = None
    tag: str | None = None


class TagStartEvent(BaseModel):
    tag: str | None = None


class ThinkingStreamHandler:
    def __init__(self, tags: list[str]) -> None:
        self.tag_names = {f"<{t}>": t for t in tags}
        self.tags = {f"<{t}>": f"</{t}>" for t in tags}
        self.start_tags = list(self.tags.keys())
        self.end_tags = list(self.tags.values())
        self.buffer = ""
        self.current_tag: str | None = None
        self.end_tag_len = max([len(t) for t in self.end_tags])
        self.lookahead: deque = deque()

    def on_token(self, token: str) -> Generator[TokenEvent | TagStartEvent, None, None]:
        self.buffer += token

        # Wait for start tag
        if self.current_tag is None:
            for st in self.start_tags:
                if st in self.buffer:
                    # Strip up to end of start tag
                    self.buffer = self.buffer.split(st, 1)[1]
                    self.current_tag = st
                    yield TagStartEvent(tag=self.tag_names[self.current_tag])
        else:
            self.lookahead.append(token)
            # Build current string from lookahead
            current = "".join(self.lookahead)
            first_token_len = len(self.lookahead[0])
            end_tag = self.tags[self.current_tag]

            if end_tag in current:
                before, after = current.split(end_tag, 1)
                yield TokenEvent(token=before, tag=self.tag_names[self.current_tag])
                self.buffer = after
                self.lookahead = deque()
                self.current_tag = None

            elif len(current) - first_token_len > self.end_tag_len:
                # It's safe to emit the first token
                yield TokenEvent(token=self.lookahead.popleft(), tag=self.tag_names[self.current_tag])
