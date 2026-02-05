# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from contextlib import suppress


def sanitize_for_embedding(s: str) -> str:
    # decode unicode escapes if possible
    with suppress(UnicodeDecodeError):
        s = s.encode(encoding="utf-8").decode(encoding="unicode_escape")

    #  remove control characters
    s = "".join(c for c in s if ord(c) >= 32)
    return s
