# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


def sanitize_for_embedding(text: str) -> str:
    # Remove lone backslashes and control chars
    return text.replace("\\", "\\\\").replace("\x00", "")
