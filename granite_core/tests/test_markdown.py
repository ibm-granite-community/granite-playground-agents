# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from granite_core.markdown import get_markdown_sections, get_markdown_tokens_with_content


def test_markdown_sectioning() -> None:
    """Test markdown processing"""

    section_text = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",  # noqa: E501
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",  # noqa: E501
    ]

    markdown_str = f"""
{section_text[0]}

# Heading
{section_text[1]}

## Heading
{section_text[2]}

**bold heading**
{section_text[3]}
"""

    extracted_sections = get_markdown_sections(markdown_str)
    assert len(extracted_sections) == len(section_text)
    for i in range(len(section_text)):
        assert extracted_sections[i].content == section_text[i]


def test_markdown_tokenization() -> None:
    """Test inline token extraction"""

    section_text = [
        "*Lorem* ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",  # noqa: E501
        "Lorem",
        "Ipsum",
        "Dolor",
    ]

    markdown_text = f"""{section_text[0]}
- {section_text[1]}
- {section_text[2]}
1. {section_text[3]}
"""
    tokens = get_markdown_tokens_with_content(markdown_text)

    assert len(tokens) == len(section_text)

    for i in range(len(section_text)):
        start, end = tokens[i].start_index, tokens[i].end_index
        print(markdown_text[start:end])
        print(section_text[i])
        assert markdown_text[start:end] == section_text[i]
