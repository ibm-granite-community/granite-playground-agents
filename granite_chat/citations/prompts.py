import json

from langchain_core.documents import Document

from granite_chat.citations.types import Sentence


class CitationsPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_citations_prompt(sentences: list[Sentence], docs: list[Document]) -> str:
        json_sents = [{"sentence_id": str(i), "content": s.text} for i, s in enumerate(sentences)]
        sent_str = json.dumps(json_sents, indent=4)
        json_docs = [{"source_id": str(i), "content": d.page_content} for i, d in enumerate(docs)]
        doc_str = json.dumps(json_docs, indent=4)

        return f"""You are given:
1. A set of sources (with source_id and content).
2. A set of sentences that may contain ideas or information supported by the sources.

Your task is to produce citations that link specific sentences to the sources.
A citation is a reference to a source of information.
It tells the reader of the sentence where specific ideas, facts, or quotations came from.

Include a source citation if the sentence:
- Contains a standalone claim or fact

DO NOT include a source citation if the sentence:
- Is a title or section heading (look for markdown formatting)
- States common knowledge (widely known, undisputed facts)
- Expresses your own analysis, reasoning, or opinion
- Provides general background information that is easily verifiable and widely accepted
- Sets up procedural or instructional content

For each citation produce a summary of the supporting source content:
- The source summary is a summary of the relevant source content. DO NOT summarize the sentence itself.
- Do not mention or quote sentences.
- Each summary must be self-contained and not refer to previous or following sources or sentences.

Example of a good source summary:
- The Amazon rainforest plays a crucial role in regulating the Earth's carbon cycle.

Sources:
{doc_str}

Sentences:
{sent_str}

Produce citations. Be accurate, only produce a citation if there is very strong evidence.
"""

    @staticmethod
    def generate_references_citations_prompt(response: list[str], docs: list[str]) -> str:
        doc_str = "\n".join(docs)
        response_str = "\n".join(response)

        return f"""You are given a set of source sentences labeled <sX>, and a set of response sentences labeled <rX>.

Your task is to produce citations:
- A citation indicates that a fact or claim expressed in a response sentence is supported or substantiated by an equivalent statement of fact contained in a source sentence.
- For each response sentence <rX>, identify source sentence <sX> that explicitly support the fact or claim contained in the response sentence.
- A source sentence <sX> supports a response sentence <rX> if and only if the source sentence contains information that explicitly confirms, directly implies, or provides strong evidence for the response sentence, such that a reasonable reader could see the response sentence as being grounded by the source sentence.

Important rules that must be adhered to:
- Create a citation for a response sentence <rX> if and only if it contains an explicit claim or fact that is not obvious or widely known.
- Cite a source sentence <rX> if and only if it contains an explicit assertion or statement of fact.

NEVER create a citation if the response sentence <rX>:
- is framing or introductory content that summarizes scope or narrative (e.g., sentences about what a report/article discusses, rather than factual content).
- is conversational filler statement that simply acknowledges, agrees, or affirms (e.g., Yes, Certainly!, That's correct), since they do not contain factual content.
- is a question (this cannot express a claim or fact)
- expresses the writers analysis, reasoning, or opinion (not a statement of fact).
- sets up procedural or instructional content.

NEVER create a citation if the source sentence <sX>:
- is not an explicit well formed statement of fact.
- is a question (again this does not express a fact).
- is not well formed or difficult to read (this is of no value to the reader).

Now process this input:

<sources>
{doc_str}
</sources>

<response>
{response_str}
</response>

For each valid response sentence (that express a specific claim or fact) beginning <rX> identify the best quality supporting source sentences <sX>.
Focus on quality and make sure to follow the rules. Prioritize longer source sentences that contain more information.
If a citation is not obviously correct it is better to not produce that citation at all.
Avoid repeated citing of a source, assume the reader has already seen it the first time.

Output format:
{{
    "citations": [
        {{"r": <number>, "s": <number>}}
    ]
}}
"""  # noqa: E501
