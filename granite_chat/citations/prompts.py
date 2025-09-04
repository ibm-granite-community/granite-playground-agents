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
        return f"""You are given:
- A set of source sentences labeled <sX> that serve as evidence.
- A set of response sentences labeled <rX> that are presented to a reader.

Your task:
For each response sentence <rX>, identify source sentences <sX> that explicitly support it.
A source supports a response if it contains information that explicitly confirms, directly implies, or provides strong evidence for the response statement, such that a reasonable reader could see the statement as grounded in the source.
Try to identify high quality source statements that explicitly support response statements. Do not produce too many sources for a single response statement.
If the source statement is a question, is not a well formed coherent statement, or it is difficult to read then ignore it.

DO NOT link to a source if a response statement:
- Does not express a specific verifiable claim or fact. This is important!!
- States common knowledge (widely known, undisputed facts).
- Expresses the writers analysis, reasoning, or opinion.
- Sets up procedural or instructional content.
- is framing or introductory content that summarizes scope or narrative (e.g., sentences about what a report/article discusses, rather than factual content).
- is a conversational filler statement that simply acknowledges, agrees, or affirms (e.g., Yes, Certainly!, That's correct), since they do not contain factual content.

DO NOT link to a source if the source statement:
- Is itself a question.

Here is an example:
Sources:
<sX> Copernicus proposed that the planets, including Earth, revolve around the Sun.
<sY> Galileo observed that Jupiter has moons orbiting around it.
<sZ> Eratosthenes measured Earth's diameter as ~12,742 km using shadows and distances.
Response:
<rW> Hello!.
<rX> The Earth is ~12,742 km in diameter and revolves around the Sun.
<rY> Jupiter has its own moons.
<rZ> In this article we will discuss the solar system.
Output:
{{"citations:[
  {{"r": X, "s": Z}},
  {{"r": X, "s": X}},
  {{"r": Y, "s": Y}}
]}}

Now process this input:

Sources:
{doc_str}

Response:
{response_str}

For valid response statements (that express a specific claim or fact) beginning <rX> identify the highest quality supporting source statements <sX>. Focus on accuracy.
"""  # noqa: E501
