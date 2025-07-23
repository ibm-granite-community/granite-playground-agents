import json

from langchain_core.documents import Document

from granite_chat.citations.types import Sentence


class CitationsPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_citations_prompt(sentences: list[Sentence], docs: list[Document]) -> str:
        sent_str = json.dumps([s.model_dump_json() for s in sentences], indent=4)
        json_docs = [{"doc_id": str(i), "content": d.page_content} for i, d in enumerate(docs)]
        doc_str = json.dumps(json_docs, indent=4)

        return f"""You are given:
1. A set of reference documents (with doc_id and text content).
2. A set of sentences that may contain ideas or information supported by the reference documents.

Your task is to produce citations. A citation is a reference to a source of information. It tells the reader where specific ideas, facts, or quotations in a piece of writing came from.

Include a citation if the sentence:
- Contains a standalone claim, fact, or idea

Do not include a citation if the sentence:
- States common knowledge (widely known, undisputed facts)
- Expresses your own analysis, reasoning, or opinion
- Provides general background that's easily verifiable and widely accepted
- Sets up procedural or instructional content

Documents:
{doc_str}

Sentences:
{sent_str}

Output format:
[
   {{
        "sentence_id": "<sentence_id_1>",
        "citations": ["<doc_id_1>", "<doc_id_2>"]
    }},
]
"""  # noqa: E501

    @staticmethod
    def generate_response_with_citations_prompt(response: str, docs: list[Document]) -> str:
        json_docs = [{"doc_id": str(i), "content": d.page_content} for i, d in enumerate(docs)]
        doc_str = json.dumps(json_docs, indent=4)

        return f"""You are given:
- A set of documents, each with a unique `doc_id` and content text.
- A response that answers a question.

Your task is to add citations to the response
- Insert inline citations to the documents using the format [doc_id].
- Each substantial claim in the response that is supported by information in one or more documents should be followed by a corresponding [doc_id] citation(s).

Example:
The color of the sky changes because of Rayleigh scattering [1].

Requirements:
- Only include a citation if the information is clearly supported by the document.
- You may cite multiple documents for a single claim if needed (e.g., [1, 2]).

Do not alter the original response in any way beyond the insertion of citation annotations.
Do not include any reference list or notes after the response.

Documents:
{doc_str}

Response:
{response}

Annotate the response, inserting inline citations.
"""  # noqa: E501
