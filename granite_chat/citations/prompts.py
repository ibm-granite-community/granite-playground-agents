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
1. A set of reference documents (with text and document IDs).
2. A set of sentences that may contain ideas or information supported by the reference documents.

Your task is to extract citations. For each sentence, identify which document(s) the information is derived from or supported by. Include the document id and the verbatim supporting text from the document.

1. For each sentence, return a list of document that explicitly support or justify the information in that sentence.
2. Only cite a document if the sentence is semantically equivalent to, or clearly restates, quotes, or closely paraphrases a specific passage from it.
3. Do not infer or guess. If no supporting document is found skip the sentence.
4. Avoid citing documents that are only vaguely or topically related.

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
- A set of documents, each with a unique `doc_id` and associated text.
- A response that answers a question.

Your task is to regenerate the response, inserting inline citations to the documents using the format [doc_id]. Each claim, sentence or paragraph in the response that is supported by information in one or more documents should be followed by the corresponding [doc_id] citation(s).

Example:
This is a statement that is supported by content from doc with doc_id 1 [1].

Requirements:
- Only include a citation if the information is clearly supported by a document.
- You may cite multiple documents for a single claim if needed (e.g., [1, 2]).

Do not alter the original response beyond the insertion of citations. Do not include any reference list or notes after the response.

Documents:
{doc_str}

Response:
{response}

Regenerate the response, inserting inline citations.
"""  # noqa: E501
