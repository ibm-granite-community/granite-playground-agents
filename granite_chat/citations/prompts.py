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
1. A set of reference documents (with their text and source IDs).
2. A set of sentences that may contain ideas or information supported by the reference documents.

Your task is to extract citations. For each sentence, identify which document(s) it is most likely derived from or supported by.

1. For each sentence, return a list of document IDs that best support or justify the information in that sentence.
2. Only cite a document if the sentence clearly restates, quotes, or closely paraphrases a specific passage.
3. Do not infer or guess. If no supporting document is found ignore the sentence.
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
