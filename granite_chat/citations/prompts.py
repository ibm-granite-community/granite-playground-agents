import json

from langchain_core.documents import Document

from granite_chat.citations.types import Sentence


class CitationsPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_citations_prompt(sentences: list[Sentence], docs: list[Document]) -> str:
        sent_str = json.dumps([s.model_dump_json(exclude={"offset", "length"}) for s in sentences], indent=4)
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
