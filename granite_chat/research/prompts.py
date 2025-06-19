import json
from datetime import UTC, datetime

from langchain_core.documents import Document

from granite_chat.research.types import ResearchReport


class ResearchPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def research_plan_prompt(topic: str, max_queries: int = 3) -> str:
        dynamic_example = ", ".join([f'"sub query {i + 1}"' for i in range(max_queries)])

        return f"""You are a research planner.
Given a user-defined topic, you will generate a list of specific, actionable sub-queries that will serve as the foundation for investigating the topic.

The sub-queries should:
- Cover the key aspects required to fully understand and research the topic.
- Be phrased naturally as search engine queries or research questions.
- Be diverse, non-redundant, and ordered logically (e.g. from foundational to advanced)

Here is the topic: {topic}

The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
Generate exactly {max_queries} sub queries that will guide the research.
You must respond with a list of strings in the following format: [{dynamic_example}].
The response should contain ONLY the list.
"""  # noqa: E501

    @staticmethod
    def research_report_prompt(topic: str, docs: list[Document]) -> str:
        json_docs = [
            {
                "doc_id": str(i),
                # "title": d.metadata["title"],
                "content": d.page_content,
            }
            for i, d in enumerate(docs)
        ]

        doc_str = json.dumps(json_docs, indent=4)

        return f"""You are a meticulous research assistant.
You are given a topic and a set of documents that contain information pertinent to the topic.
Your job is to produce a research report on the topic using the provided documents.
Your report should be clear and comprehensive and stay aligned with the content and facts of the documents when possible.

Topic: {topic}

<documents>
{doc_str}
</documents>

Avoid referencing or mentioning "documents" or "the documents", or alluding to their existence in any way when formulating your report.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
You have access to realtime data, you do not have a knowledge cutoff.
"""  # noqa: E501

    @staticmethod
    def final_report_prompt(topic: str, reports: list[ResearchReport]) -> str:
        reports_str = json.dumps([r.model_dump() for r in reports], indent=4)

        return f"""You are Granite, a talented researcher leading a team of research assistants.
You are given a topic and a set of intermediate research reports provided by your assistants.
Each assistant is reporting on a related sub topic.
Compile a coherent report based on the intermediate research reports submitted by your research assistants."
Your report should be comprehensive and align with the sub-reports.

Topic: {topic}

<research_reports>
{reports_str}
</research_reports>

Avoid referencing or mentioning "interim reports" or "the reports", or alluding to their existence in any way when formulating your response.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
The title of the report should be {topic} of an appropriate variation thereof.
Do not include references, they will be added later.
"""  # noqa: E501
