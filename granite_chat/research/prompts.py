import json
from datetime import UTC, datetime

from langchain_core.documents import Document

from granite_chat.research.types import ResearchReport


class ResearchPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def research_plan_prompt(topic: str, max_queries: int = 3) -> str:
        return f"""You are a research planner.
Given a user-defined topic, you will generate a list of specific, actionable sub-queries/research questions that will serve as the foundation for investigating the topic.

The sub-queries should:
- Cover the key aspects required to fully understand and research the topic.
- Be phrased naturally as queries or research questions.
- Be diverse, non-redundant, and ordered logically (e.g. from foundational to advanced)

Here is the topic: {topic}

The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
Generate a maximum of {max_queries} queries/research questions that will serve as a research plan for the topic.
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
For your assigned topic, review the provided documents and produce a concise research report that summarizes key findings, insights, and relevant information. Focus on extracting the most pertinent details to support a clear understanding of the topic.

Topic: {topic}

<documents>
{doc_str}
</documents>

Avoid referencing or mentioning "documents" or "the documents", or alluding to their existence in any way when formulating your report.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
"""  # noqa: E501

    @staticmethod
    def final_report_prompt(topic: str, plan: list[str], reports: list[ResearchReport]) -> str:
        reports_str = json.dumps([r.model_dump() for r in reports], indent=4)
        plan_str = [f"- {step}\n" for step in plan]

        return f"""You are Granite, a talented researcher.
You are given a topic and a set of interim research reports provided by your assistants. Each assistant is reporting on a sub topic.
Please review the reports submitted by the research assistants and synthesize them into a single, comprehensive final report. Ensure that overlapping information is consolidated, gaps are addressed, and the overall narrative is coherent and aligned with our research objectives.

Topic: {topic}

Research plan:
{plan_str}

<research_reports>
{reports_str}
</research_reports>

Avoid referencing or mentioning "interim reports", "report" or "the reports", or alluding to their existence in any way when formulating your response.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
The title of the report should be {topic} of an appropriate variation thereof.
Do not include references, they will be added later.
"""  # noqa: E501
