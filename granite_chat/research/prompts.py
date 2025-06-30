import json
from datetime import UTC, datetime

from langchain_core.documents import Document

from granite_chat.research.types import ResearchQuery, ResearchReport


class ResearchPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def research_plan_prompt(topic: str, max_queries: int = 3) -> str:
        return f"""You are a research planner.
Given a user-defined topic, generate a list of specific search queries that serve as the foundation for investigating the topic.

The queries should:
- Clear and concise.
- Cover the key aspects required to fully address the topic.
- Be diverse (to prevent overlapping), non-redundant, and logically distributed (e.g. from foundational to advanced).

Here is the topic: {topic}

The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
Generate a maximum of {max_queries} search queries that will serve as a basis to explore the main topic.
"""  # noqa: E501

    @staticmethod
    def research_report_prompt(query: ResearchQuery, docs: list[Document]) -> str:
        json_docs = [
            {
                "doc_id": str(i),
                "title": d.metadata["title"],
                "content": d.page_content,
            }
            for i, d in enumerate(docs)
        ]

        doc_str = json.dumps(json_docs, indent=4)

        return f"""You are a research assistant.
Your role is to review a set of documents and produce a report focused on a specific query, which is provided.
Your task is to synthesize information relevant to the query from the documents and organize this information clearly and concisely.

<documents>
{doc_str}
</documents>

Query: {query.query}

Avoid referencing or mentioning "documents" or "the documents", or alluding to their existence in any way when formulating your report.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.

Identify and synthesize all content from the documents that relevant to the query. Do not include unrelated material. Focus on addressing the query in a comprehensive manner.
"""  # noqa: E501

    @staticmethod
    def final_report_prompt(topic: str, reports: list[ResearchReport]) -> str:
        reports_str = json.dumps([r.model_dump() for r in reports], indent=4)

        return f"""You are Granite, a talented researcher.
You are given a topic and a set of research reports provided by your assistants. Each report focuses on a different aspect of the same overarching topic.
Your task is to review, and consolidate these aspect-specific reports into a single, comprehensive final report.
Ensure that overlapping information is consolidated, gaps are addressed, and the overall narrative is coherent.

Use a professional, analytical tone suitable for expert readers.

Output Format: A structured final report including:
- An Introduction
- A set of distinct and coherently organized sections that guide the reader
- A Conclusion

Ensure that each section adds new insight or perspective rather than reiterating previous content.

Topic: {topic}

<research_reports>
{reports_str}
</research_reports>

The title of the report should be {topic} of an appropriate variation thereof.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
Avoid referencing or mentioning "interim reports", "report" or "the reports", or alluding to their existence in any way when formulating your response.
Do not include references or citations, they will be added later.
"""  # noqa: E501
