import json
from datetime import UTC, datetime

from langchain_core.documents import Document

from granite_chat.research.types import ResearchQuery, ResearchReport


class ResearchPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def research_plan_prompt(topic: str, max_queries: int = 3) -> str:
        return f"""You are a research planner. The current date is {datetime.now(UTC).strftime("%B %d, %Y")}.

Given a user-defined topic, generate a list of targeted search queries designed to guide and support in-depth research on the topic.

The queries should be:
- Clear and concise.
- Cover the key aspects required to fully address the topic.
- Be diverse (to prevent overlapping research) and logically distributed (e.g. from foundational to advanced, temporally etc.) depending on context.

Here is the topic: {topic}
Generate {max_queries} search queries to guide and support in-depth research on the topic.
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
Your task is to read a set of documents and produce a clear, detailed answer that addresses a specific query.

<documents>
{doc_str}
</documents>

Query: {query.query}

Avoid referencing or mentioning "documents" or "the documents", or alluding to their existence in any way when formulating your answer.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.

Focus on addressing the query in a comprehensive and detailed manner.
"""  # noqa: E501

    @staticmethod
    def final_report_prompt(topic: str, reports: list[ResearchReport]) -> str:
        reports_str = json.dumps([r.model_dump() for r in reports], indent=4)

        return f"""The current date is {datetime.now(UTC).strftime("%B %d, %Y")}.

You are given a topic along with a set detailed findings each covering a different angle of the subject.
Your task is to review and synthesize this information into a clear and cohesive output.
Ensure the content is cohesive, redundant points are merged, gaps are filled, and the overall narrative flows logically.

Output Format: A structured response including:
- Introduction that sets the stage
- Organized sections that clearly present each major aspect or theme
- Conclusion that summarizes the key insights

Ensure that each section adds new insight or perspective rather than reiterating previous content.
- Use paragraphs rather than numbered lists.

Topic: {topic}

<findings>
{reports_str}
</findings>

The title of the response should be {topic}, or an appropriate variation thereof that maintains the general idea.
"""
