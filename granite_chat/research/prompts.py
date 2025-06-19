import json
from datetime import UTC, datetime

from beeai_framework.backend import Message
from langchain_core.documents import Document

from granite_chat.research.types import ResearchReport


class ResearchPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def research_plan_prompt(messages: list[Message], max_queries: int = 3) -> str:
        task = messages[0].text
        dynamic_example = ", ".join([f'"sub query {i + 1}"' for i in range(max_queries)])

        return f"""You are a research assistant.
Given a user-defined task, generate a list of specific, actionable sub-queries that will guide research to complete the task.

The sub-queries should:
- Cover the key aspects required to fully understand or complete the task
- Be phrased naturally as search engine queries or research questions
- Be diverse, non-redundant, and ordered logically (e.g. from foundational to advanced)

Task: {task}

The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
Generate exactly {max_queries} sub queries that will guide the research.
You must respond with a list of strings in the following format: [{dynamic_example}].
The response should contain ONLY the list.
"""  # noqa: E501

    @staticmethod
    def research_report_prompt(task: str, docs: list[Document]) -> str:
        json_docs = [
            {
                "doc_id": str(i),
                # "title": d.metadata["title"],
                "content": d.page_content,
            }
            for i, d in enumerate(docs)
        ]

        doc_str = json.dumps(json_docs, indent=4)

        return f"""You are a helpful assistant tasked with generating a comprehensive, informative, and accurate research report.
You are given a task and a set of documents that may contain relevant information. You can use these documents to help formulate your report.

Your report should bee clear and comprehensive and stay aligned with the content and facts of the documents when possible.

Task: {task}

<documents>
{doc_str}
</documents>

Avoid referencing or mentioning "documents" or "the documents", or alluding to their existence in any way when formulating your report.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
You have access to realtime data, you do not have a knowledge cutoff.
"""  # noqa: E501

    @staticmethod
    def final_report_prompt(reports: list[ResearchReport]) -> str:
        reports_str = json.dumps([r.model_dump() for r in reports], indent=4)

        return f"""You are a helpful assistant tasked with generating a final research report.
You are given a set of sub-reports provided by your research assistants.
Your final report should bee clear and comprehensive and align with the sub-reports.

<research_reports>
{reports_str}
</research_reports>

Avoid referencing or mentioning "reports" or "the reports", or alluding to their existence in any way when formulating your final report.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.
"""  # noqa: E501
