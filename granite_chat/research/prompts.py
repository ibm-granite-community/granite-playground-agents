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
Given a topic, generate a list of targeted research questions designed to guide and support in-depth research on the topic.
Expect that the questions will form the basis of an in-depth report so make sure that they are diverse and have a sequential logical narrative.
Tailor the complexity of the research to the given audience.

The research questions should be:
- Clear and concise, ideally a single sentence.
- Cover the key aspects required to fully address the topic.
- Be sufficiently diverse (to prevent overlap and repetition)
- Be Logically connected (i.e. from foundational to advanced, temporally etc.) depending on the research topic.

Audience: General

Topic: {topic}

Generate a maximum of {max_queries} research questions to guide and support in-depth research on the given topic.
Include a rationale with each question indicating how it fits into logical narrative i.e. how it flows from previous questions to add to a coherent report.
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
You are given a collection of documents. Your task is to analyze the documents and produce a detailed set of findings that focus on a given theme.

Rules:
- Identify key findings, insights and evidence that *directly* relate to the given theme.
- Make sure findings are specific, concise, and grounded in the documents.
- If multiple documents mention similar points, consolidate them.
- If documents disagree or provide contrasting perspectives, note the differences.
- Do not include details that are not directly relevant to the theme. Findings should be focused on the given theme only.

<documents>
{doc_str}
</documents>

Theme: {query}

Avoid referencing or mentioning "documents" or "the documents", or alluding to their existence in any way when formulating your report.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.

Output format:
- Present your findings as a set of well-structured paragraphs.
"""  # noqa: E501

    @staticmethod
    def final_report_prompt(topic: str, findings: list[ResearchReport]) -> str:
        findings_str = json.dumps(
            [f.model_dump_json() for f in findings],
            indent=4,
        )

        return f"""You are given a topic and a set of detailed findings (each covering a different theme).
Your task is to write a comprehensive, cohesive, and in-depth report that integrates all findings into a single narrative.

Output format (Markdown):
Title (#): Use the topic as the title, or develop a compelling variation.
Introduction (##)
    - Provide context and set the stage for the report.
    - Avoid references to structure (e.g., “This report will cover…”).
Sections (##)
    - Each set of findings should become a dedicated section with a clear heading.
    - Expand on the findings: explain implications, add interpretation, connect ideas.
    - You may include subsections (###) to deepen or clarify the analysis.
    - Use paragraphs for explanations rather than numbered lists.
Conclusion (##)
    - Summarize the overall insights and significance of the report.
    - End with forward-looking reflections or open questions.

Topic: {topic}

<findings>
{findings_str}
</findings>

Write with detail and length—expand on each theme so that the report reads like a well-researched analysis, not just a summary.
Use a confident, analytical, and narrative tone, making the report engaging and authoritative.
"""  # noqa: E501
