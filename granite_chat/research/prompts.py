import json
from datetime import UTC, datetime

from langchain_core.documents import Document

from granite_chat.chat.prompts import ChatPrompts
from granite_chat.research.types import ResearchQuery, ResearchReport


class ResearchPrompts:
    def __init__(self) -> None:
        pass

    @staticmethod
    def research_plan_prompt(topic: str, context: str, max_queries: int = 3) -> str:
        return f"""You are a research planner.
Given a topic and context, generate a list of targeted research questions designed to guide in-depth research on the topic.
Expect that the questions will form the basis of a report so make sure that they are diverse and have a sequential logical narrative.
Tailor the complexity of the research to the given audience. Use the provided context as a guide for which themes to focus on and how to sequence the research.

Each research question should:
- Be clear and concise, ideally a single short sentence.
- Include a standalone search query optimized for use with a search engine. Include details and contextual keywords.

The overall plan should:
- Guide the reader step by step, mirroring natural reasoning.
- Be sufficiently diverse (to prevent overlap and repetition)
- Be Logically connected (i.e. from foundational to advanced, temporally etc.) depending on the topic.

Audience: General

Topic: {topic}

Context: {context}

Generate a maximum of {max_queries} research question to guide and support in-depth research on the given topic.
Include a rationale with each question indicating how it fits into a logical narrative i.e. how it flows from previous questions to add to a coherent report.

Output format:
{{
    "questions":[{{
        "rationale: "Brief rationale explaining the importance of the question and how it contributes to the logical flow of the investigation.",
        "question": "Research question addressing a specific aspect of the research topic.",
        "search_query: "An optimized standalone research query. Include all available contextual keywords."
    }}]
}}
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

Guidelines:
- Identify key findings, insights and evidence *directly* related to the given theme.
- Make sure findings are specific, concise, and grounded in the documents.
- If multiple documents mention similar information then consolidate, do not repeat.
- If documents disagree or provide contrasting perspectives, note the differences.
- Do not include details that are not directly relevant to the theme.
- Do not invent facts that are not contained in the documents.

<documents>
{doc_str}
</documents>

Theme: {query.question} ({query.search_query})

Avoid referencing or mentioning "documents" or "the documents", or alluding to their existence in any way when formulating your report.
The current date is {datetime.now(UTC).strftime("%B %d, %Y")} if required.

Output format:
- Present your findings as a set of well-structured paragraphs. Avoid duplicating information.
"""  # noqa: E501

    @staticmethod
    def final_report_prompt(topic: str, findings: list[ResearchReport]) -> str:
        findings_str = json.dumps(
            [f.model_dump_json() for f in findings],
            indent=4,
        )

        return f"""You are given a topic and a set of detailed findings (each covering a different theme of the topic).
Your task is to write a comprehensive, cohesive, and in-depth report that integrates all findings into a single narrative.

Use markdown output format.

Here is the structure of the report:
# Title: Use the topic as the title, or develop a compelling variation.
## Introduction
    - Provide context and set the stage for the report.
## Divide the report into multiple sections, each with a clear heading that represents a specific topic or finding.
    - Each section heading should reflect the topic it covers.
    - Expand on the findings: explain implications, add interpretation, connect ideas.
    - You may include sub sections (###) to deepen or clarify the analysis.
    - Use paragraphs for explanations rather than numbered lists.
    - Dont duplicate information in multiple sections, try to consolidate.
## Conclusion
    - Summarize the overall insights and significance of the report.
    - End with forward-looking reflections or open questions.

Use # for the report title, ## for Introduction, ## for each main topic section, ### for subsections that expand on a topic, and ## for Conclusion.

{ChatPrompts.math_format_instructions()}

Topic: {topic}

<findings>
{findings_str}
</findings>

DO NOT produce any references or citations!!
Write with detail so that the report reads like a well-researched analysis, not just a summary.
Use a confident, analytical, and narrative tone, making the report engaging and authoritative.
"""  # noqa: E501

    @staticmethod
    def generate_search_queries_prompt(query: ResearchQuery, context: str, max_queries: int = 3) -> str:
        return f"""
Given the following research topic and some background context generate {max_queries} search engine queries that can be used to find information specific to the topic.
The search queries should be clear, concise, and suitable for use in a web search. Include variations to cover possible angles or phrasings. Include all important keywords.

Tips:
- Do not assume or introduce information that is not directly mentioned in the query or context.
- Use the date to augment queries if the user is asking of recent or latest information but be very precise. (Assume the current date is {datetime.now(UTC).strftime("%B %d, %Y")})

Here is an example:
Context: Snack food companies are experimenting with seaweed-based packaging as an alternative to single-use plastic. Researchers are examining its durability, cost, and environmental benefits.
Research Topic: Reducing plastic waste in snack foods.
Search Queries:
["seaweed based biodegradable packaging snack food industry plastic alternative"]

Now here is your task:

Context:
{context}

Research Topic:
{query.question}({query.search_query})

Generate {max_queries} search queries that can be used to find information specific to this topic.
"""  # noqa: E501
