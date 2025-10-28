# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from pydantic import BaseModel, Field


class ResearchQuery(BaseModel):
    question: str = Field(description="Research question addressing a specific aspect of the research topic.")
    search_query: str = Field(
        description="Optimized standalone research query. Include all available contextual keywords."
    )
    rationale: str = Field(
        description="Brief rationale explaining the importance of the question and how it contributes to the logical flow of the investigation."  # noqa: E501
    )


class ResearchReport(BaseModel):
    query: ResearchQuery
    report: str


class ResearchPlanSchema(BaseModel):
    questions: list[ResearchQuery] = Field(description="A list of search queries that address the research topic.")


class ResearchTopicSchema(BaseModel):
    research_topic: str = Field(
        description="Standalone research topic that clearly and concisely reflects the user's intent."
    )


class LanguageIdentificationSchema(BaseModel):
    language: str = Field(description="The identified language.")
