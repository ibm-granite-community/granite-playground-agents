from pydantic import BaseModel, Field


class Sentence(BaseModel):
    id: str
    text: str
    offset: int
    length: int


class CitationSchema(BaseModel):
    sentence_id: str = Field(description="The id of the sentence.")
    source_id: str = Field(description="The cited source id.")
    source_summary: str = Field(description="A brief summary of the supporting source information.")


class CitationsSchema(BaseModel):
    citations: list[CitationSchema] = Field(description="Citations")


class Citation(BaseModel):
    url: str | None = None
    title: str | None = None
    context_text: str | None = None
    start_index: int | None = None
    end_index: int | None = None
