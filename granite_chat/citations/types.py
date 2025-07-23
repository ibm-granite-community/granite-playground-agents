from pydantic import BaseModel, Field


class Sentence(BaseModel):
    id: str
    text: str
    offset: int
    length: int


class CitationSchema(BaseModel):
    sentence_id: str = Field(description="The id of the sentence.")
    doc_ids: list[str] = Field(description="List of document ids that support this sentence.")


class CitationsSchema(BaseModel):
    citations: list[CitationSchema] = Field(description="Citations")


class Citation(BaseModel):
    url: str | None = None
    title: str | None = None
    context_text: str | None = None
    start_index: int | None = None
    end_index: int | None = None
