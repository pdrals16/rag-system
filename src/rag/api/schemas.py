from typing import Any

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    source: str
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")


class QueryResponse(BaseModel):
    answer: str
    question: str
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
