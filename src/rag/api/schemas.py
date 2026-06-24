from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")


class QueryResponse(BaseModel):
    answer: str
    question: str
