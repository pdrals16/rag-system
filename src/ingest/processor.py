from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from ingest.config import Settings


class DocumentTags(BaseModel):
    """Structured tags extracted from a document by the LLM."""

    subject: Literal["gameplay", "characters", "campaign", "end-game"] = Field(
        description="The primary subject of the document.",
    )


class DocumentProcessor:
    """Enrich documents with an LLM-generated summary and subject tag."""

    def __init__(self, settings: Settings, llm: BaseChatModel | None = None):
        self._llm = llm or ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
        )

    def resume(self, document: Document) -> str:
        prompt = f"Summarize the following text in 2-3 sentences:\n\n{document.page_content}"
        return self._llm.invoke(prompt).content

    def extract_tags(self, document: Document) -> DocumentTags:
        structured = self._llm.with_structured_output(DocumentTags)
        return structured.invoke(
            f"Classify the subject of the following text:\n\n{document.page_content}"
        )
