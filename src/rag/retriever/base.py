from typing import Protocol

from langchain_core.documents import Document


class RetrieverProtocol(Protocol):
    """Interface for retrievers. Implement this to swap the vector store backend."""

    def invoke(self, query: str) -> list[Document]: ...
