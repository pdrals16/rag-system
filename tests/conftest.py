import tempfile

import pytest
from langchain_core.embeddings import Embeddings

from rag.config import Settings


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for testing — no API calls."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] * 8 for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 8


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    return str(tmp_path / "chroma_test")


@pytest.fixture
def settings(tmp_chroma_dir) -> Settings:
    return Settings(
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        chroma_persist_dir=tmp_chroma_dir,
        chunk_size=200,
        chunk_overlap=20,
        top_k=2,
    )
