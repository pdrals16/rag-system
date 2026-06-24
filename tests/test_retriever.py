import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.retriever.chroma_retriever import get_retriever


def test_get_retriever_returns_documents(settings, fake_embeddings, tmp_chroma_dir):
    docs = [
        Document(page_content="The capital of France is Paris."),
        Document(page_content="Python is a programming language."),
    ]
    Chroma.from_documents(
        documents=docs,
        embedding=fake_embeddings,
        persist_directory=tmp_chroma_dir,
    )

    # Patch embeddings inside get_retriever by passing a pre-built vectorstore
    vectorstore = Chroma(
        persist_directory=tmp_chroma_dir,
        embedding_function=fake_embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    results = retriever.invoke("capital of France")
    assert len(results) >= 1
    assert any("Paris" in r.page_content for r in results)
