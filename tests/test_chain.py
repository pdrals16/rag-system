import pytest
from langchain_community.llms.fake import FakeListLLM
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag.chain.prompts import build_prompt
from rag.chain.rag_chain import build_chain
from rag.chain.steps.doc_formatter import format_docs


def test_format_docs_joins_content():
    docs = [Document(page_content="First."), Document(page_content="Second.")]
    result = format_docs(docs)
    assert result == "First.\n\nSecond."


def test_format_docs_empty():
    assert format_docs([]) == ""


def test_build_chain_returns_string(settings):
    fake_retriever = RunnableLambda(
        lambda _: [Document(page_content="Paris is the capital of France.")]
    )
    fake_llm = FakeListLLM(responses=["Paris is the answer."])

    chain = build_chain(settings, retriever=fake_retriever, llm=fake_llm)
    result = chain.invoke("What is the capital of France?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_prompt_has_required_variables():
    prompt = build_prompt()
    variables = prompt.input_variables
    assert "context" in variables
    assert "question" in variables
