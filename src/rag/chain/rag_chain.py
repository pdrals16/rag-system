from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from rag.chain.prompts import build_prompt
from rag.chain.steps.doc_formatter import format_docs
from rag.config import Settings
from rag.retriever.base import RetrieverProtocol
from rag.retriever.chroma_retriever import get_retriever


def build_chain(
    settings: Settings,
    *,
    retriever: RetrieverProtocol | None = None,
    llm: Runnable | None = None,
    prompt: ChatPromptTemplate | None = None,
) -> Runnable:
    """Build the RAG chain.

    Each dependency is injectable so components can be swapped independently:
    - retriever: change the vector store backend
    - llm: swap to a different model or provider
    - prompt: customize the system instructions
    """
    _retriever = retriever or get_retriever(settings)
    _llm = llm or ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
    )
    _prompt = prompt or build_prompt()

    chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | _prompt
        | _llm
        | StrOutputParser()
    )
    return chain
