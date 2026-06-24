import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest.config import Settings
from ingest.processor import DocumentProcessor

logger = logging.getLogger(__name__)


def build_index(documents: list[Document], settings: Settings) -> Chroma:
    processor = DocumentProcessor(settings)
    for doc in documents:  # TODO: batch/cap LLM calls for large multi-page files
        if settings.enable_summary:
            doc.metadata["summary"] = processor.resume(doc)
        if settings.enable_subject:
            doc.metadata["subject"] = processor.extract_tags(doc).subject

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunk(s)", len(chunks))

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    logger.info("Index built and persisted to %s", settings.chroma_persist_dir)
    return vectorstore
