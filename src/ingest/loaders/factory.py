import logging
from pathlib import Path

from langchain_core.documents import Document

from ingest.loaders.registry import LOADER_REGISTRY

logger = logging.getLogger(__name__)


def load_documents(path: str) -> list[Document]:
    """Load documents from a file or directory.

    Walks directories recursively. Skips unsupported extensions with a warning.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = list(source.rglob("*")) if source.is_dir() else [source]
    documents: list[Document] = []

    for file in files:
        if not file.is_file():
            continue
        loader_cls = LOADER_REGISTRY.get(file.suffix.lower())
        if loader_cls is None:
            logger.warning("Skipping unsupported file type: %s", file)
            continue
        logger.info("Loading %s", file)
        loader = loader_cls(str(file))
        documents.extend(loader.load())

    logger.info("Loaded %d document(s) from %s", len(documents), path)
    return documents
