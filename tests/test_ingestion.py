import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest.loaders.factory import load_documents
from ingest.loaders.registry import LOADER_REGISTRY


def test_loader_registry_has_expected_extensions():
    assert ".pdf" in LOADER_REGISTRY
    assert ".txt" in LOADER_REGISTRY
    assert ".md" in LOADER_REGISTRY
    assert ".docx" in LOADER_REGISTRY


def test_load_documents_txt(tmp_path):
    doc = tmp_path / "sample.txt"
    doc.write_text("Hello world. This is a test document.")
    documents = load_documents(str(tmp_path))
    assert len(documents) >= 1
    assert any("Hello world" in d.page_content for d in documents)


def test_load_documents_skips_unsupported(tmp_path):
    (tmp_path / "file.xyz").write_text("unsupported")
    (tmp_path / "file.txt").write_text("supported")
    documents = load_documents(str(tmp_path))
    assert len(documents) == 1


def test_load_documents_raises_for_missing_path():
    with pytest.raises(FileNotFoundError):
        load_documents("/nonexistent/path")


def test_chunking_produces_expected_splits():
    text = " ".join(["word"] * 300)
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_documents(docs)
    assert len(chunks) > 1
