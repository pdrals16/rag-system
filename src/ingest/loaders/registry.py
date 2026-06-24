from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

# Maps file extensions to LangChain loader classes.
# To support a new format, add one line here — no other changes needed.
LOADER_REGISTRY: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": Docx2txtLoader,
}
