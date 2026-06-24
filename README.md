# RAG System

A modular Retrieval-Augmented Generation (RAG) system built with LangChain and FastAPI.

## Architecture

```
User Question
    │
    ▼
POST /query (FastAPI)
    │
    ▼
RAG Chain (LCEL)
    ├── Retriever ──► ChromaDB (vector search)
    └── Prompt Builder ──► Claude (LLM) ──► Answer
```

## Stack

| Component | Technology |
|---|---|
| LLM | Anthropic Claude (claude-sonnet-4-6) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (local persistence) |
| API | FastAPI + Uvicorn |
| Framework | LangChain (LCEL) |

---

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key — get one at https://console.anthropic.com |
| `OPENAI_API_KEY` | OpenAI API key — used only for embeddings |
| `CHROMA_PERSIST_DIR` | Directory where ChromaDB persists its data (default: `chroma_db`) |
| `LLM_MODEL` | LLM model for generation (default: `claude-sonnet-4-6`) |
| `EMBEDDING_MODEL` | Embedding model (default: `text-embedding-3-small`) |
| `CHUNK_SIZE` | Token chunk size for splitting documents (default: `512`) |
| `CHUNK_OVERLAP` | Overlap between chunks (default: `50`) |
| `TOP_K` | Number of documents retrieved per query (default: `4`) |

---

## Running the Ingest Pipeline

The ingest pipeline reads documents from a local path, chunks them, generates embeddings, and persists them in ChromaDB.

**Supported formats:** PDF, TXT, MD, DOCX

### Step 1 — Add documents

Drop your files into the `data/` folder (create it if it doesn't exist):

```bash
mkdir data
# copy your files into data/
```

### Step 2 — Run the ingestion

After installing with `pip install -e .`, a `rag-ingest` CLI is available:

```bash
rag-ingest --path data/
```

Options:

```
--path        Path to a file or a directory (required)
--log-level   Logging verbosity: DEBUG | INFO | WARNING | ERROR (default: INFO)
```

Examples:

```bash
# Ingest an entire folder
rag-ingest --path data/

# Ingest a single file
rag-ingest --path data/report.pdf

# Ingest with verbose output
rag-ingest --path data/ --log-level DEBUG
```

On success you will see:

```
Done. Indexed 3 document(s) into chroma_db/
```

### Run ingest with Docker

```bash
cp .env.example .env   # fill in your API keys
docker compose -f .infra/ingestor/docker-compose.yml up --build
docker compose -f .infra/ingestor/docker-compose.yml run --rm ingest --path data
```

The container mounts `./data` and `./chroma_db` so the index persists on your host.

---

## Running the RAG API

The API exposes a single `POST /query` endpoint that answers questions using the indexed documents.

### Start the server

```bash
uvicorn rag.api.main:app --reload
```

The server starts on `http://localhost:8000`.

Available endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Ask a question against indexed documents |
| `GET` | `/docs` | Interactive Swagger UI |

### Query the system

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

Response:

```json
{
  "question": "What is this document about?",
  "answer": "..."
}
```

### Run the API with Docker

```bash
cp .env.example .env   # fill in your API keys
docker compose -f .infra/rag/docker-compose.yml up --build
```

The container maps port `8000` and mounts `./data` and `./chroma_db` from your host.

---

## End-to-end flow (local)

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env  # fill ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3. Add documents and ingest
mkdir data && cp my-docs/*.pdf data/
rag-ingest --path data/

# 4. Start the API
uvicorn rag.api.main:app --reload

# 5. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the main findings"}'
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
src/
├── ingest/
│   ├── cli.py                   # rag-ingest CLI entry point
│   ├── config.py                # Settings via pydantic-settings
│   ├── indexer.py               # Chunk → embed → store
│   ├── processor.py             # Document processing logic
│   └── loaders/
│       ├── registry.py          # File extension → loader mapping
│       └── factory.py           # load_documents(path)
└── rag/
    ├── config.py                # Settings via pydantic-settings
    ├── retriever/
    │   ├── base.py              # RetrieverProtocol interface
    │   └── chroma_retriever.py  # ChromaDB retriever
    ├── chain/
    │   ├── steps/
    │   │   └── doc_formatter.py # format_docs() — standalone step
    │   ├── prompts.py           # ChatPromptTemplate
    │   └── rag_chain.py         # build_chain() — injectable LCEL chain
    └── api/
        ├── main.py              # FastAPI app
        ├── routes/
        │   └── query.py         # POST /query
        └── schemas.py           # Request/response models

.infra/
├── rag/
│   ├── Dockerfile
│   └── docker-compose.yml      # Runs the API service
└── ingest/
    ├── Dockerfile
    └── docker-compose.yml      # Runs the ingest job
```

---

## Extending the System

**Add a new document format** — edit `src/ingest/loaders/registry.py`:

```python
LOADER_REGISTRY[".html"] = BSHTMLLoader
```

**Swap the vector store** — implement `RetrieverProtocol` from `rag/retriever/base.py` and pass it to `build_chain()`.

**Add a reranking step** — insert a function between retriever and formatter in `rag_chain.py`:

```python
{"context": retriever | rerank_docs | format_docs, "question": RunnablePassthrough()}
```

**Swap the LLM** — pass a different model to `build_chain()`:

```python
from langchain_openai import ChatOpenAI
chain = build_chain(settings, llm=ChatOpenAI(model="gpt-4o"))
```
