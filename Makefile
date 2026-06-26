PYTHON   := .venv/Scripts/python
UVICORN  := .venv/Scripts/uvicorn
PYTEST   := .venv/Scripts/pytest
INGEST   := .venv/Scripts/rag-ingest

# ── Setup ─────────────────────────────────────────────────────────────────────

.PHONY: install
install:
	python -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

# ── App ───────────────────────────────────────────────────────────────────────

.PHONY: run
run:
	$(UVICORN) rag.api.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: ingest
ingest:
	$(INGEST)

# ── Docker ────────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up:
	docker compose -f .infra/rag/docker-compose.yml up --build -d

.PHONY: docker-down
docker-down:
	docker compose -f .infra/rag/docker-compose.yml down

.PHONY: docker-ingest
docker-ingest:
	docker compose -f .infra/ingestor/docker-compose.yml up --build

# ── Quality ───────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(PYTEST) -v