import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api.routes import query
from rag.chain.rag_chain import build_chain
from rag.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.chain = build_chain(settings)
    logger.info("RAG chain initialized")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG System",
        description="Retrieval-Augmented Generation API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(query.router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
