import logging

from fastapi import APIRouter, HTTPException, Request

from rag.api.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    chain = request.app.state.chain
    try:
        answer = chain.invoke(body.question)
    except Exception as e:
        logger.exception("Chain invocation failed")
        raise HTTPException(status_code=500, detail="Failed to generate answer") from e
    return QueryResponse(answer=answer, question=body.question)
