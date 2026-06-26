import logging

from fastapi import APIRouter, HTTPException, Request

from rag.api.schemas import QueryRequest, QueryResponse, RetrievedDocument

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    chain = request.app.state.chain
    try:
        result = chain.invoke(body.question)
    except Exception as e:
        logger.exception("Chain invocation failed")
        raise HTTPException(status_code=500, detail="Failed to generate answer") from e
    return QueryResponse(
        answer=result["answer"],
        question=body.question,
        retrieved_documents=[
            RetrievedDocument(
                source=doc.metadata.get("source", "unknown"),
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in result["documents"]
        ],
    )
