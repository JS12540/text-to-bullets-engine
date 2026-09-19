"""API endpoints."""
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import BulletsRequest
from src.engine.inference import generate_bullets
from src.engine.model import get_model, get_tokenizer
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


async def _sse_format(chunks: AsyncGenerator[str | dict, None]) -> AsyncGenerator[str, None]:
    """Wrap text chunks / metrics dicts as SSE `data: {...}\\n\\n` frames the frontend expects."""
    async for chunk in chunks:
        payload = chunk if isinstance(chunk, dict) else {"text": chunk}
        yield f"data: {json.dumps(payload)}\n\n"


@router.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check():
    """Readiness check: verify model is loaded."""
    try:
        model = get_model()
        tokenizer = get_tokenizer()
        if model is None or tokenizer is None:
            return {"ready": False, "reason": "Model or tokenizer not loaded"}
        return {"ready": True}
    except Exception as e:
        logger.error("readiness_check_failed", extra={"error": str(e)})
        return {"ready": False, "reason": str(e)}


@router.post("/v1/bullets/stream")
async def stream_bullets(request: BulletsRequest):
    """
    Stream bullet points from input text.

    Request body:
    {
        "text": "Your text here..."
    }

    Response: Server-Sent Events stream of text chunks.
    """
    model = get_model()
    tokenizer = get_tokenizer()

    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    # Generate streaming response
    return StreamingResponse(
        _sse_format(generate_bullets(request, model, tokenizer)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
