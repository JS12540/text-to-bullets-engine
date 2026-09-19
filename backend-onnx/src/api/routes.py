"""API endpoints. Mirrors backend/src/api/routes.py."""
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.schemas import BulletsRequest
from src.engine.inference import generate_bullets
from src.engine.model import get_decoder_session, get_encoder_session, get_tokenizer

logger = logging.getLogger(__name__)
router = APIRouter()


async def _sse_format(chunks: AsyncGenerator[str | dict, None]) -> AsyncGenerator[str, None]:
    """Wrap text chunks / metrics dicts as SSE `data: {...}\\n\\n` frames the frontend expects."""
    async for chunk in chunks:
        payload = chunk if isinstance(chunk, dict) else {"text": chunk}
        yield f"data: {json.dumps(payload)}\n\n"


@router.get("/health")
async def health_check():
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check():
    encoder = get_encoder_session()
    decoder = get_decoder_session()
    tokenizer = get_tokenizer()
    if encoder is None or decoder is None or tokenizer is None:
        return {"ready": False, "reason": "Model or tokenizer not loaded"}
    return {"ready": True}


@router.post("/v1/bullets/stream")
async def stream_bullets(request: BulletsRequest):
    encoder = get_encoder_session()
    decoder = get_decoder_session()
    tokenizer = get_tokenizer()

    if encoder is None or decoder is None or tokenizer is None:
        return {"error": "Model not loaded"}

    return StreamingResponse(
        _sse_format(generate_bullets(request, encoder, decoder, tokenizer)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
