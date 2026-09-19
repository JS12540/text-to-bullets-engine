"""Orchestration: validate -> prefill -> decode loop -> stream -> cleanup.

Mirrors backend/src/engine/inference.py, with real token-by-token streaming
(each token yielded the moment it's decoded, not after the full loop
completes — same fix backend/ needed).
"""
import logging
from typing import AsyncGenerator

import numpy as np
import onnxruntime as ort
from fastapi import HTTPException
from tokenizers import Tokenizer

from src.api.schemas import BulletsRequest, TokenLimitError
from src.config.settings import settings
from src.constants.prompts import SYSTEM_PROMPT, TASK_PROMPT
from src.engine.cache import cleanup_state
from src.engine.decode import decode_step
from src.engine.prefill import run_prefill
from src.engine.state import GenerationState

logger = logging.getLogger(__name__)


async def generate_bullets(
    request: BulletsRequest,
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    tokenizer: Tokenizer,
) -> AsyncGenerator[str | dict, None]:
    """Full inference pipeline: validate -> prefill -> decode -> stream -> cleanup."""
    state = GenerationState()

    try:
        full_text = f"{SYSTEM_PROMPT}\n\n{TASK_PROMPT}{request.text}"

        encoded = tokenizer.encode(full_text)
        num_input_tokens = len(encoded.ids)

        if num_input_tokens > settings.MAX_INPUT_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=TokenLimitError(
                    error_code="TOKEN_LIMIT_EXCEEDED",
                    message="Input exceeds maximum token count",
                    actual_tokens=num_input_tokens,
                    max_allowed_tokens=settings.MAX_INPUT_TOKENS,
                ).model_dump(),
            )

        state.input_ids = np.array([encoded.ids], dtype=np.int64)
        state.attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        logger.info("request_started request_id=%s input_tokens=%d", state.request_id, num_input_tokens)

        run_prefill(state, encoder_session)

        max_new = settings.MAX_NEW_TOKENS
        last_yielded = 0
        while len(state.generated_token_ids) < max_new and not state.is_eos:
            decode_step(state, decoder_session, temperature=request.temperature)

            if not state.generated_token_ids:
                continue

            text = tokenizer.decode(state.generated_token_ids, skip_special_tokens=False)
            new_text = text[last_yielded:]
            last_yielded = len(text)

            if new_text:
                yield new_text.replace("<BULLET>", "\n- ")

        metrics = {
            "input_tokens": num_input_tokens,
            "output_tokens": len(state.generated_token_ids),
            "prefill_time": state.prefill_time,
            "ttft": state.ttft(),
            "mean_itl": state.mean_itl(),
            "total_latency": state.total_latency(),
            "reached_eos": state.is_eos,
        }
        logger.info("request_completed request_id=%s %s", state.request_id, metrics)
        yield {"metrics": metrics}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("request_failed request_id=%s error=%s", state.request_id, str(e))
        raise
    finally:
        cleanup_state(state)
