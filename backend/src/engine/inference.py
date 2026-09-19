"""Orchestration: validate → create state → prefill → decode loop → stream → cleanup."""
import logging
from typing import AsyncGenerator

from fastapi import HTTPException
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.api.schemas import BulletsRequest, TokenLimitError
from src.config.settings import settings
from src.constants.prompts import SYSTEM_PROMPT, TASK_PROMPT
from src.engine.cache import cleanup_state
from src.engine.decode import decode_step
from src.engine.prefill import run_prefill
from src.engine.state import GenerationState
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def generate_bullets(
    request: BulletsRequest,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
) -> AsyncGenerator[str | dict, None]:
    """
    Full inference pipeline: validate → prefill → decode → stream → cleanup.

    Uses try/finally to ensure cleanup happens even on error or client disconnect.
    """
    state = GenerationState()

    try:
        # Construct full prompt: system + task + user text
        full_text = f"{SYSTEM_PROMPT}\n\n{TASK_PROMPT}{request.text}"

        # Tokenize and validate token count
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            padding=False,
            truncation=False,  # ponytail: don't silently truncate
        )

        num_input_tokens = encoded["input_ids"].shape[1]
        if num_input_tokens > settings.MAX_INPUT_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=TokenLimitError(
                    error_code="TOKEN_LIMIT_EXCEEDED",
                    message=f"Input exceeds maximum token count",
                    actual_tokens=num_input_tokens,
                    max_allowed_tokens=settings.MAX_INPUT_TOKENS,
                ).model_dump(),
            )

        # Initialize state with input
        state.input_ids = encoded["input_ids"]
        state.attention_mask = encoded["attention_mask"]

        logger.info(
            "request_started",
            extra={
                "request_id": state.request_id,
                "input_tokens": num_input_tokens,
            },
        )

        # Prefill: run encoder once
        run_prefill(state, model, tokenizer)

        # Decode loop: generate + yield one token at a time for true streaming
        max_new = settings.MAX_NEW_TOKENS
        last_yielded = 0
        while len(state.generated_token_ids) < max_new and not state.is_eos:
            decode_step(state, model)

            if not state.generated_token_ids:
                continue

            # Decode from full sequence each step so SentencePiece spaces stay correct
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

        # Log completion metrics
        logger.info("request_completed", extra={"request_id": state.request_id, **metrics})

        # Send metrics to client as the final SSE frame
        yield {"metrics": metrics}

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(
            "request_failed",
            extra={
                "request_id": state.request_id,
                "error": str(e),
            },
        )
        raise

    finally:
        # Always cleanup request-specific state
        cleanup_state(state)
