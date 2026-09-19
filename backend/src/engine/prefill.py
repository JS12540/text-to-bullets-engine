"""Prefill phase: run T5 encoder once per request."""
import time

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.engine.state import GenerationState


def run_prefill(
    state: GenerationState,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
) -> None:
    """
    Run T5 encoder on input tokens, save encoder outputs to state.

    The encoder processes the full input (system prompt + user text)
    and produces contextualized representations. These are reused
    for all decode steps.
    """
    start = time.time()

    with torch.no_grad():
        # Run encoder; decoder_input_ids=None means encoder-only
        encoder_output = model.encoder(
            input_ids=state.input_ids,
            attention_mask=state.attention_mask,
        )
        state.encoder_outputs = encoder_output.last_hidden_state

    state.prefill_time = time.time() - start
