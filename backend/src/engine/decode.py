"""Single decode step: generate one token at a time."""
import time

import torch
from transformers import T5ForConditionalGeneration

from src.engine.state import GenerationState


def decode_step(
    state: GenerationState,
    model: T5ForConditionalGeneration,
    num_beams: int = 1,
) -> int | None:
    """
    Perform one decode step, update state with new token and KV cache.

    First step (state.decoder_input_id is None):
      - Initialize decoder_input_id to model's decoder start token
      - Attend to full encoder outputs
      - past_key_values is None

    Later steps:
      - Pass only the newest decoder token
      - Reuse past_key_values from cache
      - Attend to cached encoder representations

    Returns the generated token ID, or None if EOS reached.
    """
    token_start = time.time()

    # First decode step: initialize decoder input to start token
    if state.decoder_input_id is None:
        state.decoder_input_id = model.config.decoder_start_token_id or 0

    with torch.no_grad():
        # Prepare decoder input for this step
        # ponytail: always pass as batch (shape [1, 1]) even for single token
        decoder_input_ids = torch.tensor(
            [[state.decoder_input_id]], dtype=torch.long, device=model.device
        )

        # Forward pass
        outputs = model(
            encoder_outputs=(state.encoder_outputs,),
            decoder_input_ids=decoder_input_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
            return_dict=True,
        )

        # Extract logits for greedy decoding
        logits = outputs.logits[:, -1, :]  # Last position, all vocabulary
        next_token_id = torch.argmax(logits, dim=-1).item()

        # Update cache for next step
        state.past_key_values = outputs.past_key_values

        # Check for EOS token
        if next_token_id == model.config.eos_token_id:
            state.is_eos = True
            return None

    # Record the new token and timing
    state.generated_token_ids.append(next_token_id)
    token_time = time.time() - token_start
    state.add_decode_time(token_time)

    # Set TTFT on first generated token
    if not state.first_token_time:
        state.first_token_time = time.time()

    # Update decoder_input_id for next step (only the newest token)
    state.decoder_input_id = next_token_id

    return next_token_id
