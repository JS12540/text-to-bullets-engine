"""Per-request memory cleanup. Mirrors backend/src/engine/cache.py."""
from src.engine.state import GenerationState


def cleanup_state(state: GenerationState) -> None:
    """Release all request-specific arrays. Sessions/tokenizer stay loaded."""
    state.past_key_values = None
    state.encoder_hidden_states = None
    state.input_ids = None
    state.attention_mask = None
