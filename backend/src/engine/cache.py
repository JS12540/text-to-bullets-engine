"""KV cache and memory management."""
import torch

from src.engine.state import GenerationState


def get_cache_size(past_key_values: tuple | None) -> int:
    """Estimate KV cache size in bytes."""
    if not past_key_values:
        return 0

    total = 0
    for layer_cache in past_key_values:
        # Each layer_cache is (key_tensor, value_tensor)
        for tensor in layer_cache:
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()

    return total


def inspect_cache(past_key_values: tuple | None) -> dict:
    """Inspect KV cache structure and sizes."""
    if not past_key_values:
        return {"layers": 0, "size_bytes": 0}

    cache_size = get_cache_size(past_key_values)
    num_layers = len(past_key_values)

    return {
        "layers": num_layers,
        "size_bytes": cache_size,
        "size_mb": cache_size / (1024 * 1024),
    }


def cleanup_state(state: GenerationState) -> None:
    """
    Release all request-specific tensors.

    - past_key_values: decoder KV cache from all decode steps
    - encoder_outputs: encoder contextual representations
    - input_ids/attention_mask: tokenized input
    - generated_token_ids: converted to list (already CPU)

    Model and tokenizer stay loaded in app state.
    """
    # Release KV cache
    if state.past_key_values:
        for layer_cache in state.past_key_values:
            for tensor in layer_cache:
                if tensor is not None and hasattr(tensor, "detach"):
                    del tensor
        state.past_key_values = None

    # Release encoder outputs
    if state.encoder_outputs is not None:
        del state.encoder_outputs
        state.encoder_outputs = None

    # Release input tensors
    if state.input_ids is not None:
        del state.input_ids
        state.input_ids = None

    if state.attention_mask is not None:
        del state.attention_mask
        state.attention_mask = None

    # generated_token_ids is already a list, no cleanup needed
