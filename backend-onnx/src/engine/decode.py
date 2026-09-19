"""Single decode step against the merged ONNX decoder graph, with real KV cache reuse.

Mirrors backend/src/engine/decode.py's first-step-vs-later-step split, but
driven by onnxruntime.InferenceSession.run() with numpy arrays instead of
torch tensors, and the merged decoder graph's `use_cache_branch` flag instead
of passing/omitting `past_key_values=None`.
"""
import time

import numpy as np
import onnxruntime as ort

from src.config.settings import settings
from src.engine.state import GenerationState

_LAYER_NAMES = range(settings.NUM_LAYERS)


def _empty_kv(batch_size: int, seq_len: int) -> np.ndarray:
    return np.zeros((batch_size, settings.NUM_HEADS, seq_len, settings.HEAD_DIM), dtype=np.float32)


def decode_step(state: GenerationState, decoder_session: ort.InferenceSession) -> int | None:
    """
    Perform one decode step, update state with new token and KV cache.

    First step (state.decoder_input_id is None):
      - decoder_input_id = decoder_start_token_id
      - use_cache_branch = False (graph computes cross-attention KV from
        encoder_hidden_states directly and returns it in `present.*.encoder.*`)
      - past_key_values fed as empty (zero-length sequence) placeholders

    Later steps:
      - Pass only the newest decoder token
      - use_cache_branch = True (graph reuses cached decoder + encoder KV)

    Returns the generated token ID, or None if EOS reached.
    """
    token_start = time.time()
    batch_size = state.input_ids.shape[0]

    is_first_step = state.decoder_input_id is None
    if is_first_step:
        state.decoder_input_id = settings.DECODER_START_TOKEN_ID

    decoder_input_ids = np.array([[state.decoder_input_id]], dtype=np.int64)

    feed = {
        "input_ids": decoder_input_ids,
        "encoder_hidden_states": state.encoder_hidden_states,
        "encoder_attention_mask": state.attention_mask,
        "use_cache_branch": np.array([not is_first_step], dtype=bool),
    }

    for i in _LAYER_NAMES:
        if is_first_step:
            feed[f"past_key_values.{i}.decoder.key"] = _empty_kv(batch_size, 0)
            feed[f"past_key_values.{i}.decoder.value"] = _empty_kv(batch_size, 0)
            feed[f"past_key_values.{i}.encoder.key"] = _empty_kv(batch_size, 0)
            feed[f"past_key_values.{i}.encoder.value"] = _empty_kv(batch_size, 0)
        else:
            feed[f"past_key_values.{i}.decoder.key"] = state.past_key_values[i]["decoder.key"]
            feed[f"past_key_values.{i}.decoder.value"] = state.past_key_values[i]["decoder.value"]
            feed[f"past_key_values.{i}.encoder.key"] = state.past_key_values[i]["encoder.key"]
            feed[f"past_key_values.{i}.encoder.value"] = state.past_key_values[i]["encoder.value"]

    output_names = ["logits"] + [
        f"present.{i}.{part}"
        for i in _LAYER_NAMES
        for part in ("decoder.key", "decoder.value", "encoder.key", "encoder.value")
    ]
    outputs = decoder_session.run(output_names, feed)
    logits = outputs[0]

    # Suppress EOS until MIN_NEW_TOKENS is reached (see settings.py comment —
    # close EOS-vs-continue calls can flip due to quantization imprecision).
    # Capped by input length: a short input (e.g. "My name is Jay Shah") has
    # nothing left to say after a few tokens, so forcing a flat 30 makes the
    # model degenerate into repetition instead of stopping.
    effective_min_new_tokens = min(settings.MIN_NEW_TOKENS, state.input_ids.shape[1])
    if len(state.generated_token_ids) < effective_min_new_tokens:
        logits = logits.copy()
        logits[0, -1, settings.EOS_TOKEN_ID] = -np.inf

    # The merged graph's use_cache_branch=True path never recomputes
    # cross-attention KV — its present.*.encoder.* outputs are bogus fixed-
    # shape placeholders (see export_onnx.py's "Adding a constant output"
    # log). Real encoder KV only comes from the first (use_cache_branch=False)
    # step; keep reusing that forever instead of overwriting it each step.
    new_past_key_values = {}
    idx = 1
    for i in _LAYER_NAMES:
        new_past_key_values[i] = {
            "decoder.key": outputs[idx],
            "decoder.value": outputs[idx + 1],
            "encoder.key": outputs[idx + 2] if is_first_step else state.past_key_values[i]["encoder.key"],
            "encoder.value": outputs[idx + 3] if is_first_step else state.past_key_values[i]["encoder.value"],
        }
        idx += 4
    state.past_key_values = new_past_key_values

    next_token_id = int(np.argmax(logits[0, -1, :]))

    if next_token_id == settings.EOS_TOKEN_ID:
        state.is_eos = True
        return None

    state.generated_token_ids.append(next_token_id)
    token_time = time.time() - token_start
    state.add_decode_time(token_time)

    if not state.first_token_time:
        state.first_token_time = time.time()

    state.decoder_input_id = next_token_id
    return next_token_id
