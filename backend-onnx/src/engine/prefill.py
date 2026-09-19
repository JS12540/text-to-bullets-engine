"""Prefill phase: run T5 encoder once per request. Mirrors backend/src/engine/prefill.py."""
import time

import onnxruntime as ort

from src.engine.state import GenerationState


def run_prefill(state: GenerationState, encoder_session: ort.InferenceSession) -> None:
    """Run the ONNX encoder graph once; save last_hidden_state to state."""
    start = time.time()

    outputs = encoder_session.run(
        ["last_hidden_state"],
        {
            "input_ids": state.input_ids,
            "attention_mask": state.attention_mask,
        },
    )
    state.encoder_hidden_states = outputs[0]

    state.prefill_time = time.time() - start
