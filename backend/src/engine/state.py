"""Per-request generation state management."""
import time
import uuid
from dataclasses import dataclass, field

import torch


@dataclass
class GenerationState:
    """Holds state for a single generation request."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Input
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None

    # Encoder outputs
    encoder_outputs: torch.Tensor | None = None

    # Decoder state
    decoder_input_id: int | None = None
    past_key_values: tuple | None = None

    # Generated tokens
    generated_token_ids: list[int] = field(default_factory=list)

    # Metrics
    prefill_time: float = 0.0
    first_token_time: float = 0.0
    decode_times: list[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    # Flags
    is_eos: bool = False
    is_cancelled: bool = False

    def add_decode_time(self, token_time: float) -> None:
        """Record decode latency for a single token."""
        self.decode_times.append(token_time)

    def total_latency(self) -> float:
        """Total time since request start."""
        return time.time() - self.start_time

    def mean_itl(self) -> float:
        """Mean inter-token latency."""
        if not self.decode_times:
            return 0.0
        return sum(self.decode_times) / len(self.decode_times)

    def ttft(self) -> float:
        """Time to first token."""
        return self.first_token_time - self.start_time if self.first_token_time else 0.0
