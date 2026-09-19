"""Application configuration constants."""
import os
from pathlib import Path


class Settings:
    """Application settings."""

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    # Model
    MODEL_PATH: str = str(Path(__file__).parent.parent.parent / "artifacts" / "text-to-bullets-onnx-int8")
    # Local copy, not ../backend/ — this directory must be deployable standalone (see DEPLOYMENT.md).
    TOKENIZER_PATH: str = str(Path(MODEL_PATH) / "tokenizer.json")

    # Remote model weights: set these (Vercel Blob URLs) to keep the .onnx files out of the
    # deployed function bundle — bundling them pushes past Vercel's 225MB Python function cap.
    # Unset locally — falls back to the bundled artifacts/ copy.
    ENCODER_MODEL_URL: str | None = os.environ.get("ENCODER_MODEL_URL")
    DECODER_MODEL_URL: str | None = os.environ.get("DECODER_MODEL_URL")
    MODEL_CACHE_DIR: str = "/tmp/text-to-bullets-model"
    MAX_INPUT_TOKENS: int = 2048
    MAX_NEW_TOKENS: int = 512
    # Suppress EOS below this length. ONNX Runtime's dynamic INT8 quantization
    # is measurably less precise than torchao's calibrated quantization
    # (confirmed: a real EOS-vs-continue decision came down to a 0.5-logit
    # margin on a multi-bullet input), which can flip close EOS/continue
    # decisions and truncate output after just one bullet. Matches the base
    # model's own task_specific_params.summarization.min_length in config.json.
    MIN_NEW_TOKENS: int = 30

    # T5 special tokens (from generation_config.json)
    PAD_TOKEN_ID: int = 0
    EOS_TOKEN_ID: int = 1
    DECODER_START_TOKEN_ID: int = 0
    NUM_LAYERS: int = 6
    NUM_HEADS: int = 8
    HEAD_DIM: int = 64

    # T5's 100 pretraining sentinel tokens (<extra_id_0> .. <extra_id_99>), contiguous in the
    # tokenizer vocab. Never valid in this task's output — greedy decoding never picks them
    # (they're never top-1), but sampling (temperature > 0) can draw them since they carry
    # small nonzero probability. Masked out in decode.py regardless of decoding mode.
    SENTINEL_TOKEN_ID_RANGE: tuple[int, int] = (32000, 32100)  # [start, end)


settings = Settings()
