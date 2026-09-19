"""Application configuration constants."""
from pathlib import Path


class Settings:
    """Application settings."""

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Model
    MODEL_PATH: str = str(Path(__file__).parent.parent.parent / "artifacts" / "text-to-bullets-int8")
    MAX_INPUT_TOKENS: int = 2048  # model trained with this context length
    MAX_NEW_TOKENS: int = 512


settings = Settings()
