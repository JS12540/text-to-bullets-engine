"""Global ONNX Runtime session and tokenizer management."""
import logging

import onnxruntime as ort
from tokenizers import Tokenizer

from src.config.settings import settings

logger = logging.getLogger(__name__)

_encoder_session: ort.InferenceSession | None = None
_decoder_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None


def get_encoder_session() -> ort.InferenceSession | None:
    return _encoder_session


def get_decoder_session() -> ort.InferenceSession | None:
    return _decoder_session


def get_tokenizer() -> Tokenizer | None:
    return _tokenizer


def load_model() -> None:
    """Load ONNX Runtime sessions and tokenizer from disk."""
    global _encoder_session, _decoder_session, _tokenizer

    logger.info("startup_begin model_path=%s", settings.MODEL_PATH)

    _tokenizer = Tokenizer.from_file(settings.TOKENIZER_PATH)
    logger.info("tokenizer_loaded")

    _encoder_session = ort.InferenceSession(f"{settings.MODEL_PATH}/encoder_model.onnx")
    _decoder_session = ort.InferenceSession(f"{settings.MODEL_PATH}/decoder_model_merged.onnx")
    logger.info("onnx_sessions_loaded")


def unload_model() -> None:
    global _encoder_session, _decoder_session, _tokenizer

    logger.info("shutdown_begin")
    _encoder_session = None
    _decoder_session = None
    _tokenizer = None
    logger.info("shutdown_complete")
