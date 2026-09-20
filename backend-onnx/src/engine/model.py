"""Global ONNX Runtime session and tokenizer management."""
import logging
import os
import urllib.request

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


def _resolve_weights_path(url: str | None, filename: str, bundled_path: str) -> str:
    """Download a model file to /tmp on cold start if a remote URL is set, else use the bundled copy.

    Uses BLOB_READ_WRITE_TOKEN as a Bearer token when set — required for private Vercel Blob
    URLs (which carry no expiry, unlike the ~12h signed delegation URLs the dashboard's
    "copy URL" button generates)."""
    if not url:
        return bundled_path

    os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
    cached_path = os.path.join(settings.MODEL_CACHE_DIR, filename)
    if not os.path.exists(cached_path):
        logger.info("downloading_model_weights file=%s url=%s", filename, url)
        request = urllib.request.Request(url)
        bearer_token = settings.VERCEL_OIDC_TOKEN or settings.BLOB_READ_WRITE_TOKEN
        if bearer_token:
            request.add_header("Authorization", f"Bearer {bearer_token}")
        with urllib.request.urlopen(request) as response, open(cached_path, "wb") as f:
            f.write(response.read())
        logger.info("downloaded_model_weights file=%s", filename)
    return cached_path


def load_model() -> None:
    """Load ONNX Runtime sessions and tokenizer from disk."""
    global _encoder_session, _decoder_session, _tokenizer

    logger.info("startup_begin model_path=%s", settings.MODEL_PATH)

    _tokenizer = Tokenizer.from_file(settings.TOKENIZER_PATH)
    logger.info("tokenizer_loaded")

    encoder_path = _resolve_weights_path(
        settings.ENCODER_MODEL_URL, "encoder_model.onnx", f"{settings.MODEL_PATH}/encoder_model.onnx"
    )
    decoder_path = _resolve_weights_path(
        settings.DECODER_MODEL_URL, "decoder_model_merged.onnx", f"{settings.MODEL_PATH}/decoder_model_merged.onnx"
    )
    _encoder_session = ort.InferenceSession(encoder_path)
    _decoder_session = ort.InferenceSession(decoder_path)
    logger.info("onnx_sessions_loaded")


def unload_model() -> None:
    global _encoder_session, _decoder_session, _tokenizer

    logger.info("shutdown_begin")
    _encoder_session = None
    _decoder_session = None
    _tokenizer = None
    logger.info("shutdown_complete")
