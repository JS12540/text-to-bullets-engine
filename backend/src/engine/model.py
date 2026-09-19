"""Global model and tokenizer management."""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Global model and tokenizer (loaded once at startup)
_model: T5ForConditionalGeneration | None = None
_tokenizer: T5Tokenizer | None = None


def get_model() -> T5ForConditionalGeneration | None:
    """Get the loaded model instance."""
    return _model


def get_tokenizer() -> T5Tokenizer | None:
    """Get the loaded tokenizer instance."""
    return _tokenizer


def load_model() -> None:
    """Load model and tokenizer from disk."""
    global _model, _tokenizer

    try:
        logger.info(
            "startup_begin",
            extra={"model_path": settings.MODEL_PATH},
        )

        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
        logger.info("tokenizer_loaded", extra={"model_path": settings.MODEL_PATH})

        # Load model (INT8 quantized)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.MODEL_PATH,
            device_map="cpu",  # CPU inference only for V1
            torch_dtype=None,  # Use model's native dtype
        )
        _model.eval()
        logger.info("model_loaded", extra={"model_path": settings.MODEL_PATH})

    except Exception as e:
        logger.error("startup_failed", extra={"error": str(e)})
        raise


def unload_model() -> None:
    """Cleanup model and tokenizer on shutdown."""
    global _model, _tokenizer

    logger.info("shutdown_begin")
    if _model is not None:
        del _model
    if _tokenizer is not None:
        del _tokenizer
    logger.info("shutdown_complete")
