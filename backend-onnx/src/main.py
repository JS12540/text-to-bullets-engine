"""FastAPI application factory. Mirrors backend/src/main.py."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config.settings import settings
from src.engine.model import load_model, unload_model

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    unload_model()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Text-to-Bullets API (ONNX)",
        description="Convert text to bullet points using INT8 T5 via ONNX Runtime",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()  # Module-level `app` — this is what Vercel's Python runtime auto-detects.

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level=settings.LOG_LEVEL.lower())
