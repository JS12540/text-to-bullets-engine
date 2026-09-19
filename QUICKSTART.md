# Quick Start Guide

## 1. Install Dependencies (uv)

```bash
cd backend
uv sync
```

This installs all dependencies from `pyproject.toml` including FastAPI, PyTorch, transformers, and more.

## 2. Run FastAPI Locally

```bash
cd backend
uv run python -m src.main
```

Server starts on `http://localhost:8000`.

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## 3. Test Health/Ready Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Readiness (verifies model loaded)
curl http://localhost:8000/ready
```

Both should return `200 OK`.

## 4. Test Streaming Endpoint

```bash
curl -X POST http://localhost:8000/v1/bullets/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."}'
```

Expected response: SSE stream of text chunks, e.g.:
```
data: "- "
data: "Machine learning is a subset of artificial intelligence\n"
data: "- "
data: "Systems learn and improve from experience without explicit programming\n"
```

## 5. Build and Run Docker

```bash
cd backend
docker build -t text-to-bullets-backend:0.1.0 .

docker run \
  -p 8000:8000 \
  -e MODEL_PATH=/app/artifacts/text-to-bullets-int8 \
  text-to-bullets-backend:0.1.0
```

Server runs on `http://localhost:8000` inside container, mapped to host port 8000.

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:
```
MODEL_PATH=./artifacts/text-to-bullets-int8
MAX_INPUT_TOKENS=2000
MAX_NEW_TOKENS=512
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

Run with:
```bash
cd backend
uv run python -m src.main
```

## Key Files

- **`backend/src/main.py`** — FastAPI app factory & model loading
- **`backend/src/api/routes.py`** — API endpoints
- **`backend/src/engine/inference.py`** — Full inference pipeline
- **`backend/src/engine/prefill.py`** — Encoder pass
- **`backend/src/engine/decode.py`** — Token generation with KV cache
- **`backend/src/engine/streamer.py`** — SSE streaming
- **`backend/README.md`** — Complete architecture & documentation

## Troubleshooting

### ModuleNotFoundError
Install dependencies: `uv sync`

### Model not loading
Check `MODEL_PATH` in `.env` or environment. Ensure `backend/artifacts/text-to-bullets-int8/` exists.

### Port already in use
Change `PORT` in `.env` or set `PORT=8001 uv run python -m src.main`.

### Docker build fails
Ensure you're in `backend/` directory: `cd backend && docker build ...`

---

See [backend/README.md](./backend/README.md) for full architecture, deployment, and observability.
