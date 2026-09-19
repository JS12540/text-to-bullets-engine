# Text-to-Bullets Backend

FastAPI inference server for streaming bullet-point generation using an INT8-quantized T5 model.

---

## Quick Start

### 1. Install Dependencies

```bash
cd backend
uv sync
```

This creates a virtual environment and installs all dependencies (FastAPI, PyTorch, transformers, etc.).

### 2. Run the Server

```bash
cd backend
uv run python -m src.main
```

Server starts on **http://localhost:8000**

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 3. Test Health & Ready Endpoints

```bash
# Health check
curl http://localhost:8000/health
→ {"status": "ok"}

# Readiness (verifies model loaded)
curl http://localhost:8000/ready
→ {"ready": true}
```

### 4. Test Streaming Endpoint

```bash
curl -X POST http://localhost:8000/v1/bullets/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is transforming AI. Deep learning powers modern systems."}'
```

Expected: Server-Sent Events stream of text chunks

```
data: "- Machine learning is transforming AI\n"
data: "- Deep learning powers modern systems\n"
```

### 5. Verify Full Stack (Backend + Frontend)

**Terminal 1: Backend**
```bash
cd backend
uv run python -m src.main
# Running on http://localhost:8000
```

**Terminal 2: Frontend**
```bash
cd frontend
npm run dev
# Running on http://localhost:3000
```

**Browser: Open http://localhost:3000**
- Paste text → Click "Generate Bullets" → Watch streaming output

---

## Configuration

All settings are hardcoded constants in `src/config/settings.py`:

```python
class Settings:
    # Server
    HOST = "0.0.0.0"
    PORT = 8000
    LOG_LEVEL = "INFO"

    # Model
    MODEL_PATH = "./artifacts/text-to-bullets-int8"
    MAX_INPUT_TOKENS = 2048
    MAX_NEW_TOKENS = 512
```

To change these, edit `src/config/settings.py` and restart the server.

---

## Architecture

```
Request Flow:
  POST /v1/bullets/stream
    ↓
  [Validation] → Pydantic models, token count check
    ↓
  [Prefill] → Run T5 encoder once, save encoder outputs
    ↓
  [Decode Loop] → Generate tokens one at a time, reuse KV cache
    ↓
  [Streaming] → Convert token IDs to text, yield chunks via SSE
    ↓
  [Cleanup] → Release per-request tensors (KV cache, encoder outputs, input IDs)
```

### Key Components

- **`src/config/settings.py`** — Pydantic Settings for environment configuration (model path, token limits, server host/port).
- **`src/constants/prompts.py`** — System and task prompts (stored server-side only).
- **`src/api/routes.py`** — FastAPI route handlers (`/health`, `/ready`, `/v1/bullets/stream`).
- **`src/api/schemas.py`** — Pydantic request/response schemas and error models.

### Inference Engine

- **`src/engine/state.py`** — `GenerationState` dataclass holding per-request tensors, metrics, and flags.
- **`src/engine/prefill.py`** — Encoder pass: run T5 encoder once per request, save outputs for decoder reuse.
- **`src/engine/decode.py`** — Single decoder step: greedy token generation with KV cache.
- **`src/engine/cache.py`** — KV cache inspection and per-request memory cleanup.
- **`src/engine/streamer.py`** — Async generator for token-by-token streaming with SentencePiece handling.
- **`src/engine/inference.py`** — Orchestration layer: validate → prefill → decode → stream → cleanup (with try/finally).

### Utilities

- **`src/utils/logging.py`** — Structured JSON logging with request ID context.
- **`src/utils/context.py`** — Context variables for request tracking.
- **`src/utils/memory.py`** — Memory usage stats and GC helpers.

---

## Startup Flow

1. **Application lifespan context manager** (`src/main.py`)
   - Called at server startup
   - Loads tokenizer from `MODEL_PATH`
   - Loads T5 model (INT8, CPU)
   - Logs successful initialization

2. **Model stays in memory** for entire server lifetime
3. **Per-request cleanup** removes only request-specific tensors (KV cache, input IDs, encoder outputs)

---

## Request Lifecycle

### 1. Validation
- **Input**: `{ "text": "..." }`
- Pydantic validates non-empty string
- Constructs full prompt: `SYSTEM_PROMPT + TASK_PROMPT + user.text`
- Tokenizes without truncation
- Checks token count against `MAX_INPUT_TOKENS` (default 2048)
- Returns **400 with structured error** if exceeded (includes actual vs. max tokens)

### 2. Prefill
- `run_prefill(state, model, tokenizer)`
- Runs T5 encoder once on full input (system + task + user text)
- Encoder outputs shape: `[1, input_seq_len, hidden_dim]`
- Saved to `state.encoder_outputs` for decoder reuse
- Records prefill latency

### 3. Decode Loop
- **First decode step:**
  - `state.decoder_input_id = model.config.decoder_start_token_id`
  - Attend to full encoder outputs
  - `past_key_values = None`
  - Generate first token

- **Subsequent steps:**
  - Pass only the newest decoder token (shape `[1, 1]`)
  - Reuse `past_key_values` from previous step
  - Attend to cached encoder representations
  - Generate next token

- **Stopping conditions:**
  - EOS token reached → `state.is_eos = True`
  - Generated token count reaches `MAX_NEW_TOKENS` (default 512)

### 4. Streaming
- `stream_tokens(state, tokenizer)` async generator
- Decodes token IDs incrementally to preserve SentencePiece spaces
- Replaces `<BULLET>` tokens with `\n- `
- Yields text chunks token-by-token via SSE

### 5. Cleanup
- `cleanup_state(state)` releases:
  - `past_key_values` (decoder KV cache from all steps)
  - `encoder_outputs` (contextual representations)
  - `input_ids`, `attention_mask` (tokenized input)
- **Model and tokenizer remain loaded** for next request

---

## Per-Request KV Cache

**T5 Decoder KV Cache Structure:**
- `past_key_values = tuple of (num_layers,)` tuples
- Each layer: `(key, value)` tensors
- Shape per layer: `[batch_size=1, num_heads, seq_len, head_dim]`
- Grows linearly with decode steps

**Usage in this implementation:**
1. First decode step: `past_key_values=None`
2. After first step: capture `outputs.past_key_values` from model
3. Subsequent steps: pass cached KV and only new decoder token
4. Cleanup: release entire `past_key_values` tuple after streaming

---

## Observability

### Structured Logging
- **JSON format** with request ID, level, message, and extra context
- Log level: configurable via `LOG_LEVEL` env var

### Metrics Per Request
- **`prefill_time`** — Encoder latency (ms)
- **`ttft`** (Time-to-First-Token) — Wall clock from request start to first generated token (ms)
- **`mean_itl`** (Mean Inter-Token Latency) — Average latency per decode step (ms)
- **`total_latency`** — Complete request duration (ms)
- **`input_tokens`** — Tokenized system + task + user text
- **`output_tokens`** — Number of generated tokens
- **`reached_eos`** — Boolean, whether generation hit EOS token

### Log Events
- `startup_begin`, `tokenizer_loaded`, `model_loaded` at server startup
- `request_started` with request ID and input token count
- `request_completed` with full metrics
- `request_failed` with error context
- `shutdown_begin`, `shutdown_complete` at server shutdown

---

## Configuration

All configuration is hardcoded in `src/config/settings.py`:

```python
class Settings:
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Model
    MODEL_PATH: str = "./artifacts/text-to-bullets-int8"
    MAX_INPUT_TOKENS: int = 2048
    MAX_NEW_TOKENS: int = 512
```

To change any setting:
1. Edit `src/config/settings.py`
2. Change the constant value
3. Restart the server

No environment variables are needed.

---

## API Examples

### Health Check
```bash
curl http://localhost:8000/health
```
Response:
```json
{"status": "ok"}
```

### Readiness
```bash
curl http://localhost:8000/ready
```
Response:
```json
{"ready": true}
```

### Streaming Bullets (curl)
```bash
curl -X POST http://localhost:8000/v1/bullets/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog. This is a test."}'
```

Response: SSE stream
```
data: "- "
data: "The quick brown fox jumps over the lazy dog.\n"
data: "- "
data: "This is a test.\n"
```

### Token Limit Exceeded (400)
Request with >2048 tokens returns:
```json
{
  "detail": {
    "error_code": "TOKEN_LIMIT_EXCEEDED",
    "message": "Input exceeds maximum token count",
    "actual_tokens": 2567,
    "max_allowed_tokens": 2048,
    "details": null
  }
}
```

---

## Local Development

### Install Dependencies
```bash
cd backend
uv sync
```

### Run Server
```bash
uv run python -m src.main
```
Server listens on `http://localhost:8000`

### Test Streaming Endpoint
```bash
curl -X POST http://localhost:8000/v1/bullets/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is changing the world. AI is becoming smarter."}'
```

---

## Docker

### Build Image
```bash
cd backend
docker build -t text-to-bullets-backend:0.1.0 .
```

### Run Container
```bash
docker run -p 8000:8000 text-to-bullets-backend:0.1.0
```

Configuration is hardcoded, so no environment variables are needed.

### Health Check
```bash
curl http://localhost:8000/health
```

---

## Testing the Full Pipeline

```bash
# 1. Start server
uv run python -m src.main &
sleep 2

# 2. Check readiness
curl http://localhost:8000/ready

# 3. Stream bullets
curl -X POST http://localhost:8000/v1/bullets/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Your input text here"}'

# 4. Kill server
pkill -f "python -m src.main"
```

---

## Notes on Version 1

- **CPU inference only** (no CUDA)
- **Greedy decoding** (no beam search or sampling)
- **No batching** (single request at a time)
- **No authentication** (development mode)
- **Per-request cleanup** ensures memory remains stable across requests
- **No continuous batching or scheduler** (simple sequential dispatch)

## Future Enhancements

- Beam search or nucleus sampling
- Request batching / continuous batching
- Redis for distributed caching
- Distributed tracing (OpenTelemetry)
- Authentication & rate limiting
- Database for request history
