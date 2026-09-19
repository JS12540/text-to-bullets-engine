# Text-to-Bullets

> Transform long, unstructured text into concise, factual bullet points — using a small, specialized T5 model I trained, quantized, and served two different ways.

## The Story, Short Version

I started with a question: could a small, specialized model beat a much larger general-purpose LLM at one narrow task? I tested that against BitNet, Qwen3, and SmolLM; fine-tuned a decoder-only model; hit its architectural limits; moved to a T5 encoder-decoder; quantized it eight different ways; benchmarked it against native Rust; built a streaming FastAPI backend with manual prefill/decode/KV-cache management; built a Next.js frontend to actually use it; and then, later, converted the whole thing to ONNX Runtime to see if it could run somewhere leaner — which came with its own real bugs and real trade-offs.

The full, chapter-by-chapter account of all of that — every table, every dead end, every number I actually measured — lives in **[Experiment_README.md](./Experiment_README.md)**. This README is the practical entry point: what's in this repo, what each folder is for, and how to run it. If you want the *why* behind every decision, that's where it is.

**The original, unquantized, fine-tuned T5 model is public on Hugging Face:** [JayShah07/falconai-text-bullet-t5](https://huggingface.co/JayShah07/falconai-text-bullet-t5). The `backend/` and `backend-onnx/` folders in this repo both serve *derived* (quantized) versions of that checkpoint — see below for exactly how.

---

## Project Structure

```text
text-to-bullets-engine/
├── backend/                  # Production server: PyTorch + TorchAO INT8
├── backend-onnx/             # Experimental server: ONNX Runtime, no torch/transformers at inference time
├── frontend/                 # Next.js web UI — talks to whichever backend you point it at
├── notebooks/                # The full training/evaluation journey, as Jupyter notebooks (01 → 09)
├── datasets/                 # Training data and every raw evaluation result behind Experiment_README.md's tables
├── Experiment_README.md      # The full story: model selection → fine-tuning → quantization → serving → ONNX
├── README.md                 # This file
└── ... (build notes, checklists — see "Other files" below)
```

---

## What `backend/` and `backend-onnx/` Actually Stand For

These are **two independent servers for the same task**, not a client/server pair or a v1/v2 relationship. Point the frontend at either one and it behaves identically from the outside — same API, same streaming format. They differ underneath:

| | `backend/` | `backend-onnx/` |
|---|---|---|
| **Status** | Production | Experimental — validated locally, not load-tested |
| **Runtime** | PyTorch + `transformers` + TorchAO | ONNX Runtime + standalone `tokenizers` (no `torch`/`transformers` at inference time) |
| **Quantization** | TorchAO INT8 weight-only (per-row, calibrated) | ONNX Runtime dynamic INT8 (per-op, uncalibrated) |
| **Why it exists** | The model I actually trained, quantized, and shipped | Built afterward to test whether this could deploy somewhere size-constrained, like a serverless platform |
| **Dependency footprint** | ~300MB (`torch` + `transformers`) | ~100MB (`onnxruntime` + `tokenizers`) |
| **Model artifact size** | 76.1MB | 193MB (larger — ONNX's dynamic quantizer doesn't touch embedding tables; full story in Experiment_README.md §27.4) |
| **Accuracy vs. the trained model** | Reference | Exact match after fixing two real bugs I found along the way (§27.6, §27.7) |

Short version: `backend/` is what you should actually deploy. `backend-onnx/` is what I built to answer a specific question about serverless deployment size, and the honest answer — with the real numbers, real bugs, and real fixes — is [Section 27 of Experiment_README.md](./Experiment_README.md#27-the-onnx-experiment-torch-vs-onnx-runtime).

---

## Components

### `backend/` — Production Server

**FastAPI + PyTorch inference server, running a TorchAO INT8-quantized T5 model.**

```text
backend/
├── src/
│   ├── main.py                # FastAPI app factory + lifespan (model load/unload)
│   ├── api/                   # Routes (/health, /ready, /v1/bullets/stream) & Pydantic schemas
│   ├── engine/
│   │   ├── prefill.py         # T5 encoder pass, run once per request
│   │   ├── decode.py          # Token-by-token generation with KV cache reuse
│   │   ├── cache.py           # Per-request cleanup
│   │   └── inference.py       # Orchestration: validate → prefill → decode → stream → cleanup
│   ├── config/                # Settings (hardcoded constants, no .env)
│   ├── constants/              # System/task prompts (server-side only)
│   └── utils/                  # Logging, request context
├── artifacts/text-to-bullets-int8/   # The quantized model (derived from the HF checkpoint above)
├── pyproject.toml / requirements.txt
└── Dockerfile
```

**Quick start:**
```bash
cd backend
uv sync
uv run python -m src.main
# or, with hot reload during development:
uv run uvicorn src.main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

Full details: [backend/README.md](./backend/README.md)

### `backend-onnx/` — Experimental ONNX Server

**The same task, served via raw `onnxruntime` sessions instead of PyTorch — no `torch` or `transformers` needed to run it.**

```text
backend-onnx/
├── src/                        # Mirrors backend/'s structure — same prefill/decode/cache split,
│                                # reimplemented against onnxruntime.InferenceSession + numpy
├── scripts/                    # One-off conversion + accuracy-validation scripts (not the server)
│   ├── dequantize.py           # torchao checkpoint → plain float (phase 1)
│   ├── export_onnx.py          # plain checkpoint → ONNX → ONNX Runtime INT8 (phase 2)
│   └── compare_outputs.py      # torch vs onnx output diff
├── artifacts/text-to-bullets-onnx-int8/   # encoder_model.onnx + decoder_model_merged.onnx
├── .venv-serve/                # The only venv needed to actually run the server
├── pyproject.toml / requirements.txt          # Lean runtime deps
└── requirements-dequant.txt / requirements-export.txt  # Conversion-only deps (separate venvs — see README)
```

**Quick start:**
```bash
cd backend-onnx
.venv-serve/bin/uvicorn src.main:create_app --factory --port 8001 --host 0.0.0.0 --reload
```

Full details, including the conversion process and every bug I hit building this: [backend-onnx/README.md](./backend-onnx/README.md)

### `frontend/` — Web UI

**Next.js + TypeScript + Tailwind.** Paste text in, watch bullets stream out, see real inference metrics (TTFT, mean inter-token latency, prefill time, total latency) update live.

```text
frontend/
├── app/                # Next.js App Router pages
├── components/         # Input card, output card, technical-details panel, etc.
├── lib/                # API client, SSE stream parser, types
└── public/
```

**Quick start:**
```bash
cd frontend
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local   # or :8001 for backend-onnx
npm install
npm run dev
```

Open `http://localhost:3000`. Works identically against either backend — just change the URL.

### `notebooks/` — The Training Journey

Nine Jupyter notebooks, in order, covering the full path from initial model comparison to the final quantized artifact:

| # | Notebook | What it covers |
|---|---|---|
| 01 | `Testing_existing_models_for_parpagaroh_to_bullet` | Zero-shot baseline comparison (BitNet, Qwen3, SmolLM) |
| 02 | `BitNet_for_parpagaroh_to_bullet` | BitNet-specific evaluation |
| 03 | `Instruction_Finetuning_for_Parapgraph_to_bullet_training_Smol135_model` | SmolLM2-135M SFT fine-tuning |
| 04 | `falconsai_t5_final_bullet_training` | Final T5 encoder-decoder fine-tuning |
| 05 | `falconai_t5_cpu_quantization_benchmark` | CPU quantization comparison (FP32/FP16/BF16/INT8/INT4) |
| 06 | `falconai_t5_gpu_quantization_A_to_Z` | Full A-to-Z quantization sweep |
| 07 | `rust_candle_t5_fp32_native_5_text_benchmark` | Native Rust/Candle inference experiment |
| 08 | `t5_torchao_int8_cpu_streaming_5_texts` | TorchAO INT8 streaming validation |
| 09 | `prepare_torchao_int8_model` | Final artifact preparation (the checkpoint `backend/` actually serves) |

Each notebook's results are narrated in [Experiment_README.md](./Experiment_README.md) — the notebooks are the raw work, the README is the story.

### `datasets/` — Training Data & Raw Evaluation Results

```text
datasets/
├── training_bullet_dataset_10000 (2).csv   # Fine-tuning dataset
├── synthetic_evaluation_500.csv            # 500-example evaluation set (Experiment_README §9)
├── output_with_bullet_points.csv           # Generated outputs for review
├── A_to_Z_notebook_results/                # Raw per-example CSVs for every quantization method
│   └── (FP32, FP16, BF16, BnB INT8, BnB NF4 INT4, Quanto, TorchAO, ...)
└── CPU results/                            # Raw per-example CSVs for the CPU-specific comparison
    └── (FP32, BnB INT8 BF16, BnB NF4 INT4)
```

Every quantization table in [Experiment_README.md](./Experiment_README.md) (Sections 11–13) was computed from these files.

### Other files

- **[BUILD_CHECKLIST.md](./BUILD_CHECKLIST.md)**, **[QUICKSTART.md](./QUICKSTART.md)**, **[FRONTEND_SETUP.md](./FRONTEND_SETUP.md)**, **[FRONTEND_BUILD_COMPLETE.txt](./FRONTEND_BUILD_COMPLETE.txt)**, **[COMMANDS.txt](./COMMANDS.txt)** — working notes from building the backend and frontend.
- **Frontend.png** — a screenshot of the UI.

---

## Running the Full Stack

```bash
# Terminal 1 — backend (pick one)
cd backend && uv run python -m src.main                                   # production, port 8000
# or
cd backend-onnx && .venv-serve/bin/uvicorn src.main:create_app --factory --port 8001 --host 0.0.0.0

# Terminal 2 — frontend
cd frontend && npm run dev
```

Then open `http://localhost:3000`.

---

## API (identical on both backends)

```bash
GET  /health                    → {"status": "ok"}
GET  /ready                     → {"ready": true}
POST /v1/bullets/stream         → Server-Sent Events stream of text chunks + a final {"metrics": {...}} frame
```

```bash
curl -X POST http://localhost:8000/v1/bullets/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

---

## Configuration

Both backends use hardcoded config constants (no `.env` files) — see `backend/src/config/settings.py` / `backend-onnx/src/config/settings.py`:

```python
MAX_INPUT_TOKENS = 2048   # matches how the model was trained — see Experiment_README.md §27 for how this was verified
MAX_NEW_TOKENS = 512
```

---

## Version Scope

**Included:** streaming inference, per-request KV cache with guaranteed cleanup, structured logging, Docker (backend/), a full web frontend, and a second ONNX-based serving path.

**Not included (see [Future Work](./Experiment_README.md#28-future-work)):** request batching/scheduling, authentication, a production tokenizer path for `backend-onnx/` that drops `transformers` entirely, and static ONNX quantization to fix the weight-size regression noted above.

---

## Where to Go Next

1. **Want the full story?** → [Experiment_README.md](./Experiment_README.md)
2. **Want to run the production backend?** → [backend/README.md](./backend/README.md)
3. **Want to run the ONNX experiment?** → [backend-onnx/README.md](./backend-onnx/README.md)
4. **Want the original trained model?** → [huggingface.co/JayShah07/falconai-text-bullet-t5](https://huggingface.co/JayShah07/falconai-text-bullet-t5)
5. **Want to see how the model was trained?** → `notebooks/` (01 through 09, in order)
