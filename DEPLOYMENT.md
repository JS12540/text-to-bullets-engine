# Deploying to Vercel

This covers deploying **`frontend/`** and **`backend-onnx/`** to Vercel as two separate projects. `backend/` (the PyTorch/TorchAO server) is **not** covered here — as discussed in [Experiment_README.md §27](./Experiment_README.md#27-the-onnx-experiment-torch-vs-onnx-runtime), it depends on `torch`+`transformers` (~300MB) and a persistent-process model-loading pattern that doesn't fit Vercel's serverless model well. `backend-onnx/` was built specifically to be deployable this way.

## Deployment Order: `backend-onnx` First

Deploy the backend before the frontend. The frontend needs the backend's live URL (`NEXT_PUBLIC_API_URL`) at build time, so there's nothing to point it at until the backend exists.

```text
1. Deploy backend-onnx  → get its Vercel URL
2. Set that URL as the frontend's NEXT_PUBLIC_API_URL
3. Deploy frontend
```

---

## Prerequisites

```bash
npm install -g vercel
vercel login
```

You'll need a Vercel account (Hobby is fine — see the caveats in Experiment_README.md's Vercel research if you plan to monetize this).

---

## Part 1: Deploy `backend-onnx`

### 1.1 Files already in place

These were created specifically for this deployment — you shouldn't need to touch them:

```text
backend-onnx/
├── src/main.py            # FastAPI app — Vercel auto-detects this as the entrypoint (zero-config)
├── vercel.json             # Function memory/timeout + excludeFiles (scripts/, model weights)
└── requirements.txt        # fastapi, onnxruntime, numpy, tokenizers, uvicorn
```

`src/main.py` exposes a module-level `app` object, which is what Vercel's Python runtime looks for automatically at `src/main.py` — no wrapper file or custom `rewrites` needed:
```python
app = create_app()  # module-level — this is what Vercel's Python runtime auto-detects
```

`vercel.json`:
```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "functions": {
    "src/main.py": {
      "memory": 1024,
      "maxDuration": 60,
      "excludeFiles": "{scripts/**,artifacts/text-to-bullets-onnx-int8/encoder_model.onnx,artifacts/text-to-bullets-onnx-int8/decoder_model_merged.onnx}"
    }
  }
}
```

### 1.2 Sanity-check locally first

Don't debug import errors on Vercel's build servers — verify the entrypoint imports cleanly first:

```bash
cd backend-onnx
.venv-serve/bin/python -c "from src.main import app; print(app.title)"
# Expected: Text-to-Bullets API (ONNX)
```

### 1.3 Model weights are excluded from the bundle, fetched from Blob at cold start

The two `.onnx` files (~195MB combined) are excluded from the function bundle via `excludeFiles` above — Vercel's Python bundle cap made bundling them directly too tight alongside `onnxruntime`'s own footprint. Instead:

- Upload `encoder_model.onnx` and `decoder_model_merged.onnx` to **Vercel Blob** (dashboard → Storage → Blob)
- Set `ENCODER_MODEL_URL` and `DECODER_MODEL_URL` env vars on the project to their Blob URLs
- `src/engine/model.py` downloads them into `/tmp` on cold start if those env vars are set; otherwise it falls back to the bundled `artifacts/` copy (local dev is unaffected)

### 1.4 Deploy

```bash
cd backend-onnx
vercel
```

Answer the prompts:
- **Set up and deploy?** Yes
- **Which scope?** (your account)
- **Link to existing project?** No (first time)
- **Project name?** e.g. `text-to-bullets-onnx`
- **Directory?** `./` (current directory — `backend-onnx/`)
- **Override settings?** No (the `vercel.json` already handles routing)

This creates a **preview** deployment first. Once it succeeds:

```bash
vercel --prod
```

This gives you a stable production URL, e.g. `https://text-to-bullets-onnx.vercel.app`.

### 1.5 Verify the deployed backend

```bash
BACKEND_URL="https://text-to-bullets-onnx.vercel.app"

curl -s "$BACKEND_URL/health"
# {"status": "ok"}

curl -s "$BACKEND_URL/ready"
# {"ready": true}   ← if this is slow or fails, see "Cold starts" below

curl -sN -X POST "$BACKEND_URL/v1/bullets/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "Acme reported quarterly revenue of $4.2 billion, up 12% year over year."}'
# Should stream `data: {"text": "..."}` chunks, ending with `data: {"metrics": {...}}`
```

If `/ready` returns `false` or times out, the model likely hasn't finished loading within the first request's window — see the cold-start note below.

#### Optional `temperature` parameter

`/v1/bullets/stream` also accepts `temperature` (float, `0.0`–`1.0`, default `0.0`):

```bash
curl -sN -X POST "$BACKEND_URL/v1/bullets/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "temperature": 0.7}'
```

- `0.0` (default) — greedy/deterministic decoding, same input always produces the same output (this is what the frontend uses by default)
- `> 0.0` — samples from the softmax distribution instead of always taking the top token; output becomes non-deterministic and can vary between runs, with quality degrading as temperature rises toward `1.0`

---

## Part 2: Deploy `frontend`

### 2.1 Point it at the deployed backend

```bash
cd frontend
vercel env add NEXT_PUBLIC_API_URL production
# Paste the backend-onnx URL from step 1.4 when prompted, e.g.:
# https://text-to-bullets-onnx.vercel.app
```

(Or set it in the Vercel dashboard under Project → Settings → Environment Variables — same effect.)

### 2.2 Deploy

```bash
vercel        # preview deploy
vercel --prod # production deploy
```

### 2.3 Verify

Open the frontend's production URL, paste some text, click Generate Bullets, and confirm streaming works end-to-end against the deployed backend — not just `localhost`.

---

## CORS

`backend-onnx/src/main.py` already sets `allow_origins=["*"]`, so it'll accept requests from the deployed frontend's Vercel domain (or any domain) without extra configuration. If you want to lock this down to only your frontend's domain once both are live:

```python
# backend-onnx/src/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app"],
    ...
)
```

Redeploy (`vercel --prod`) after changing this.

---

## Cold Starts — What to Expect

Per the measurements in [Experiment_README.md §27.5](./Experiment_README.md#275-latency-a-genuine-trade-off-not-a-clean-win):

- **Load time** (deserializing the ONNX graphs): ~0.4s — fast.
- **First inference call** on a cold instance: ~2.5s — ONNX Runtime does graph optimization lazily on first use.
- **Warm requests** after that: ~0.27s — faster than the torch backend.

So the first request to a cold Vercel instance will feel slow (load + first-call cost stacked together, roughly 3s). This is expected, not a misconfiguration. If Vercel spins your function down between requests (likely on low traffic with Hobby), every "first request after idle" pays this cost again.

If this matters for your use case, Vercel's **Fluid Compute** (mentioned in the plan) keeps instances warm longer across requests — worth enabling if cold starts are a problem in practice.

---

## Troubleshooting

**Build fails on `onnxruntime` or `tokenizers` install** — Vercel's Python builder should resolve these fine from `requirements.txt`; if it doesn't, check the build log for a Python version mismatch (this project was validated on Python 3.13).

**Function fails with an import error for `src.*`** — make sure you deployed from inside `backend-onnx/` (`cd backend-onnx && vercel`), not the repo root. `src/main.py`'s internal imports (e.g. `from src.api.routes import router`) assume `backend-onnx/` is the function's root.

**`/ready` returns `{"ready": false}`** — the model files didn't load. Check the function logs (`vercel logs <deployment-url>`) for the actual error; the most common cause during initial setup is `TOKENIZER_PATH` or `MODEL_PATH` pointing outside the deployed directory (already fixed — both now resolve to local paths inside `backend-onnx/artifacts/`, not `../backend/`).

**Output looks truncated after one bullet on longer inputs** — this was a real bug we found and fixed (`MIN_NEW_TOKENS` in `src/config/settings.py`); if you're running an older copy of `backend-onnx/`, pull the latest — see [Experiment_README.md §27.7](./Experiment_README.md#277-a-second-subtler-bug-quantization-precision-flipping-a-real-decision).

**Garbled output on very long, multi-topic input** — not a deployment issue; this is a real model-capacity limit documented in [Experiment_README.md §27.8](./Experiment_README.md#278-a-finding-that-turned-out-not-to-be-onnx-specific-at-all). Split multi-topic input into separate requests.

---

## Redeploying After Changes

```bash
cd backend-onnx   # or frontend
vercel --prod
```

Vercel keeps deployment history — `vercel rollback` reverts to a previous production deployment if a change breaks something.
