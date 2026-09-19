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
├── api/
│   └── index.py          # Vercel's Python entrypoint — exposes the FastAPI app
├── vercel.json            # Routes all paths to api/index.py, sets memory/timeout
├── .vercelignore          # Excludes local venvs and conversion-only scripts
└── requirements.txt        # fastapi, onnxruntime, numpy, tokenizers, uvicorn
```

`api/index.py`:
```python
from src.main import create_app
app = create_app()
```

`vercel.json`:
```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "functions": {
    "api/index.py": {
      "memory": 1024,
      "maxDuration": 60
    }
  },
  "rewrites": [
    { "source": "/(.*)", "destination": "/api/index" }
  ]
}
```

### 1.2 Sanity-check locally first

Don't debug import errors on Vercel's build servers — verify the entrypoint imports cleanly first:

```bash
cd backend-onnx
.venv-serve/bin/python -c "from api.index import app; print(app.title)"
# Expected: Text-to-Bullets API (ONNX)
```

### 1.3 Check what will actually get uploaded

The model artifacts are ~195MB (`encoder_model.onnx` + `decoder_model_merged.onnx` + tokenizer files) — comfortably under Vercel's 500MB Python function limit, but worth confirming `.vercelignore` is excluding the venvs and scripts before you upload:

```bash
du -sh artifacts/text-to-bullets-onnx-int8/
# ~195M — this is what gets bundled into the function
```

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

**Function fails with an import error for `src.*`** — make sure you deployed from inside `backend-onnx/` (`cd backend-onnx && vercel`), not the repo root. `api/index.py`'s `from src.main import create_app` assumes `backend-onnx/` is the function's root.

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
