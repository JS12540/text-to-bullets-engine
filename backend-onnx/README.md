# text-to-bullets — ONNX conversion & accuracy validation

This is a **sibling** to `../backend/`, not a replacement. `backend/` is untouched
and remains the reference implementation / current production path.

## Why this exists

Deploying `backend/` to a serverless platform (e.g. Vercel) means shipping
`torch` + `transformers` (~300MB) just to run a T5-small model. `onnxruntime`
+ `numpy` alone measure at **100MB unpacked / 27MB compressed** (real numbers,
not an estimate — see "Results" below) and don't need `torch`/`transformers`
at inference time — those heavy libraries are only needed **once**, offline,
to export the model. That's what this directory does: convert, then prove the
converted model's output quality matches the original.

**Scope of this directory (current state):** conversion + accuracy validation
only. It is not yet a FastAPI server — see "Not yet done" below.

## Results (measured, not estimated)

**Accuracy: exact match.** On the test input (the frontend's example paragraph
+ system/task prompts), the ONNX INT8 pipeline produced **byte-identical**
output to the original torchao INT8 pipeline:

```
<pad><BULLET> Acme reported quarterly revenue of $4.2 billion, up 12% year over year.<BULLET> Operating profit increased 8% to $620 million, although operating margin declined from 17.2% to 14.8%.<BULLET> The company added 1.3 million customers during the quarter and raised full-year revenue guidance from $16 billion to $17.5 billion.<BULLET> Management warned that European demand weakened in July.</s>
```

**Size: went up, not down.** This is an honest, measured result, not what we
expected going in:

| | Size |
|---|---|
| torchao source (`backend/artifacts/text-to-bullets-int8`) | 76.1 MB |
| ONNX INT8 output (`text-to-bullets-onnx-int8/`) | 193.0 MB |

**Why it's bigger:** ONNX Runtime's `quantize_dynamic` only quantizes
`MatMul`/`Gemm` weights (the Linear layers). It does **not** quantize
embedding tables, which use a `Gather` op. T5-small's vocab embedding
(`32101 × 512 ≈ 16.4M` params, tied between input embedding and the LM head)
stays FP32 — about 65MB — and gets duplicated across both `encoder_model.onnx`
and `decoder_model_merged.onnx` (ONNX doesn't support weight sharing across
separate files; only the with-past/without-past duplication *within* the
decoder file was solved by `use_merged=True` + `accelerate`'s dedup pass).
Two FP32 copies of that embedding table already account for ~130MB before the
actually-quantized transformer weights are even counted.

**Library size, measured (not estimated):**

| | Unpacked | Compressed (zip) |
|---|---|---|
| `onnxruntime` + `numpy` + transitive deps | 100 MB | 27 MB |
| `torch` + `transformers` (backend's deps) | ~300 MB | — |

`onnxruntime` alone is 75MB unpacked (bigger than commonly-quoted "~20-30MB"
figures, which describe compressed download size, not installed size).

**What this means for total deployment size:** the library swap is real —
100MB vs ~300MB, roughly a 3x reduction — but because the ONNX *weights*
came out larger (193MB vs 76MB, see above), the total deployment size only
drops from ~376MB (torch+transformers+torchao weights) to ~293MB (onnxruntime
+numpy+ONNX weights) — a real but far smaller win than the library size alone
suggests. If minimizing the weight file matters too, static (not dynamic) ONNX
quantization, or explicitly quantizing the `Gather` embedding op, would need
to be explored — not done here; out of scope for this validation pass.

**Latency (measured, single request, 80 output tokens, greedy decoding):**

| | torch (torchao) | ONNX (ORT) |
|---|---|---|
| Load time | 2.337s | **0.389s** (6x faster) |
| Cold generate (1st call) | 0.504s | 2.520s (5x slower) |
| Warm generate (2nd+ call) | 0.398s | **0.266s** (33% faster) |

ONNX Runtime loads almost instantly (deserializing a graph) but pays a large
one-time cost on its *first* inference call — session/graph-optimization
warm-up happening lazily rather than at load time. Once warmed up, ONNX is
~33% faster per request than torch. Net effect depends on deployment shape:
on a serverless cold-start (load + first request, then the instance dies),
totals are roughly a wash (~2.84s torch vs ~2.91s ONNX); on a persistent warm
server handling many requests per loaded instance (matching how `backend/`
already runs), ONNX wins clearly on a per-request basis after the first call.

## Structure

```
backend-onnx/
├── pyproject.toml              # lean runtime-only deps (onnxruntime, numpy) — no torch/transformers
├── requirements.txt             # same lean deps as pyproject.toml, for a future deployed server
├── requirements-dequant.txt    # phase 1 conversion deps — use with .venv-dequant
├── requirements-export.txt     # phase 2 conversion deps — use with .venv-export
├── scripts/                     # one-off conversion + validation scripts (not a server)
│   ├── dequantize.py              # phase 1: torchao checkpoint -> plain float checkpoint
│   ├── export_onnx.py             # phase 2: plain checkpoint -> ONNX -> ONNX Runtime INT8
│   ├── tokenize_input.py          # phase 1: tokenize test input (dequant venv has a working tokenizer)
│   ├── decode_output.py           # phase 1: decode ONNX output ids back to text
│   ├── run_torch_baseline.py      # phase 1: run the original torchao model, save output
│   ├── run_onnx.py                # phase 2: run the ONNX model on pre-tokenized ids, save output ids
│   └── compare_outputs.py         # no heavy deps: diff the two saved outputs
└── artifacts/
    └── text-to-bullets-onnx-int8/
        ├── encoder_model.onnx
        └── decoder_model_merged.onnx
```

**Which requirements file do I use?**

| File | Purpose | Venv |
|---|---|---|
| `requirements.txt` (or `pyproject.toml`'s base deps) | Lean runtime — what an eventual deployed server needs | not built yet |
| `requirements-dequant.txt` | Phase 1 of conversion: load torchao checkpoint, dequantize | `.venv-dequant` |
| `requirements-export.txt` | Phase 2 of conversion: export to ONNX, quantize | `.venv-export` |

You only need `requirements-dequant.txt`/`requirements-export.txt` if you're
re-running the conversion. `requirements.txt` is for running/serving the
already-converted model — no server exists yet to consume it (see "Not yet
done" below).

## Known issues (why two separate venvs, and why the odd version pins)

This conversion hit a chain of real, confirmed ecosystem incompatibilities —
worth documenting since they'll resurface on any re-run:

1. **`optimum-onnx` (current `optimum[onnxruntime]` releases) hard-pins
   `transformers<5`**, but loading this repo's torchao-quantized checkpoint
   requires `transformers>=5.17.0` (confirmed — that's what `backend/`'s own
   working venv uses; older transformers can't deserialize this torchao tensor
   format). These two requirements cannot be satisfied in one environment.
2. **`uv` resolves *all* declared `[project.optional-dependencies]` groups
   jointly**, even when you only `uv sync --extra X` one of them — so putting
   the conflicting dequant/export dependency sets in the same `pyproject.toml`
   as optional-dependency groups doesn't work; `uv` reports them as mutually
   unsatisfiable. Fixed by using two fully separate venvs and plain
   `pip`/`uv pip install`, not project extras.
3. **`transformers.PretrainedConfig.save_pretrained()` crashes on this
   checkpoint** (`NotImplementedError` in `revert_weight_conversion`) even
   after dequantizing the weights, because the config still carries torchao's
   weight-packing metadata. Fixed in `dequantize.py` by writing the safetensors
   state dict directly (`safetensors.torch.save_file`) instead of using
   `model.save_pretrained()`'s weight-writing path, and by *deleting* (not
   nulling) `quantization_config` before saving — a `null` value round-trips
   through `config.json` as a literal `None` that crashes a later
   `.to_dict()` call.
4. **Newer `optimum` releases (2.x) are broken against `transformers>=4.58`**
   (`ImportError: cannot import name 'is_tf_available'` — an internal
   `transformers` API `optimum` still references was removed) and separately
   broken against `transformers` 5.x's `PretrainedConfig` internals
   (`NormalizedConfig.__init__() got multiple values for argument 'allow_new'`).
   The only combination that actually worked end-to-end was the **older,
   pre-split monolithic `optimum==1.19.0`** with `transformers==4.39.3`.
5. **`optimum==1.19.0`'s post-export cleanup code assumes torch's legacy
   (TorchScript-tracing) ONNX exporter's temp-file naming**, which doesn't
   match torch>=2.3's newer dynamo-based exporter (`FileNotFoundError` on a
   `.onnx.data` temp file). Fixed by pinning `torch==2.2.2` in the export venv
   specifically (needs Python 3.12 — no 3.13+ wheels exist for this torch
   version).
6. **`use_merged=True` (to avoid decoder/decoder-with-past weight duplication)
   silently produces a *larger* file if `accelerate` isn't installed** — the
   weight-deduplication pass it depends on gets skipped with only a warning,
   not an error. `accelerate` is required in the export venv for this to work
   as intended.
7. **The export venv's old `transformers` (4.39.3) requires `tokenizers<0.19`**,
   which can't parse the `tokenizer.json` format this checkpoint's *newer*
   tokenizer was saved with (`Exception: data did not match any variant of
   untagged enum PyPreTokenizerTypeWrapper`) — and bumping `tokenizers` alone
   breaks transformers' own import-time version check the other way. Resolved
   by not tokenizing/decoding in the export venv at all: `tokenize_input.py`
   and `decode_output.py` run in the (working) dequant venv and hand off plain
   token-ID JSON across the venv boundary; `run_onnx.py` only ever sees numbers.
8. **torch 2.2.2 + numpy 2.x are ABI-incompatible** (`RuntimeError: Numpy is
   not available`, silently preceded by a `_ARRAY_API not found` warning at
   import time) — needs `numpy<2` in the export venv specifically.

## Running the conversion

```bash
cd backend-onnx

# Phase 1: dequantize
python3 -m venv .venv-dequant
.venv-dequant/bin/pip install -r requirements-dequant.txt
.venv-dequant/bin/python scripts/dequantize.py

# Phase 2: export + quantize to ONNX (separate venv, Python 3.12 required)
uv venv --python 3.12 .venv-export
uv pip install --python .venv-export/bin/python -r requirements-export.txt
.venv-export/bin/python scripts/export_onnx.py
```

## Running the accuracy check

```bash
.venv-dequant/bin/python scripts/tokenize_input.py      # tokenize the test input once
.venv-dequant/bin/python scripts/run_torch_baseline.py  # original torchao model's output
.venv-export/bin/python scripts/run_onnx.py             # ONNX model's output (as token ids)
.venv-dequant/bin/python scripts/decode_output.py       # decode those ids back to text
.venv-dequant/bin/python scripts/compare_outputs.py     # or .venv-export — no heavy deps either way
```

Uses the same system+task prompt as `backend/src/constants/prompts.py` and the
same example input as the frontend's "Example" button (kept as literal copies
in these scripts, not imported, so `backend-onnx` has zero dependency on
`backend`'s source tree), with greedy decoding on both sides to match
`backend/src/engine/decode.py`'s strategy.

## Not yet done (future work)

- Investigating why the ONNX weights came out larger (static quantization or
  explicit embedding-table quantization, per the "Results" section above)
- `onnxruntime`-based rewrite of `prefill.py` / `decode.py` / `cache.py` for
  true incremental per-token streaming (this validation used the simpler
  high-level `.generate()` API, not the manual encoder-once /
  decoder-with-cache loop `backend/` uses for SSE streaming)
- A FastAPI server wrapping this ONNX pipeline
- A production tokenizer path that doesn't depend on `transformers` at all
  (e.g. the standalone `tokenizers` library) — this validation still uses
  `transformers.AutoTokenizer` for convenience
