# Text to Bullets

> An end-to-end experiment in building a small, specialized text-to-bullet-points system: model selection, fine-tuning, architecture changes, evaluation, quantization, CPU inference, manual prefill/decode, KV caching, streaming, FastAPI serving, ONNX export, and a web frontend.

## Overview

I started this project with a simple question:

> **Can a small model specialized for one transformation task compete with a much larger general-purpose language model?**

The task I picked is intentionally narrow:

```text
Long, unstructured English text
                ↓
        Text-to-Bullets model
                ↓
Concise, factual bullet points
```

What began as a single fine-tuning experiment grew into something much bigger — I ended up walking the full path from model research to a deployable application, and later to a second deployment path entirely:

```text
Model comparison → Fine-tuning → Architecture selection → Evaluation
→ Quantization → CPU optimization → Prefill/Decode → KV cache
→ Streaming → FastAPI → Frontend → ONNX conversion → Torch vs ONNX serving
```

Here's the story of how I got there, what worked, what broke, and what I learned from each dead end along the way. The final direction uses a fine-tuned **T5 encoder-decoder model (~60.5M parameters)**, **TorchAO INT8 weight-only quantization**, and a streaming inference backend — with a second, lighter-weight ONNX Runtime backend built afterward to explore serverless deployment.

---

## Table of Contents

1. [Why Text-to-Bullets?](#1-why-text-to-bullets)
2. [Initial Model Experiment](#2-initial-model-experiment)
3. [Fine-Tuning a Small Decoder-Only Model](#3-fine-tuning-a-small-decoder-only-model)
4. [Decoder-Only vs Encoder-Decoder](#4-decoder-only-vs-encoder-decoder)
5. [How the T5 Encoder Works](#5-how-the-t5-encoder-works)
6. [How the T5 Decoder Works](#6-how-the-t5-decoder-works)
7. [Why KV Cache Matters](#7-why-kv-cache-matters)
8. [Decoder-Only Prefill vs T5 Prefill](#8-decoder-only-prefill-vs-t5-prefill)
9. [Final T5 FP32 Baseline](#9-final-t5-fp32-baseline)
10. [Qwen vs Specialist T5](#10-qwen-vs-specialist-t5)
11. [Precision and Quantization Experiments](#11-precision-and-quantization-experiments)
12. [Exact Output Regression Analysis](#12-exact-output-regression-analysis)
13. [CPU Deployment Experiment](#13-cpu-deployment-experiment)
14. [Why TorchAO Was Selected](#14-why-torchao-was-selected)
15. [TorchAO CPU Strategy Comparison](#15-torchao-cpu-strategy-comparison)
16. [Rust/Candle Experiment](#16-rustcandle-experiment)
17. [Final Local Streaming Result](#17-final-local-streaming-result)
18. [Streaming Inference](#18-streaming-inference)
19. [Per-Request State and Cleanup](#19-per-request-state-and-cleanup)
20. [Context-Length Safety](#20-context-length-safety)
21. [FastAPI Backend](#21-fastapi-backend)
22. [Why Scheduling Is Future Work](#22-why-scheduling-is-future-work)
23. [Frontend](#23-frontend)
24. [Repository Structure](#24-repository-structure)
25. [Major Findings](#25-major-findings)
26. [Final Technology Choices](#26-final-technology-choices)
27. [The ONNX Experiment: Torch vs ONNX Runtime](#27-the-onnx-experiment-torch-vs-onnx-runtime)
28. [Future Work](#28-future-work)
29. [Conclusion](#conclusion)

---

## 1. Why Text-to-Bullets?

I chose this task because it sits in a sweet spot. A general-purpose LLM can obviously convert paragraphs into bullets, but it carries capacity for hundreds of unrelated tasks it will never use here. Text-to-bullets, by contrast, only needs to understand a source, identify important information, remove repetition, preserve facts and numbers, compress related information, and emit concise structured bullets. That's it.

This suggested my core hypothesis:

> **A smaller model with the right architecture and task-specific training may be more appropriate than serving a much larger general-purpose LLM.**

Everything that follows is me testing that hypothesis, one experiment at a time.

---

## 2. Initial Model Experiment

I started by evaluating four small language models to get a baseline for what "small" could realistically achieve.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg Latency (s) | Median Latency (s) | Tokens/s | Bullet Format |
|---|---:|---:|---:|---:|---:|---:|---:|
| BitNet | 0.6006 | 0.4041 | 0.4814 | 15.52 | 12.33 | 10.98 | 0.9996 |
| **Qwen3 0.6B** | **0.6494** | **0.4702** | **0.5463** | 7.31 | 6.11 | 21.77 | **0.9999** |
| SmolLM 135M | 0.4528 | 0.3180 | 0.3722 | 9.91 | 8.00 | **26.75** | 0.0828 |
| SmolLM 360M | 0.5827 | 0.4427 | 0.4900 | **7.09** | **5.21** | 24.79 | 0.4661 |

### Findings

**Qwen3 0.6B** came out as the strongest quality baseline right away. SmolLM 135M had the highest throughput of the four, but its ROUGE was weak and it only followed the bullet format ~8.3% of the time — it was fast at producing the wrong thing. SmolLM 360M was a real step up in both quality and latency, but bullet formatting was still only ~46.6%.

This taught me an early lesson that shaped everything after it: **throughput is irrelevant if the model cannot reliably perform the task.**

---

## 3. Fine-Tuning a Small Decoder-Only Model

My next question was whether a much smaller model could learn this transformation directly, rather than doing it zero-shot. So I fine-tuned a SmolLM decoder-only model (SFT) specifically for this task.

A decoder-only model represents source and target as one autoregressive sequence:

```text
[Instruction][Source][Target bullets]
                ↓
        Decoder-only Transformer
                ↓
          Next-token prediction
```

I fine-tuned **SmolLM2-135M** (SFT) on the task and compared it against **Qwen3-0.6B** running zero-shot (no fine-tuning) as a reference point:

| Model | Parameters (M) | ROUGE-1 | ROUGE-2 | ROUGE-L | Bullet Format |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M SFT (fine-tuned) | 135 | 0.4318 | 0.2374 | 0.3083 | **1.0000** |
| Qwen3-0.6B Zero-shot | 600 | **0.6494** | **0.4702** | **0.5463** | 0.9999 |

### Findings

Fine-tuning fixed the format-compliance problem completely. The original zero-shot SmolLM 135M in Section 2 only produced valid bullet formatting **8.3%** of the time; after SFT, the very same size model hit **100%**. That told me the model *could* reliably learn the output structure at this parameter count — training, not capacity, had been the bottleneck for formatting.

But ROUGE told a different story. Even after fine-tuning, SmolLM2-135M still trailed Qwen3-0.6B running zero-shot at roughly 4.4x the parameter count. Fine-tuning closed the format gap but not the content-quality gap. This was the result that shifted my investigation from "can a small decoder-only model learn this task" to a bigger question: **architectural fit**. Maybe the problem wasn't insufficient training — maybe a decoder-only architecture just isn't the most natural fit for a source-to-target compression task in the first place.

---

## 4. Decoder-Only vs Encoder-Decoder

So I stepped back and thought about the architectures themselves. Decoder-only models like Qwen, Llama, and SmolLM treat the prompt and the generated answer as one continuous sequence:

```text
Prompt tokens → Decoder → Next token → Next token → ...
```

That's ideal for flexible, general-purpose generation where the "prompt" and "answer" don't have a rigid structural relationship.

But text-to-bullets has an explicit source/target structure:

```text
SOURCE DOCUMENT
       ↓
understand + compress
       ↓
TARGET BULLETS
```

T5 explicitly models exactly this relationship:

```text
SOURCE → ENCODER → source representation → DECODER → TARGET
```

The encoder builds a contextual representation of the *entire* source up front. The decoder then generates the target while separately tracking its own generated history and attending back to that fixed source representation. This felt like a much closer match to the actual shape of my task, so I decided to pursue it seriously.

---

## 5. How the T5 Encoder Works

To understand what I was building, I traced through a representative request: the backend prompt plus user input came out to **279 tokens**.

```text
input_ids: (1, 279)
      ↓ embeddings
(1, 279, 512)
      ↓
Encoder Layer 1
      ↓
...
      ↓
Encoder Layer 6
      ↓
Final encoder hidden states
(1, 279, 512)
```

Each self-attention layer creates Query, Key, and Value projections:

```text
Hidden state X
 ├─ Wq → Q
 ├─ Wk → K
 └─ Wv → V

Q × Kᵀ → scaled scores → softmax → attention weights → weights × V
```

The final encoder output is a contextual representation of the complete source.

### Encoder Prefill

I call this encoder pass **prefill** in the serving implementation:

```text
System prompt + user text
        ↓
Tokenizer
        ↓
Context validation
        ↓
Encoder (once)
        ↓
Encoder hidden states
```

For that representative 279-token request, encoder prefill took about **159 ms**.

Crucially, the encoder only runs **once** per request — its output gets reused by every single decoder step that follows. That's the whole point of separating source from target.

---

## 6. How the T5 Decoder Works

The decoder starts with no visible generated text, so it begins from `decoder_start_token_id` (0 for this model).

```text
<START> → embedding → (1,1,512)
```

Every output token then passes through **all six decoder layers**:

```text
Current decoder token
      ↓
Decoder Layer 1
  self-attention
  cross-attention
  feed-forward
      ↓
Decoder Layer 2
      ↓
...
      ↓
Decoder Layer 6
      ↓
LM head
      ↓
vocabulary logits
      ↓
next token
```

No individual decoder layer generates a token by itself. All six layers progressively transform the representation before the LM head finally predicts the next token.

### Decoder Self-Attention

Self-attention answers, roughly:

> **What have I generated so far?**

At the first position:

```text
Q/K/V ≈ (1, 8, 1, 64)
```

where:

- `1` = batch
- `8` = attention heads
- `1` = decoder positions
- `64` = dimensions per head
- `8 × 64 = 512` hidden dimensions

As generation grows, decoder self-attention is causal: position N can attend to positions `1..N`, never future positions.

### Decoder Cross-Attention

Cross-attention answers, roughly:

> **What information from the original source is relevant now?**

Here:

```text
Query  ← decoder
Keys   ← encoder outputs
Values ← encoder outputs
```

For my 279 source positions:

```text
K_cross = (1, 8, 279, 64)
V_cross = (1, 8, 279, 64)
```

The decoder's current Query is compared against all source Keys, and the resulting attention weights select a weighted combination of source Values.

So at every step, the decoder has two complementary views:

```text
Self-attention  → "What have I written?"
Cross-attention → "What does the source say?"
```

---

## 7. Why KV Cache Matters

Without KV caching, autoregressive generation would repeatedly recompute old decoder positions from scratch on every step.

```text
Step 1: process position 1
Step 2: process positions 1..2 again
Step 3: process positions 1..3 again
...
```

For 100 generated positions, that historical work grows roughly as:

```text
1 + 2 + 3 + ... + 100
```

That's obviously wasteful, and it's exactly what KV caching avoids:

```text
Step 1: calculate K1/V1 → save
Step 2: reuse K1/V1 + calculate K2/V2
Step 3: reuse K1..K2/V1..V2 + calculate K3/V3
```

After the first step, the model only ever receives the newest decoder token plus its `past_key_values`.

### T5 Has Two Decoder Caches

I learned that T5's **self-attention cache** tracks generated decoder history and grows one position per step:

```text
position 9:
K/V = (1,8,9,64)

position 10:
K/V = (1,8,10,64)
```

Its **cross-attention cache**, on the other hand, represents the fixed source and never changes size:

```text
K/V = (1,8,279,64)
```

and stayed at length 279 for the entire generation.

### Why Every Layer Has Its Own Cache

Each of the six decoder layers has its own learned projection matrices and receives a different hidden representation, so each layer needs its own self- and cross-attention K/V — there's no sharing across layers.

```text
Request KV Cache
├─ Layer 1: self K/V + cross K/V
├─ Layer 2: self K/V + cross K/V
...
└─ Layer 6: self K/V + cross K/V
```

### Measured Cache Memory

I actually measured this rather than just estimating it. At decoder position 9:

```text
Self cache = 108 KB
```

At position 10:

```text
Self cache = 120 KB
```

So one position adds **12 KB** across all six decoder layers. The math checks out:

```text
K: 8 heads × 64 × 2 BF16 bytes = 1024 bytes/layer
V:                              = 1024 bytes/layer
K+V                             = 2048 bytes/layer
× 6 layers                      = 12,288 bytes = 12 KB/token
```

The 279-position cross-attention cache occupied approximately **3348 KB** across all six layers and stayed fixed for the whole request.

By the end of one 279-input-token / 79-decoder-position request, total KV tensor memory came to approximately **4.20 MB**.

> **KV caching is a memory-for-compute trade.**

---

## 8. Decoder-Only Prefill vs T5 Prefill

I wanted to understand exactly how this differed from the decoder-only path I'd already tried in Section 3. In a decoder-only model like Qwen, there's no separate encoder — a 279-token prompt is itself decoder history:

```text
279 prompt tokens → decoder → self-attention KV for 279 positions
```

So the prompt's KV cache exists before the model even generates its first token.

T5 instead separates the two sequences entirely:

```text
SOURCE → Encoder → fixed source representation
OUTPUT → Decoder → growing decoder history
```

Which produces:

```text
T5:
fixed cross-attention source cache
+
growing decoder self-attention cache

Decoder-only:
one growing self-attention history containing prompt + output
```

That structural difference — a fixed source cache instead of one continuously growing sequence — is the concrete, mechanical reason T5's architecture matches this task better than a decoder-only model does.

---

## 9. Final T5 FP32 Baseline

With the architecture decided, I trained the final model: approximately **60,492,800 parameters**.

Estimated FP32 parameter storage:

```text
~230.76 MB
```

I ran a 500-example evaluation to get a real baseline before touching quantization:

| Metric | T5 FP32 |
|---|---:|
| Examples | 500 |
| ROUGE-1 | 0.6041 |
| ROUGE-2 | 0.5330 |
| ROUGE-L | 0.5603 |
| BERTScore Precision | 0.9128 |
| BERTScore Recall | 0.8568 |
| BERTScore F1 | 0.8826 |
| Bullet Format | 1.000 |
| Avg Predicted Bullets | 3.59 |
| Avg Reference Bullets | 4.94 |
| Mean Bullet Count Error | 1.806 |
| Avg Latency | 0.970 s |
| Median Latency | 0.895 s |
| P95 Latency | 1.947 s |
| Avg Output Tokens | 89.93 |
| Throughput | 97.28 tok/s |

This became my reference point for every quantization experiment that followed.

---

## 10. Qwen vs Specialist T5

Now I could finally put my original hypothesis to the test directly.

| Metric | Qwen3 0.6B | Fine-Tuned T5 |
|---|---:|---:|
| Approx. Parameters | ~600M | **60.5M** |
| Architecture | Decoder-only | Encoder-decoder |
| ROUGE-1 | **0.6494** | 0.6041 |
| ROUGE-2 | 0.4702 | **0.5330** |
| ROUGE-L | 0.5463 | **0.5603** |
| Bullet Format | 0.9999 | **1.0000** |

I want to be honest about the caveat here: these measurements came from different stages of the experiment, so I shouldn't treat this as a perfectly controlled head-to-head unless the evaluation split and generation settings were identical. But the defensible finding still stands:

> **A ~60M specialist encoder-decoder became competitive with a ~600M general-purpose decoder model, and actually exceeded the original Qwen result on some task-specific overlap metrics.**

This doesn't mean T5 is a better general-purpose model — it isn't. It demonstrates the value of **specialization and architectural fit**, which was my original hypothesis, now with real numbers behind it.

---

## 11. Precision and Quantization Experiments

With a model I was happy with, the next question was how small I could make it without losing quality. I evaluated the model under multiple numerical representations.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERT F1 | Median Latency | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.6041 | 0.5330 | 0.5603 | 0.8826 | **0.895 s** | **97.28** |
| FP16 | 0.6040 | 0.5329 | 0.5601 | 0.8826 | 0.961 s | 87.12 |
| BF16 | 0.6064 | 0.5352 | 0.5629 | 0.8831 | 1.064 s | 80.73 |
| BnB LLM INT8 | 0.6052 | 0.5349 | 0.5628 | 0.8829 | 2.698 s | 31.59 |
| BnB INT8 Skip LM Head | 0.6052 | 0.5349 | 0.5628 | 0.8829 | 2.722 s | 31.71 |
| BnB NF4 INT4 | **0.6273** | **0.5578** | **0.5841** | **0.8891** | 1.360 s | 68.61 |
| Quanto INT8 W8 | 0.6054 | 0.5343 | 0.5618 | 0.8831 | 1.482 s | 59.92 |
| TorchAO INT8 W8 | 0.6100 | 0.5381 | 0.5656 | 0.8842 | 1.364 s | 64.48 |
| Encoder INT8 / Decoder FP16 | 0.6060 | 0.5347 | 0.5624 | 0.8835 | 1.012 s | 84.59 |

### Findings

Some of this surprised me:

- **FP16 was nearly identical to FP32 in quality but was not faster in that environment.**
- **BF16 preserved quality but likewise did not automatically improve latency.**
- **BitsAndBytes INT8 preserved quality but was much slower for the tested workload.**
- **NF4 INT4 produced surprisingly strong aggregate metrics, but substantially changed generations.**
- **TorchAO provided the most useful path toward a native PyTorch CPU deployment artifact.**

The broader lesson I took from this table:

> **Lower precision does not automatically mean lower latency. Kernel/runtime support matters.**

That lesson would come back to matter a lot later, when I started working with ONNX Runtime.

---

## 12. Exact Output Regression Analysis

Quality metrics like ROUGE are aggregate scores — they don't tell you whether a quantized model produces the *same* output as the FP32 baseline, or just a *similarly good* one. So I checked.

| Model | Exact Same as FP32 | Changed Examples |
|---|---:|---:|
| FP32 | 100.0% | 0 |
| FP16 | 98.6% | 7 |
| BF16 | 85.8% | 71 |
| BnB INT8 | 71.0% | 145 |
| BnB NF4 INT4 | 32.8% | 336 |
| Quanto INT8 W8 | 80.2% | 99 |
| TorchAO INT8 W8 | 75.4% | 123 |
| Encoder INT8 / Decoder FP16 | 83.6% | 82 |

A different generation isn't automatically a regression — small logit perturbations can produce different but semantically valid phrasing. That's exactly why I never relied on exact equality alone. My evaluation combined multiple signals:

- ROUGE
- BERTScore
- bullet formatting
- bullet counts
- exact equality
- latency
- throughput
- per-example regression analysis

This same idea — "different output isn't necessarily wrong output" — turned out to matter again much later, in a very concrete way, when I compared my torch backend against an ONNX Runtime backend (Section 27).

---

## 13. CPU Deployment Experiment

Since this was always meant to run on CPU, I ran a separate 100-example CPU evaluation comparing FP32 against BitsAndBytes quantization.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERT F1 | Median Latency | P95 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.6265 | 0.5648 | 0.5939 | 0.8911 | **2.38 s** | **6.95 s** |
| BnB INT8 BF16 | 0.6358 | 0.5762 | 0.6041 | 0.8937 | 14.03 s | 35.57 s |
| BnB NF4 INT4 | **0.6590** | **0.5998** | **0.6231** | **0.8993** | 7.64 s | 16.30 s |

This one really stood out to me: BitsAndBytes reduced precision but made CPU inference *dramatically* slower — 6x slower for INT8, 3x slower for NF4 INT4, despite the smaller numerical representation. It cleanly separated two concepts I'd been conflating:

```text
smaller model representation ≠ faster runtime
```

That single line became a recurring theme for the rest of the project.

---

## 14. Why TorchAO Was Selected

Given everything above, my final quantization configuration became:

```text
TorchAO Int8WeightOnlyConfig
Per-row quantization
Version 2
```

When I verified the resulting model, I found:

```text
96 TorchAO Int8Tensor weights
38 non-quantized higher-precision weights
```

Components like embeddings, layer norms, and attention biases stayed at higher precision, while the appropriate linear weights were stored in INT8.

### Final Size

```text
FP32 estimate:              ~230.76 MB
TorchAO runtime footprint:  ~115.38 MB
Serialized safetensors:      ~73.78 MB
```

I learned here that disk size and live model memory are genuinely different measurements — the loaded representation includes runtime structures, quantization metadata, and non-INT8 tensors that don't show up in the serialized file size.

The final artifact was self-contained:

```text
text-to-bullets-int8/
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

This 73.78 MB checkpoint is the one every later chapter of this story builds on — including the ONNX experiment.

---

## 15. TorchAO CPU Strategy Comparison

TorchAO itself offers more than one quantization strategy, so I compared them before settling on my final choice.

| Runtime | Avg TTFT | Mean ITL | Avg Latency | Decode Tokens/s |
|---|---:|---:|---:|---:|
| A8W8 Dynamic | 8.25 s | 85.29 ms | 18.75 s | 12.01 |
| **INT8 Weight-Only** | **3.32 s** | **48.47 ms** | **10.35 s** | **21.24** |

Weight-only INT8 was clearly, unambiguously preferable to dynamic A8W8 for this model and runtime — nearly 2x the throughput. This is the strategy I carried forward into production.

---

## 16. Rust/Candle Experiment

At this point I had a nagging question: was Python itself part of my latency problem? So I tested native Rust/Candle FP32 inference to find out whether removing Python would automatically improve serving performance.

| Metric | Rust / Candle FP32 |
|---|---:|
| Checkpoint | 230.78 MB |
| Peak RSS | 506.15 MB |
| Avg Prefill | 0.540 s |
| Avg TTFT | 0.619 s |
| Median TTFT | 0.579 s |
| Mean ITL | 86.0 ms |
| Avg P95 ITL | 121.7 ms |
| Avg Total Latency | 14.60 s |
| Decode Throughput | 12.34 tok/s |

The answer was no. This experiment taught me:

> **Native orchestration alone does not guarantee faster neural-network inference.**

Matrix kernels, attention implementation, threading, memory layout, quantization support, and runtime optimization dominate this workload — not the host language. This is a lesson I'd end up relearning from a completely different angle in Section 27, when I moved my TorchAO model to ONNX Runtime and the *library*, not the language, turned out to be what mattered.

---

## 17. Final Local Streaming Result

With the architecture, quantization strategy, and runtime all decided, I loaded the serialized TorchAO INT8 artifact locally and verified it end-to-end:

```text
Model footprint:       115.38 MB
TorchAO INT8 tensors:  96
Input tokens:          279
```

Representative request:

```text
Prefill:         0.162 s
TTFT:            0.213 s
Mean ITL:        5.86 ms
Total latency:   0.670 s
Output tokens:   78
```

This became the basis of the production backend — everything from here on is about serving this exact model well.

---

## 18. Streaming Inference

Instead of making the client wait for the complete generation:

```text
request → full generation → response
```

I built the backend to stream:

```text
request
   ↓
prefill
   ↓
token 1 ─────→ client
token 2 ─────→ client
token 3 ─────→ client
...
```

The metrics I care about for a streaming server are:

- **TTFT:** request start to first available token.
- **ITL:** time between generated tokens.
- **Total latency:** request start to generation completion.
- **Prefill latency:** source encoding time.
- **Decode throughput:** output tokens generated per second.

---

## 19. Per-Request State and Cleanup

I designed the server so the model and tokenizer are shared globally, but inference state is private to each request.

```text
Shared:
  model
  tokenizer

Per request:
  request_id
  input_ids
  attention_mask
  encoder_outputs
  decoder_input_ids
  past_key_values
  generated_ids
  timing data
  EOS/cancellation flags
```

With concurrent users, that looks like:

```text
                 SHARED MODEL
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
    Request A     Request B     Request C
        │             │             │
      KV A          KV B          KV C
```

Cleanup is guaranteed through a `try/finally` lifecycle:

```text
complete / error / cancellation / disconnect
                    ↓
release past_key_values
release encoder outputs
release request tensors/state
                    ↓
model remains loaded
```

This prevents stale caches from accumulating in a long-running service — something I was careful about from the start, because I knew this backend would need to serve many requests without a restart.

---

## 20. Context-Length Safety

I made a deliberate decision here: the backend combines the server-controlled task instruction with the user's text and tokenizes it **without truncation**.

If:

```text
input_tokens > MAX_INPUT_TOKENS
```

the request is rejected outright.

I chose to never silently truncate source text, because doing so could remove important facts and produce an incomplete or misleading summary — a summarizer that quietly drops the end of your document is worse than one that tells you your input is too long.

---

## 21. FastAPI Backend

Version 1 of the backend follows this lifecycle:

```text
HTTP request
      ↓
Pydantic validation
      ↓
Context validation
      ↓
Create GenerationState
      ↓
Encoder prefill
      ↓
Decoder step
      ├─ next token
      └─ updated KV
      ↓
Stream chunk
      ↓
repeat decode
      ↓
EOS / max tokens / cancellation
      ↓
cleanup
```

The tokenizer and INT8 model load once at application startup and stay in memory for the server's lifetime.

The server-controlled task prompt stays on the backend rather than being exposed to the browser — the frontend only ever sends raw user text.

---

## 22. Why Scheduling Is Future Work

I kept Version 1 deliberately focused on one thing: a correct, understandable inference path for a single request at a time.

When many requests arrive simultaneously, a scheduler becomes necessary to decide:

- which request gets model compute next;
- whether to prioritize prefill or decode;
- how many requests may remain active;
- whether compatible decode steps can be batched;
- whether KV-cache memory is sufficient;
- how cancelled requests are removed.

I see the progression intentionally as:

```text
V1: one correct inference lifecycle
V2: request queue / worker
V3: multiple active GenerationStates
V4: batched decode
V5: continuous batching
V6: cache-aware scheduling
```

I wanted this learning path to stay explicit instead of hiding it behind a serving framework I didn't fully understand.

---

## 23. Frontend

To actually use this thing, I built a frontend with a simple input/output workflow plus real inference metrics:

```text
┌───────────────────────┬───────────────────────┐
│ Input                 │ Output                │
│ source text           │ streamed bullets      │
│ Generate Bullets      │ Copy                  │
└───────────────────────┴───────────────────────┘

Technical Details:
Latency | TTFT | Mean ITL | Input Tokens | Output Tokens | Model
```

I made the metrics deliberately visible, because this project is both a useful application to me and an educational inference-serving experiment — I wanted to see the numbers every time I used it, not just when I was debugging.

---

## 24. Repository Structure

```text
text-to-bullets/
├── backend/
│   ├── src/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── engine/
│   │   │   ├── inference.py
│   │   │   ├── state.py
│   │   │   ├── prefill.py
│   │   │   ├── decode.py
│   │   │   └── cache.py
│   │   ├── config/
│   │   ├── constants/
│   │   └── utils/
│   ├── artifacts/
│   │   └── text-to-bullets-int8/
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── Dockerfile
├── backend-onnx/
│   ├── src/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── engine/
│   │   │   ├── inference.py
│   │   │   ├── state.py
│   │   │   ├── prefill.py
│   │   │   ├── decode.py
│   │   │   └── cache.py
│   │   └── config/
│   ├── scripts/
│   │   ├── dequantize.py
│   │   ├── export_onnx.py
│   │   └── compare_outputs.py
│   ├── artifacts/
│   │   └── text-to-bullets-onnx-int8/
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── requirements-dequant.txt
│   └── requirements-export.txt
├── frontend/
├── notebooks/
└── README.md
```

---

## 25. Major Findings

### Architecture matters as much as size
My strongest direction came not from simply changing decoder-model size, but from moving to an architecture naturally suited to source-to-target transformation.

### Specialization can compensate for capacity
A ~60.5M specialist became competitive with a ~600M general-purpose baseline on the narrow target task.

### Lower precision does not automatically improve speed
FP16, BF16, INT8, and INT4 performance depended heavily on hardware and kernels — a lesson that repeated itself yet again with ONNX Runtime.

### Different generation does not equal worse generation
Quantized models often produced different sequences while maintaining comparable semantic metrics — though I'd later find this same idea has a sharp edge: a "different but fine" token choice can occasionally be a "different and wrong" one, if it happens to be the EOS token.

### KV cache is a central serving resource
It saves repeated computation at the cost of per-request memory, and getting its reuse logic exactly right matters — I found this out the hard way while building the ONNX backend.

### Disk size, model footprint, and process RAM are different
A 73.78 MB checkpoint can occupy ~115 MB as a live model and still result in a larger process RSS.

### Python is not automatically the bottleneck
The Rust/Candle experiment showed that tensor kernels and runtime design matter more than host-language overhead for this workload.

### Training and serving are separate engineering problems
A trained checkpoint still needs serialization, quantization, validation, prefill, decoding, cache management, streaming, cleanup, APIs, and deployment infrastructure.

### A leaner runtime doesn't automatically mean a smaller deployment
This is the headline finding from Section 27: swapping `torch`+`transformers` for `onnxruntime` shrank the *library* footprint by roughly 3x, but the *model weights* actually grew, so the total deployment size only improved modestly.

---

## 26. Final Technology Choices

| Area | Final Direction |
|---|---|
| Task | Text → bullet points |
| Architecture | T5 encoder-decoder |
| Parameters | ~60.5M |
| Quantization | TorchAO INT8 weight-only |
| Runtime | PyTorch |
| Hardware target | CPU |
| Backend | FastAPI |
| Validation | Pydantic + token context validation |
| Streaming | Async streaming response |
| Decoder state | Per-request KV cache |
| Cleanup | Guaranteed after request |
| Local environment | uv |
| Deployment | Docker |
| Frontend | Streaming web application |

This is the production configuration. The next chapter is a separate, later experiment layered on top of it — not a replacement.

---

## 27. The ONNX Experiment: Torch vs ONNX Runtime

Everything up to this point was about getting the model right and serving it well on a normal server. But once I had a working backend, a new question came up: **could I deploy this somewhere like Vercel** — a serverless platform — **instead of a persistent server?**

That question turned into its own multi-day investigation, with real bugs, real dead ends, and a real answer at the end. Here's how it went.

### 27.1 Why I looked at ONNX at all

My `backend/` depends on `torch` + `transformers` + `torchao`. That combination is roughly **300MB** of libraries before I even add the 73.78MB model. Serverless platforms have tight function-size limits, and shipping 300MB+ of Python libraries just to run a 60M-parameter model felt disproportionate.

`onnxruntime` doesn't need `torch` or `transformers` at inference time — those heavy libraries are only needed *once*, offline, to export a model into the ONNX format. That was the appeal: convert once, deploy something much smaller forever after.

### 27.2 Converting the model wasn't simple

My first assumption — that I could just export the already-quantized TorchAO checkpoint straight to ONNX — turned out to be wrong. TorchAO's `Int8WeightOnlyConfig` wraps weights in a custom PyTorch tensor subclass that `torch.onnx.export` can't trace directly.

So I had to split the conversion into two phases:

```text
Phase 1: torchao checkpoint → dequantize → plain float checkpoint
Phase 2: plain checkpoint → export to ONNX → re-quantize with ONNX Runtime's own INT8 quantizer
```

This alone was already a departure from what I expected — I couldn't just "reuse" the TorchAO quantization, I had to throw it away and requantize from scratch using a completely different tool.

### 27.3 A wall of real dependency conflicts

Getting these two phases working was its own saga. In order:

1. `optimum-onnx` (the library that provides the convenient `ORTModelForSeq2SeqLM` export/inference API) hard-pins `transformers<5`. But loading my TorchAO checkpoint requires `transformers>=5.17.0` — the same version my production backend already uses. These two requirements are simply incompatible in one environment.
2. I tried splitting them into `uv` optional-dependency groups in one `pyproject.toml`. That failed too — `uv` resolves *all* declared extras jointly for lockfile consistency, even when you only ask to sync one of them. It correctly reported the two groups as mutually unsatisfiable.
3. The fix was two fully separate virtual environments — one for dequantizing (`transformers>=5.17.0`), one for exporting (`optimum` + an older `transformers`) — handing off a plain checkpoint on disk between them.
4. Even saving the dequantized checkpoint hit a wall: `transformers`' own `save_pretrained()` tried to re-apply TorchAO's weight-packing conversion and crashed with `NotImplementedError`, even after the weights were already plain floats. I had to write the safetensors state dict directly instead of going through that codepath.
5. Every recent `optimum` release I tried was broken against my `transformers` version in a different way — a removed internal API on one side, a constructor signature clash on the other. The only combination that actually worked end-to-end was an **older, pre-split `optimum==1.19.0`** with `transformers==4.39.3`.
6. That older `optimum` assumed torch's *legacy* ONNX exporter's temp-file naming, which didn't match torch 2.3+'s newer exporter. I had to pin `torch==2.2.2` specifically for the export step (which also meant Python 3.12 — no 3.13+ wheels exist for that torch version).
7. Merging the decoder's with/without-cache branches into one file (to avoid duplicating weights) silently produced a *larger* file unless `accelerate` was installed — the weight-deduplication pass it depends on failed silently, with only a warning, not an error.
8. Finally, the export environment's old `transformers` couldn't even parse my checkpoint's newer `tokenizer.json` format, and I couldn't just upgrade the `tokenizers` library either — that broke `transformers`' own import-time version check the other way. I ended up not tokenizing in that environment at all, and handing off plain token-ID JSON between environments instead.

None of this was a fundamental blocker — it was ecosystem fragmentation, version skew, and a genuinely under-documented export path. But it took real, patient debugging to get a working `.onnx` model out the other end.

### 27.4 The size result surprised me — and not in the direction I expected

Once I had working ONNX files, I measured everything rather than assuming.

| | Size |
|---|---|
| torchao source (`backend/artifacts/text-to-bullets-int8`) | 76.1 MB |
| ONNX INT8 output (`text-to-bullets-onnx-int8/`) | 193.0 MB |

The ONNX weights came out **larger**, not smaller. The reason: ONNX Runtime's dynamic quantization only quantizes `MatMul`/`Gemm` weights — it doesn't touch embedding tables, which use a `Gather` op. My tied vocab embedding (~16.4M params, ~65MB) stayed FP32 and got duplicated across both the encoder and decoder ONNX files, since ONNX has no mechanism to share weights across separate files the way a single PyTorch module can.

The library size, on the other hand, was exactly the win I'd hoped for:

| | Unpacked | Compressed |
|---|---|---|
| `onnxruntime` + `numpy` | 100 MB | 27 MB |
| `torch` + `transformers` | ~300 MB | — |

So the total picture was more nuanced than I expected: total deployment size went from ~376MB (torch+transformers+torchao weights) to ~293MB (onnxruntime+numpy+ONNX weights) — a real improvement, but a much smaller one than the library size alone suggested, because the weight-size regression ate into the gain.

### 27.5 Latency: a genuine trade-off, not a clean win

I measured a single representative request on both backends:

| Metric | torch (torchao) | ONNX (ORT) |
|---|---:|---:|
| Load time | 2.337 s | **0.389 s** (6x faster) |
| Cold generate (1st call) | 0.504 s | 2.520 s (5x slower) |
| Warm generate (2nd+ call) | 0.398 s | **0.266 s** (33% faster) |

ONNX Runtime loads almost instantly — it's just deserializing a graph — but pays a large one-time cost on its *first* inference call, apparently doing session/graph optimization lazily rather than at load time. Once warmed up, it's meaningfully faster per request than torch.

That means the right answer depends entirely on deployment shape: on a serverless cold start (load + one request, then the instance dies), the two are roughly a wash. On a persistent warm server handling many requests per loaded instance — which is how my `backend/` already runs — ONNX wins clearly after the first request.

### 27.6 Building the actual server surfaced a real bug

Getting accuracy validation to pass with `optimum`'s high-level `.generate()` API was one thing. Building a real FastAPI server with true token-by-token streaming — matching my `backend/`'s exact prefill/decode/KV-cache architecture, but driven by raw `onnxruntime.InferenceSession` calls instead of torch — was another, and it surfaced a bug my validation scripts never would have caught.

The merged decoder graph exposes a `use_cache_branch` flag. I assumed that when `True`, its `present.*.encoder.key/value` outputs were the cross-attention KV cache being correctly passed through. They weren't. I found this in the export logs after the fact: *"Adding a constant output for present.0.encoder.key of shape [0, 8, 1, 64]"* — the cached branch never actually computes real cross-attention KV at all; it just returns a fixed-shape placeholder to satisfy ONNX's requirement that both branches of an `If` node expose identical output signatures.

My server was overwriting its real encoder KV cache with this garbage on every step after the first, and generation reliably broke down after 1–2 tokens. The fix: cache the real encoder KV once, from the very first (`use_cache_branch=False`) step, and never let any later step's output overwrite it.

### 27.7 A second, subtler bug: quantization precision flipping a real decision

After fixing the KV-cache bug, a short test input matched the torch backend's output *exactly* — byte-for-byte identical. I thought I was done. Then a longer, multi-bullet input told a different story: the ONNX backend generated only the first bullet and stopped, while torch correctly produced all four.

I instrumented the decode loop to inspect the actual logits at the failure point, and found the real cause:

```text
EOS score:               21.09
<BULLET> (continue) score: 20.59
```

A margin of **0.5** — essentially a coin flip. ONNX Runtime's dynamic INT8 quantization (unlike TorchAO's calibrated, per-channel weight-only quantization) is measurably less precise, and on most tokens that doesn't matter. But greedy decoding takes whichever logit is higher, so a close call like this one gets amplified into an all-or-nothing outcome: continue generating, or stop for good.

The fix I landed on: suppress the EOS logit until a minimum number of tokens has been generated — `MIN_NEW_TOKENS = 30`, which not coincidentally matches the base T5 model's own `task_specific_params.summarization.min_length` in its config. After that fix, the same input produced all four bullets, matching torch's output word-for-word.

### 27.8 A finding that turned out not to be ONNX-specific at all

While testing, I gave both backends a much longer input that concatenated three unrelated articles — the Internet, the Roman Empire, and DNA — into one request. Both backends produced degraded, partially garbled output, each hallucinating a final bullet that spliced together sentence fragments from two different topics.

This wasn't a backend bug at all — it's a genuine capability limit of a 60M-parameter single-document summarizer. It was never trained to coherently summarize three unrelated documents glued together in one request. I added a note to the frontend UI ("best results with a single topic or document") rather than trying to engineer around a limitation that isn't really fixable at the serving layer.

### 27.9 Where this leaves things

| | torch (`backend/`) | ONNX (`backend-onnx/`) |
|---|---|---|
| Runtime deps | `torch` + `transformers` (~300MB) | `onnxruntime` + `tokenizers` (~100MB) |
| Total deployment size | ~376MB | ~293MB |
| Accuracy (single-topic input, after fixes) | reference | exact match |
| Warm-server latency | reference | ~33% faster |
| Cold-start latency | reference | roughly a wash |
| Streaming architecture | prefill/decode/KV cache | same, reimplemented against raw ONNX Runtime sessions |
| Production-ready | yes | validated locally, not yet load-tested |

The ONNX path is real and it works, but it cost two genuine bugs to get there — one in KV-cache handling, one in decoding precision — and it didn't deliver the dramatic size win I originally expected, because the weight file itself grew even as the library shrank. It's a legitimate option for a size- and cold-start-sensitive serverless deployment; it is not a strictly-better replacement for the torch backend.

---

## 28. Future Work

V1 already covers the fundamental single-request inference path:

```text
Fine-tuned T5
→ TorchAO INT8
→ context validation
→ encoder prefill
→ manual decode
→ per-request KV cache
→ streaming
→ cache cleanup
→ FastAPI
→ Docker
→ frontend
→ real TTFT / ITL / latency metrics
```

The natural next step (V2) isn't more model features — it's turning this into a
**concurrent inference engine**. Model retraining stays out of scope through V2; the
current model is already sufficient to learn the harder problem of serving it well
under concurrent load.

### V2 — Concurrent Inference & Scheduling

1. **Request queue** — API requests enter a controlled inference queue rather than every
   FastAPI coroutine independently hitting the model.
2. **Inference scheduler** — explicitly manage requests through states:
   ```text
   WAITING → PREFILL → DECODING → FINISHED
   ```
3. **Multiple active `GenerationState`s** — each request retains its own encoder outputs,
   KV cache, generated tokens, timings, and cancellation state.
4. **Decode interleaving** — instead of completing A before B:
   ```text
   A1 → B1 → C1 → A2 → B2 → C2 → ...
   ```
   A good intermediate step before batching.
5. **Request cancellation** — when the user hits Stop or disconnects, stop generation
   immediately and free that request's encoder/KV state.
6. **Timeouts** — prevent pathological requests from occupying inference capacity
   indefinitely.
7. **Admission control** — limits such as `MAX_ACTIVE_REQUESTS`, `MAX_QUEUED_REQUESTS`,
   `MAX_TOTAL_KV_MEMORY`.
8. **Backpressure / overload handling** — return a clean `429`/`503` instead of letting
   unlimited requests exhaust RAM.
9. **Queue metrics** — separate queue time, prefill time, TTFT, decode time, ITL, and
   end-to-end latency; these become far more meaningful under load.
10. **Load testing** — benchmark 1, 2, 4, 8, 16 concurrent users; report throughput,
    TTFT p50/p95, latency p50/p95, RAM, and requests/sec.

### V3 — Batching

Once V2 works, move to actual inference optimization:

```text
Dynamic batching → batched prefill → batched decode → continuous batching
```

Instead of:
```text
Model(A)
Model(B)
Model(C)
```
the goal is:
```text
Model([A, B, C])
        ↓
 tokenA tokenB tokenC
```
And when B finishes, a new request slots in:
```text
[A, B, C] → B finishes → [A, D, C]
```
That's **continuous batching** — building it by hand is what makes clear why systems
like vLLM/SGLang need real schedulers.

### V4 — Observability & Reliability

Make the server operationally mature:

- Prometheus metrics, Grafana dashboard, OpenTelemetry traces
- structured JSON logs, request IDs across frontend/backend
- `/health`, `/live`, `/ready`
- memory/RSS monitoring, KV-cache memory monitoring
- queue depth, active generations, tokens/sec, error-rate metrics
- graceful shutdown, proper SIGTERM handling, container resource limits

### V5 — Performance Engineering

Once observability exists, optimize using actual measurements:

- `torch.compile` experiments, thread-count tuning, `OMP_NUM_THREADS`/MKL tuning
- CPU affinity experiments, preallocated buffers, reduced tensor allocations during decode
- tokenizer profiling, quantization kernel comparison, BF16 vs INT8 across CPU architectures
- PyTorch Profiler, flame graphs, memory profiling
- revisit Candle/ONNX/OpenVINO/llama.cpp-compatible approaches, only where the
  architecture/runtime supports them cleanly

### V6 — Production Platform Features

- API keys, per-key rate limits, usage quotas, request-size limits
- CORS policy, authentication, HTTPS
- model/version metadata endpoint, API versioning, deployment revisions, rollback
- CI/CD, automated Docker builds, benchmark regression tests, model-quality regression tests

### The full progression

```text
V1
Single-request inference
Prefill + Decode + KV Cache + Streaming
              ↓
V2
Queue + Scheduler + Concurrency + Cancellation
              ↓
V3
Dynamic/Continuous Batching
              ↓
V4
Observability + Reliability
              ↓
V5
CPU Performance Engineering
              ↓
V6
Production Platform
```

This takes the project from "I deployed a fine-tuned model" toward "I built and
understood an LLM inference serving system."

### Other potential experiments (not part of the V2–V6 line above)

- comparison with specialized inference runtimes;
- static (rather than dynamic) ONNX quantization, or explicit `Gather`/embedding quantization, to actually shrink the ONNX weight file instead of growing it;
- a production tokenizer path for the ONNX backend that never touches `transformers` at all;
- deploying the ONNX backend to a real serverless platform and measuring actual cold-start behavior under load, not just a local approximation of it.

---

# Conclusion

Looking back, this project evolved through:

```text
BitNet / Qwen / SmolLM comparison
              ↓
small decoder-only fine-tuning
              ↓
architecture limitations
              ↓
encoder-decoder T5
              ↓
quality evaluation
              ↓
FP32 / FP16 / BF16
              ↓
INT8 / INT4 experiments
              ↓
TorchAO INT8
              ↓
CPU benchmarking
              ↓
Rust/Candle experiment
              ↓
manual prefill + decode
              ↓
KV-cache analysis
              ↓
token streaming
              ↓
FastAPI backend
              ↓
web frontend
              ↓
ONNX conversion attempt
              ↓
dependency and export debugging
              ↓
KV-cache and quantization-precision bugs
              ↓
torch vs ONNX serving comparison
```

The central lesson I keep relearning, in a different form every few chapters, is that deployment cannot be reduced to one variable. It is not simply:

```text
bigger model = better
INT8 = faster
native language = faster
leaner library = smaller deployment
```

Real behavior emerges from the interaction of:

```text
task
+ dataset
+ architecture
+ training
+ parameter count
+ numerical precision
+ quantization method
+ hardware
+ kernels
+ runtime
+ KV-cache management
+ serving architecture
+ export tooling and its own version constraints
```

For this task, a small specialized encoder-decoder model proved to be a compelling approach — and later, a second serving path built on ONNX Runtime proved to be a real but genuinely trade-off-laden alternative, not a free upgrade.

The project demonstrates the complete path from:

> **model experimentation → fine-tuning → evaluation → optimization → inference internals → production serving → user-facing application → alternative deployment exploration**

and, importantly, documents *why* each engineering decision was made — including the ones that didn't pan out the way I expected — instead of treating inference as a black box.
