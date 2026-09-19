import asyncio
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "text-to-bullets-int8"

MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 256
BULLET_TOKEN = "<BULLET>"

# EDUCATIONAL MODE:
# Pause after every generated token so KV-cache growth is visible.
# Set False later for real performance measurements.
SHOW_KV_CACHE = True
KV_STEP_DELAY_SECONDS = 0.5
KV_LAYERS_TO_SHOW = 2

# ============================================================
# SERVER-CONTROLLED TASK PROMPT
#
# Frontend will eventually send ONLY user text.
# The backend owns this instruction, similar to a system prompt.
# ============================================================

TASK_INSTRUCTION = """
Convert the following English text into concise bullet points containing all materially important information.

Follow these rules:
- Extract all important and independently useful points.
- The number of bullets must depend entirely on the information in the text.
- Never use a fixed number of bullets.
- Use one bullet for each distinct important point.
- Combine details that naturally belong together.
- Remove repetition, filler, metadata, boilerplate, and trivial details.
- Do not repeat the same information in multiple bullets.
- Preserve important names, dates, numbers, quantities, comparisons, causes, conditions, decisions, and conclusions.
- Do not add, infer, or assume information that is not supported by the source text.
- Do not turn contextual information into new advice or recommendations.
- Keep every bullet concise while preserving the original meaning.
- Return only bullet points.
- Start every bullet with "- ".

Text:
""".strip()

TEST_TEXT = """
Acme reported quarterly revenue of $4.2 billion, up 12% year over year.
Operating profit increased 8% to $620 million, although operating margin
declined from 17.2% to 14.8%. The company added 1.3 million customers
during the quarter and raised full-year revenue guidance from $16 billion
to $17.5 billion. Management warned that European demand weakened in July.
""".strip()

# ============================================================
# GENERATION STATE
#
# MODEL:
#   One shared model can serve many requests.
#
# STATE:
#   Every request gets its OWN GenerationState.
#
# Request A -> encoder output A -> KV cache A
# Request B -> encoder output B -> KV cache B
#
# Never share past_key_values between independent requests.
# ============================================================

@dataclass
class GenerationState:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    input_tokens: int

    # Produced once by encoder prefill.
    encoder_outputs: Any = None

    # Decoder KV cache.
    # None initially; created by first decode step.
    past_key_values: Any = None

    # First = decoder_start_token.
    # Afterwards = ONLY newest generated token.
    decoder_input_ids: torch.Tensor | None = None

    generated_ids: list[int] = field(default_factory=list)
    token_timestamps: list[float] = field(default_factory=list)

    prefill_seconds: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    reached_eos: bool = False

# ============================================================
# INFERENCE ENGINE
# ============================================================

class InferenceEngine:
    def __init__(self):
        print("=" * 80)
        print("LOADING INFERENCE ENGINE")
        print("=" * 80)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        start = time.perf_counter()

        # Tokenizer + model are loaded ONCE when server starts.
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            device_map="cpu",
            dtype=torch.bfloat16,
        ).eval()

        int8_count = sum(
            "Int8Tensor" in type(getattr(m, "weight", None)).__name__
            for m in self.model.modules()
            if getattr(m, "weight", None) is not None
        )

        print(f"Model: {MODEL_PATH}")
        print(f"Load time: {time.perf_counter()-start:.3f}s")
        print(f"Model footprint: {self.model.get_memory_footprint()/1024**2:.2f} MB")
        print(f"TorchAO INT8 tensors: {int8_count}")

    # ========================================================
    # 1. CONTEXT VALIDATION
    #
    # We do NOT truncate.
    #
    # If:
    # system prompt + user text > model context limit
    #
    # reject the request instead of silently losing user text.
    # ========================================================

    def validate_context(self, user_text: str) -> tuple[dict, int]:
        if not user_text or not user_text.strip():
            raise ValueError("Input text cannot be empty.")

        model_input = f"{TASK_INSTRUCTION}\n{user_text.strip()}"

        encoded = self.tokenizer(
            model_input,
            return_tensors="pt",
            truncation=False,
        )

        input_tokens = int(encoded["input_ids"].shape[-1])

        print("\n[CONTEXT VALIDATION]")
        print(f"Input tokens : {input_tokens}")
        print(f"Maximum      : {MAX_INPUT_TOKENS}")

        if input_tokens > MAX_INPUT_TOKENS:
            raise ValueError(
                f"Context length exceeded: {input_tokens} tokens; "
                f"maximum is {MAX_INPUT_TOKENS}."
            )

        print("Status       : VALID")
        return encoded, input_tokens

    # ========================================================
    # 2. CREATE PER-REQUEST STATE
    #
    # At this point:
    #
    # encoder_outputs = None
    # KV cache        = None
    #
    # Nothing has been inferred yet.
    # ========================================================

    def create_state(self, user_text: str) -> GenerationState:
        encoded, input_tokens = self.validate_context(user_text)

        start_id = self.model.config.decoder_start_token_id
        if start_id is None:
            start_id = self.tokenizer.pad_token_id

        state = GenerationState(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            input_tokens=input_tokens,
            decoder_input_ids=torch.tensor([[start_id]], dtype=torch.long),
        )

        print("\n[REQUEST STATE CREATED]")
        print(f"decoder_start_token : {start_id}")
        print("encoder_outputs     : EMPTY")
        print("KV cache            : EMPTY")

        return state

    # ========================================================
    # 3. PREFILL
    #
    # T5 is encoder-decoder.
    #
    # system prompt + user text
    #             ↓
    #          ENCODER
    #             ↓
    #      encoder_outputs
    #
    # This runs ONCE per request.
    #
    # Unlike a decoder-only LLM, T5's source-text "prefill"
    # is primarily the encoder pass.
    # ========================================================

    def prefill(self, state: GenerationState) -> None:
        print("\n" + "=" * 80)
        print("PREFILL")
        print("=" * 80)
        print(f"Encoding {state.input_tokens} input tokens...")

        start = time.perf_counter()

        with torch.inference_mode():
            state.encoder_outputs = self.model.get_encoder()(
                input_ids=state.input_ids,
                attention_mask=state.attention_mask,
                return_dict=True,
            )

        state.prefill_seconds = time.perf_counter()-start

        hidden = state.encoder_outputs.last_hidden_state

        print(f"Encoder output shape : {tuple(hidden.shape)}")
        print(f"Encoder dtype        : {hidden.dtype}")
        print(f"Prefill time         : {state.prefill_seconds*1000:.2f} ms")
        print("KV cache             : STILL EMPTY")
        print("Reason: decoder has not run yet.")

    # ========================================================
    # CACHE MEMORY
    #
    # Recursively find tensors stored inside the HF cache.
    #
    # Memory = number of elements × bytes per element.
    #
    # This measures cache tensor storage, NOT entire process RAM.
    # ========================================================

    def tensor_bytes(self, obj: Any, seen: set[int] | None = None) -> int:
        if obj is None:
            return 0

        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        if torch.is_tensor(obj):
            return obj.numel()*obj.element_size()

        if isinstance(obj, dict):
            return sum(self.tensor_bytes(v, seen) for v in obj.values())

        if isinstance(obj, (list, tuple)):
            return sum(self.tensor_bytes(v, seen) for v in obj)

        total = 0

        # Transformers cache implementations differ by version.
        # Inspect the common cache containers without assuming
        # one exact implementation.
        for attr in (
            "layers",
            "self_attention_cache",
            "cross_attention_cache",
            "key_cache",
            "value_cache",
            "keys",
            "values",
        ):
            value = getattr(obj, attr, None)
            if value is not None:
                total += self.tensor_bytes(value, seen)

        return total

    # ========================================================
    # CACHE SEQUENCE LENGTH
    #
    # HF Cache objects often expose get_seq_length().
    # This tells us how many decoder positions are cached.
    # ========================================================

    def cache_seq_length(self, cache: Any) -> int | None:
        if cache is None:
            return 0

        # EncoderDecoderCache often wraps a self-attention cache.
        self_cache = getattr(cache, "self_attention_cache", None)

        for candidate in (self_cache, cache):
            if candidate is None:
                continue

            fn = getattr(candidate, "get_seq_length", None)

            if callable(fn):
                try:
                    return int(fn())
                except Exception:
                    pass

        return None

    # ========================================================
    # PRINT CACHE SHAPES
    #
    # Typical decoder self-attention cache:
    #
    # K: [batch, heads, cached_tokens, head_dim]
    # V: [batch, heads, cached_tokens, head_dim]
    #
    # cached_tokens grows:
    #
    # step 1 -> 1
    # step 2 -> 2
    # step 3 -> 3
    # ...
    #
    # T5 also has encoder-decoder cross-attention K/V.
    # Those correspond to encoder context and normally do not
    # grow with every generated decoder token.
    # ========================================================

    def print_cache_component(
        self,
        name: str,
        cache: Any,
    ) -> None:
        if cache is None:
            return

        layers = getattr(cache, "layers", None)

        if layers is not None:
            print(f"{name} layers: {len(layers)}")

            for i, layer in enumerate(layers[:KV_LAYERS_TO_SHOW]):
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)

                print(f"  Layer {i}:")

                if torch.is_tensor(keys):
                    print(f"    K {tuple(keys.shape)} {keys.dtype}")

                if torch.is_tensor(values):
                    print(f"    V {tuple(values.shape)} {values.dtype}")

                if not torch.is_tensor(keys) and not torch.is_tensor(values):
                    print(f"    stored as {type(layer).__name__}")

            return

        # Legacy tuple/list cache.
        if isinstance(cache, (list, tuple)):
            print(f"{name} layers: {len(cache)}")

            for i, layer in enumerate(cache[:KV_LAYERS_TO_SHOW]):
                print(f"  Layer {i}:")

                if not isinstance(layer, (list, tuple)):
                    continue

                names = ("self-K", "self-V", "cross-K", "cross-V")

                for j, tensor in enumerate(layer):
                    if torch.is_tensor(tensor):
                        label = names[j] if j < len(names) else f"tensor-{j}"
                        print(f"    {label}: {tuple(tensor.shape)} {tensor.dtype}")

    # ========================================================
    # 4. KV-CACHE INSPECTOR
    #
    # Called AFTER every decoder step.
    #
    # This lets us watch the request-local cache grow.
    # ========================================================

    def inspect_kv_cache(
        self,
        state: GenerationState,
        token_id: int,
        step: int,
    ) -> None:
        if not SHOW_KV_CACHE:
            return

        cache = state.past_key_values
        cache_bytes = self.tensor_bytes(cache)
        seq_len = self.cache_seq_length(cache)

        token_text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        print("\n" + "-" * 80)
        print(f"KV CACHE AFTER TOKEN {step}")
        print("-" * 80)
        print(f"Token ID            : {token_id}")
        print(f"Token piece         : {token_text!r}")
        print(f"Cache type          : {type(cache).__name__}")
        print(f"Cached decoder pos. : {seq_len if seq_len is not None else 'unknown'}")
        print(f"KV tensor memory    : {cache_bytes/1024:.2f} KB ({cache_bytes/1024**2:.4f} MB)")

        # New HF encoder-decoder cache has two logical pieces.
        self_cache = getattr(cache, "self_attention_cache", None)
        cross_cache = getattr(cache, "cross_attention_cache", None)

        if self_cache is not None:
            self_bytes = self.tensor_bytes(self_cache)
            print(f"\nSELF-ATTENTION CACHE : {self_bytes/1024:.2f} KB")
            print("This grows as decoder tokens are generated.")
            self.print_cache_component("Self-attention", self_cache)

        if cross_cache is not None:
            cross_bytes = self.tensor_bytes(cross_cache)
            print(f"\nCROSS-ATTENTION CACHE: {cross_bytes/1024:.2f} KB")
            print("This represents encoder context and is reused.")
            self.print_cache_component("Cross-attention", cross_cache)

        # Fallback for older cache representation.
        if self_cache is None and cross_cache is None:
            self.print_cache_component("KV cache", cache)

        print("-" * 80)

    # ========================================================
    # 5. ONE DECODE STEP
    #
    # FIRST STEP:
    #
    # decoder_start_token
    #        ↓
    # decoder
    #        ↓
    # next token + CREATE KV cache
    #
    # LATER STEPS:
    #
    # newest token + existing KV cache
    #        ↓
    # decoder
    #        ↓
    # next token + UPDATED KV cache
    #
    # Without KV cache:
    # decoder would repeatedly process:
    #
    # [token1]
    # [token1 token2]
    # [token1 token2 token3]
    # ...
    #
    # With KV cache:
    #
    # token1 + cache
    # token2 + cache
    # token3 + cache
    #
    # Old attention K/V values are reused.
    # ========================================================

    def decode(self, state: GenerationState) -> int:
        step = len(state.generated_ids)+1

        print("\n" + "=" * 80)
        print(f"DECODE STEP {step}")
        print("=" * 80)

        before_bytes = self.tensor_bytes(state.past_key_values)

        if state.past_key_values is None:
            print("KV before decode : EMPTY")
            print("Action           : CREATE cache")
        else:
            print(f"KV before decode : {before_bytes/1024:.2f} KB")
            print("Action           : REUSE + EXTEND cache")

        print(f"Decoder input    : {state.decoder_input_ids.tolist()}")
        print("Important        : only newest token is fed after step 1")

        start = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model(
                encoder_outputs=state.encoder_outputs,
                attention_mask=state.attention_mask,
                decoder_input_ids=state.decoder_input_ids,
                past_key_values=state.past_key_values,
                use_cache=True,
                return_dict=True,
            )

        decode_ms = (time.perf_counter()-start)*1000

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_token.item())

        # ----------------------------------------------------
        # THIS IS THE IMPORTANT KV-CACHE UPDATE.
        #
        # First decode:
        # None -> populated cache
        #
        # Every later decode:
        # old cache -> updated cache
        # ----------------------------------------------------

        state.past_key_values = outputs.past_key_values

        # Next model call gets ONLY the token just generated.
        state.decoder_input_ids = next_token

        state.token_timestamps.append(time.perf_counter())

        if token_id == self.tokenizer.eos_token_id:
            state.reached_eos = True
        else:
            state.generated_ids.append(token_id)

        print(f"Generated token  : {token_id}")
        print(f"Model decode time: {decode_ms:.2f} ms")

        self.inspect_kv_cache(
            state=state,
            token_id=token_id,
            step=step,
        )

        return token_id

    # ========================================================
    # 6. ASYNC TOKEN STREAM
    #
    # Future FastAPI endpoint can do:
    #
    # async for chunk in engine.stream(text):
    #     send chunk to browser via SSE
    #
    # IMPORTANT:
    # asyncio makes the STREAM API cooperative.
    # PyTorch CPU inference itself is still synchronous.
    # A scheduler/worker layer will handle real concurrency.
    # ========================================================

    async def stream(
        self,
        user_text: str,
    ) -> AsyncGenerator[str, None]:

        state = self.create_state(user_text)
        state.started_at = time.perf_counter()

        # Encoder executes exactly once.
        self.prefill(state)

        previous_text = ""

        for _ in range(MAX_NEW_TOKENS):
            token_id = self.decode(state)

            if state.reached_eos:
                print("\nEOS reached.")
                break

            # Decode all generated IDs cumulatively.
            # SentencePiece needs context to reconstruct spaces.
            current_text = self.tokenizer.decode(
                state.generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).replace(BULLET_TOKEN, "\n- ")

            if current_text.startswith(previous_text):
                new_text = current_text[len(previous_text):]
            else:
                new_text = current_text

            previous_text = current_text

            if new_text:
                yield new_text

            # ------------------------------------------------
            # EDUCATIONAL PAUSE
            #
            # We pause AFTER yielding the token, so you see:
            #
            # output token
            #      ↓
            # cache state
            #      ↓
            # wait
            #      ↓
            # next decode
            #
            # NEVER enable this in production.
            # ------------------------------------------------

            if SHOW_KV_CACHE:
                print(f"\n\n[PAUSE {KV_STEP_DELAY_SECONDS}s — inspect cache above]")
                await asyncio.sleep(KV_STEP_DELAY_SECONDS)
            else:
                await asyncio.sleep(0)

        state.finished_at = time.perf_counter()
        self.last_state = state

    # ========================================================
    # 7. REQUEST METRICS
    #
    # NOTE:
    # When SHOW_KV_CACHE=True, ITL/total latency include the
    # artificial educational 0.5-second pauses.
    #
    # Turn visualization OFF for real benchmarking.
    # ========================================================

    def metrics(self, state: GenerationState) -> dict:
        timestamps = state.token_timestamps

        ttft = (
            timestamps[0]-state.started_at
            if timestamps else float("nan")
        )

        itls = [
            (timestamps[i]-timestamps[i-1])*1000
            for i in range(1, len(timestamps))
        ]

        total = state.finished_at-state.started_at

        return {
            "input_tokens": state.input_tokens,
            "output_tokens": len(state.generated_ids),
            "reached_eos": state.reached_eos,
            "prefill_seconds": state.prefill_seconds,
            "ttft_seconds": ttft,
            "mean_itl_ms": statistics.mean(itls) if itls else float("nan"),
            "median_itl_ms": statistics.median(itls) if itls else float("nan"),
            "max_itl_ms": max(itls) if itls else float("nan"),
            "final_kv_cache_mb": self.tensor_bytes(state.past_key_values)/1024**2,
            "total_latency_seconds": total,
        }

# ============================================================
# TERMINAL TEST
# ============================================================

async def test_stream(engine: InferenceEngine):
    print("\n" + "#" * 80)
    print("ONE REQUEST — WATCH PREFILL + KV CACHE + STREAMING")
    print("#" * 80)

    async for chunk in engine.stream(TEST_TEXT):
        print(f"\n[STREAMED TO CLIENT] {chunk!r}", flush=True)

    print("\n" + "#" * 80)
    print("FINAL GENERATED TEXT")
    print("#" * 80)

    state = engine.last_state

    final_text = engine.tokenizer.decode(
        state.generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).replace(BULLET_TOKEN, "\n- ")

    print(final_text)

    print("\n" + "#" * 80)
    print("REQUEST METRICS")
    print("#" * 80)

    for key, value in engine.metrics(state).items():
        if isinstance(value, float):
            print(f"{key:<28}: {value:.4f}")
        else:
            print(f"{key:<28}: {value}")

# ============================================================
# CONTEXT-LIMIT DEMO
# ============================================================

def test_context_limit(engine: InferenceEngine):
    print("\n" + "#" * 80)
    print("CONTEXT LIMIT TEST")
    print("#" * 80)

    try:
        engine.validate_context("hello "*5000)
    except ValueError as exc:
        print(f"Rejected correctly: {exc}")

# ============================================================
# MAIN
# ============================================================

async def main():
    # Shared model loads once.
    engine = InferenceEngine()

    # One request gets its own GenerationState + KV cache.
    await test_stream(engine)

    # Show production-style oversized-context rejection.
    test_context_limit(engine)

if __name__ == "__main__":
    asyncio.run(main())