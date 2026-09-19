"""Run the original torchao INT8 model on the test input, save output to JSON.

Runs under the `dequant` extra (needs transformers>=5.17.0 to load this checkpoint).
Reads ../backend/artifacts/text-to-bullets-int8 read-only.

    uv sync --extra dequant
    uv run python run_torch_baseline.py
"""
import json
import time
from pathlib import Path

SOURCE_MODEL_PATH = str(Path(__file__).parent.parent.parent / "backend" / "artifacts" / "text-to-bullets-int8")
OUT_FILE = Path(__file__).parent.parent / "artifacts" / "torch_output.json"

# Same prompts as backend/src/constants/prompts.py (literal copy, not imported,
# so this script has zero dependency on backend's source tree).
SYSTEM_PROMPT = """You are an expert at converting text into concise, actionable bullet points.
Transform the input text into a clear list of key points using the following format:
- Each point should be a complete, standalone thought
- Use simple, direct language
- Remove redundancy
- Focus on the most important information
- Mark each point with <BULLET>"""
TASK_PROMPT = "Convert the following text into bullet points:\n"

# Same example text as frontend/app/page.tsx's "Example" button.
EXAMPLE_TEXT = (
    "Acme reported quarterly revenue of $4.2 billion, up 12% year over year. "
    "Operating profit increased 8% to $620 million, although operating margin "
    "declined from 17.2% to 14.8%. The company added 1.3 million customers "
    "during the quarter and raised full-year revenue guidance from $16 billion "
    "to $17.5 billion. Management warned that European demand weakened in July."
)

GENERATE_KWARGS = dict(max_new_tokens=512, num_beams=1, do_sample=False)


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    full_text = f"{SYSTEM_PROMPT}\n\n{TASK_PROMPT}{EXAMPLE_TEXT}"

    print("Loading torchao INT8 model...")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(SOURCE_MODEL_PATH)
    model.eval()
    load_time = time.time() - load_start

    inputs = tokenizer(full_text, return_tensors="pt")

    # Cold run (first call pays any lazy-init cost) and a warm run, to report both.
    cold_start = time.time()
    output_ids = model.generate(**inputs, **GENERATE_KWARGS)
    cold_time = time.time() - cold_start

    warm_start = time.time()
    model.generate(**inputs, **GENERATE_KWARGS)
    warm_time = time.time() - warm_start

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    num_output_tokens = output_ids.shape[1]

    result = {
        "output": output_text,
        "load_time_s": round(load_time, 3),
        "cold_generate_s": round(cold_time, 3),
        "warm_generate_s": round(warm_time, 3),
        "output_tokens": num_output_tokens,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT_FILE}")
    print(output_text)
    print(f"load={load_time:.3f}s cold_generate={cold_time:.3f}s warm_generate={warm_time:.3f}s output_tokens={num_output_tokens}")


if __name__ == "__main__":
    main()
