"""Tokenize the test input using the working (dequant-venv) tokenizer, save IDs to JSON.

The export venv's old transformers/tokenizers can't parse this tokenizer.json
format (see README.md's "Known issues" section), so tokenization/decoding
happens here instead and hands off plain token-ID lists via JSON.

    .venv-dequant/bin/python tokenize_input.py
"""
import json
from pathlib import Path

SOURCE_MODEL_PATH = str(Path(__file__).parent.parent.parent / "backend" / "artifacts" / "text-to-bullets-int8")
OUT_FILE = Path(__file__).parent.parent / "artifacts" / "tokenized_input.json"

SYSTEM_PROMPT = """You are an expert at converting text into concise, actionable bullet points.
Transform the input text into a clear list of key points using the following format:
- Each point should be a complete, standalone thought
- Use simple, direct language
- Remove redundancy
- Focus on the most important information
- Mark each point with <BULLET>"""
TASK_PROMPT = "Convert the following text into bullet points:\n"

EXAMPLE_TEXT = (
    "Acme reported quarterly revenue of $4.2 billion, up 12% year over year. "
    "Operating profit increased 8% to $620 million, although operating margin "
    "declined from 17.2% to 14.8%. The company added 1.3 million customers "
    "during the quarter and raised full-year revenue guidance from $16 billion "
    "to $17.5 billion. Management warned that European demand weakened in July."
)


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_PATH)
    full_text = f"{SYSTEM_PROMPT}\n\n{TASK_PROMPT}{EXAMPLE_TEXT}"
    inputs = tokenizer(full_text, return_tensors=None)
    payload = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload))
    print(f"Wrote {OUT_FILE}: {len(payload['input_ids'])} input tokens")


if __name__ == "__main__":
    main()
