"""Decode the ONNX model's output token IDs using the working (dequant-venv) tokenizer.

    .venv-dequant/bin/python decode_output.py
"""
import json
from pathlib import Path

SOURCE_MODEL_PATH = str(Path(__file__).parent.parent.parent / "backend" / "artifacts" / "text-to-bullets-int8")
IN_FILE = Path(__file__).parent.parent / "artifacts" / "onnx_output_ids.json"
OUT_FILE = Path(__file__).parent.parent / "artifacts" / "onnx_output.json"


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_PATH)
    data = json.loads(IN_FILE.read_text())
    output_text = tokenizer.decode(data["output_ids"], skip_special_tokens=False)

    result = {"output": output_text}
    for key in ("load_time_s", "cold_generate_s", "warm_generate_s", "output_tokens"):
        if key in data:
            result[key] = data[key]

    OUT_FILE.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT_FILE}")
    print(output_text)


if __name__ == "__main__":
    main()
