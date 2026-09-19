"""Run the exported ONNX INT8 model on pre-tokenized input, save output IDs to JSON.

Runs under the export venv (.venv-export). Takes token IDs from
artifacts/tokenized_input.json (produced by tokenize_input.py in the dequant
venv) instead of using AutoTokenizer directly — this venv's old
transformers/tokenizers can't parse this checkpoint's tokenizer.json format.

    .venv-export/bin/python run_onnx.py
"""
import json
import time
from pathlib import Path

import torch

ONNX_MODEL_PATH = str(Path(__file__).parent.parent / "artifacts" / "text-to-bullets-onnx-int8")
IN_FILE = Path(__file__).parent.parent / "artifacts" / "tokenized_input.json"
OUT_FILE = Path(__file__).parent.parent / "artifacts" / "onnx_output_ids.json"

GENERATE_KWARGS = dict(max_new_tokens=512, num_beams=1, do_sample=False)


def main():
    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    if not Path(ONNX_MODEL_PATH).exists():
        raise RuntimeError(f"{ONNX_MODEL_PATH} not found. Run export_onnx.py first.")
    if not IN_FILE.exists():
        raise RuntimeError(f"{IN_FILE} not found. Run tokenize_input.py (in .venv-dequant) first.")

    inputs = json.loads(IN_FILE.read_text())
    input_ids = torch.tensor([inputs["input_ids"]])
    attention_mask = torch.tensor([inputs["attention_mask"]])

    print("Loading ONNX INT8 model...")
    load_start = time.time()
    model = ORTModelForSeq2SeqLM.from_pretrained(ONNX_MODEL_PATH)
    load_time = time.time() - load_start

    cold_start = time.time()
    output_ids = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, **GENERATE_KWARGS
    )
    cold_time = time.time() - cold_start

    warm_start = time.time()
    model.generate(input_ids=input_ids, attention_mask=attention_mask, **GENERATE_KWARGS)
    warm_time = time.time() - warm_start

    num_output_tokens = output_ids.shape[1]

    result = {
        "output_ids": output_ids[0].tolist(),
        "load_time_s": round(load_time, 3),
        "cold_generate_s": round(cold_time, 3),
        "warm_generate_s": round(warm_time, 3),
        "output_tokens": num_output_tokens,
    }

    OUT_FILE.write_text(json.dumps(result))
    print(f"Wrote {OUT_FILE}")
    print(f"load={load_time:.3f}s cold_generate={cold_time:.3f}s warm_generate={warm_time:.3f}s output_tokens={num_output_tokens}")


if __name__ == "__main__":
    main()
