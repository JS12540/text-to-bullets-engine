"""Step 1/2: load the torchao-quantized checkpoint and save a plain-float copy.

Reads (read-only) from ../backend/artifacts/text-to-bullets-int8 — never writes there.
Writes a plain (non-quantized) checkpoint to ./artifacts/_tmp_fp32_checkpoint.

torchao's Int8WeightOnlyConfig wraps weights in a custom tensor subclass that
export.py's optimum-onnx dependency can't load directly (optimum-onnx hard-pins
transformers<5, but this checkpoint needs transformers>=5.17.0 to load at all).
This script runs in the `dequant` dependency group and hands off a plain
checkpoint that any transformers version can load.

Run with:
    uv sync --extra dequant
    uv run python dequantize.py
"""
from pathlib import Path

SOURCE_MODEL_PATH = str(Path(__file__).parent.parent.parent / "backend" / "artifacts" / "text-to-bullets-int8")
TMP_FP_CHECKPOINT = Path(__file__).parent.parent / "artifacts" / "_tmp_fp32_checkpoint"


def main():
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading torchao-quantized checkpoint from {SOURCE_MODEL_PATH} ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(SOURCE_MODEL_PATH)
    model.eval()

    dequantized_count = 0
    for module in model.modules():
        if hasattr(module, "weight") and hasattr(module.weight, "dequantize"):
            with torch.no_grad():
                module.weight = torch.nn.Parameter(module.weight.dequantize())
            dequantized_count += 1
    print(f"Dequantized {dequantized_count} weight tensors.")

    TMP_FP_CHECKPOINT.mkdir(parents=True, exist_ok=True)

    # save_pretrained() tries to re-apply the torchao-specific weight-packing
    # reversal on save, which isn't implemented for this codepath and raises
    # NotImplementedError even after dequantizing. Since weights are now plain
    # floats with standard T5 key names, write the safetensors state dict
    # directly and save config/tokenizer separately to sidestep that codepath.
    state_dict = model.state_dict()
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(state_dict, TMP_FP_CHECKPOINT / "model.safetensors", metadata={"format": "pt"})

    # Delete (not just null-out) quantization_config: a `null` value in the
    # saved config.json gets reloaded as a literal None that crashes
    # transformers' own config.to_dict() (`None.to_dict()`) downstream.
    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config
    model.config.save_pretrained(TMP_FP_CHECKPOINT)
    model.generation_config.save_pretrained(TMP_FP_CHECKPOINT)
    AutoTokenizer.from_pretrained(SOURCE_MODEL_PATH).save_pretrained(TMP_FP_CHECKPOINT)

    print(f"Plain checkpoint written to {TMP_FP_CHECKPOINT}")
    print("Next: uv sync --extra export && uv run python export_onnx.py")


if __name__ == "__main__":
    main()
