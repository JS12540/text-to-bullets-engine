"""Step 2/2: export the plain checkpoint (from dequantize.py) to ONNX, then INT8-quantize it.

Reads ./artifacts/_tmp_fp32_checkpoint (produced by dequantize.py).
Writes ./artifacts/text-to-bullets-onnx-int8.

Runs in the `export` dependency group (optimum-onnx), which is incompatible
with the `dequant` group in the same environment — see README.md.

Run with:
    uv sync --extra export
    uv run python export_onnx.py
"""
import shutil
from pathlib import Path

TMP_FP_CHECKPOINT = Path(__file__).parent.parent / "artifacts" / "_tmp_fp32_checkpoint"
ONNX_FP_DIR = Path(__file__).parent.parent / "artifacts" / "_onnx_fp32"
ONNX_INT8_DIR = Path(__file__).parent.parent / "artifacts" / "text-to-bullets-onnx-int8"
SOURCE_MODEL_PATH = Path(__file__).parent.parent.parent / "backend" / "artifacts" / "text-to-bullets-int8"


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / (1024 * 1024)


def export_to_onnx():
    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    if not TMP_FP_CHECKPOINT.exists():
        raise RuntimeError(
            f"{TMP_FP_CHECKPOINT} not found. Run dequantize.py first (under the `dequant` extra)."
        )

    print(f"Exporting {TMP_FP_CHECKPOINT} to ONNX ...")
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(
        str(TMP_FP_CHECKPOINT), export=True, use_merged=True
    )
    ONNX_FP_DIR.mkdir(parents=True, exist_ok=True)
    ort_model.save_pretrained(ONNX_FP_DIR)


def quantize_onnx_graphs():
    from onnxruntime.quantization import quantize_dynamic, QuantType

    ONNX_INT8_DIR.mkdir(parents=True, exist_ok=True)

    onnx_files = list(ONNX_FP_DIR.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"No .onnx files found in {ONNX_FP_DIR}")

    for onnx_file in onnx_files:
        out_path = ONNX_INT8_DIR / onnx_file.name
        print(f"Quantizing {onnx_file.name} -> INT8...")
        quantize_dynamic(
            model_input=str(onnx_file),
            model_output=str(out_path),
            weight_type=QuantType.QInt8,
        )

    for f in ONNX_FP_DIR.glob("*"):
        if f.suffix != ".onnx":
            shutil.copy(f, ONNX_INT8_DIR / f.name)

    # ort_model.save_pretrained() only saves model/config files, not the
    # tokenizer. Copy tokenizer files from the plain checkpoint directly.
    for name in ("tokenizer.json", "tokenizer_config.json"):
        src = TMP_FP_CHECKPOINT / name
        if src.exists():
            shutil.copy(src, ONNX_INT8_DIR / name)


def main():
    export_to_onnx()
    quantize_onnx_graphs()

    shutil.rmtree(ONNX_FP_DIR, ignore_errors=True)
    shutil.rmtree(TMP_FP_CHECKPOINT, ignore_errors=True)

    source_size = dir_size_mb(SOURCE_MODEL_PATH)
    onnx_size = dir_size_mb(ONNX_INT8_DIR)

    print()
    print("=" * 60)
    print(f"torchao source (backend/artifacts/text-to-bullets-int8): {source_size:.1f} MB")
    print(f"ONNX INT8 output (backend-onnx/artifacts/...):           {onnx_size:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
