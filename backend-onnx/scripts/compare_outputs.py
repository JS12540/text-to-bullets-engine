"""Compare torch_output.json vs onnx_output.json. No heavy deps — runs in either env.

    uv run python compare_outputs.py
"""
import json
from pathlib import Path

ARTIFACTS = Path(__file__).parent.parent / "artifacts"


def token_overlap_ratio(a: str, b: str) -> float:
    a_tokens, b_tokens = a.split(), b.split()
    if not a_tokens and not b_tokens:
        return 1.0
    common = sum(1 for x, y in zip(a_tokens, b_tokens) if x == y)
    return common / max(len(a_tokens), len(b_tokens))


def main():
    torch_file = ARTIFACTS / "torch_output.json"
    onnx_file = ARTIFACTS / "onnx_output.json"

    for f in (torch_file, onnx_file):
        if not f.exists():
            raise RuntimeError(f"{f} not found. Run run_torch_baseline.py and run_onnx.py first.")

    torch_data = json.loads(torch_file.read_text())
    onnx_data = json.loads(onnx_file.read_text())
    torch_output = torch_data["output"]
    onnx_output = onnx_data["output"]

    print("=" * 60)
    print("TORCH OUTPUT:")
    print(torch_output)
    print()
    print("ONNX OUTPUT:")
    print(onnx_output)
    print()
    exact_match = torch_output == onnx_output
    print(f"Exact match: {exact_match}")
    if not exact_match:
        print(f"Token overlap ratio: {token_overlap_ratio(torch_output, onnx_output):.2%}")
    print()
    print(f"{'':20s} {'torch (torchao)':>18s} {'onnx (ORT)':>18s}")
    for label, key in (
        ("Load time", "load_time_s"),
        ("Cold generate", "cold_generate_s"),
        ("Warm generate", "warm_generate_s"),
        ("Output tokens", "output_tokens"),
    ):
        t_val = torch_data.get(key, "—")
        o_val = onnx_data.get(key, "—")
        print(f"{label:20s} {str(t_val):>18s} {str(o_val):>18s}")
    print("=" * 60)


if __name__ == "__main__":
    main()
