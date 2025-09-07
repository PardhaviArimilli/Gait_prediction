import argparse
import os
import sys
import torch

# Ensure project root is on sys.path for `src` imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.registry import build_model


def export(ckpt: str, model_name: str, out_path: str, time_len: int = 500, num_classes: int = 3):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = build_model(model_name, num_classes=num_classes)
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    # Dummy input: (B=1, T, C=3)
    dummy = torch.zeros(1, time_len, 3, dtype=torch.float32)

    # Export with dynamic axes for batch and time
    dynamic_axes = {
        "input": {0: "batch", 1: "time"},
        "logits": {0: "batch", 1: "time"}
    }

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX for web inference")
    p.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    p.add_argument("--model", default="cnn_bilstm", help="Model name: cnn_bilstm or tcn")
    p.add_argument("--out", default="giwebsite/netlify/assets/model.onnx", help="Output ONNX path")
    p.add_argument("--time_len", type=int, default=500, help="Dummy sequence length for export")
    args = p.parse_args()
    export(args.ckpt, args.model, args.out, time_len=args.time_len)


if __name__ == "__main__":
    main()


