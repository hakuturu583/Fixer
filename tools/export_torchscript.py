import os
import sys
import argparse
from pathlib import Path

import torch

# Make src/ importable when running from tools/
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import Pix2Pix_Turbo


def select_device(device_str: str | None) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    # Pix2Pix_Turbo internally moves modules to CUDA; prefer CUDA when available
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export Fixer (end-to-end) to TorchScript via tracing")
    # Model config (fixed timestep acceptable)
    p.add_argument("--model", type=str, required=True, help="path to pretrained checkpoint (.pkl)")
    p.add_argument("--timestep", type=int, default=400, help="fixed diffusion timestep")
    p.add_argument("--vae-skip-connection", action="store_true", help="enable VAE skip connection")
    # IO
    p.add_argument("--out", type=str, required=True, help="output .ts path")
    # Shapes / device
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--device", type=str, default=None, help="e.g., cuda, cuda:0")
    # Warmup
    p.add_argument("--warmup-iters", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = select_device(args.device)

    # Build core model with fixed settings; batch_size is 1 for tracing
    model = Pix2Pix_Turbo(
        pretrained_path=args.model,
        timestep=args.timestep,
        vae_skip_connection=args.vae_skip_connection,
        batch_size=1,
    )
    model.set_eval()
    # Stabilize tracing by forcing FP32 path inside forward
    try:
        setattr(model, "force_fp32_for_export", True)
    except Exception:
        pass

    # Dummy input (FP32) on the chosen device
    x = torch.randn(1, 3, args.height, args.width, device=device, dtype=torch.float32)

    # Light warmup and trace
    model.eval()
    with torch.no_grad():
        for _ in range(max(0, args.warmup_iters)):
            model(x)
        traced = torch.jit.trace(model, x, strict=False)
        torch.jit.save(traced, args.out)

    print(f"Saved TorchScript: {args.out}")


if __name__ == "__main__":
    main()

