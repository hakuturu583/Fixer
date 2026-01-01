import os
import sys
import argparse
from pathlib import Path

import torch
from torch import nn

# Ensure we can import from src/ when running this script from tools/
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pix2pix_turbo_nocond_cosmos_base_faster_tokenizer import Pix2Pix_Turbo


def select_device(device_str: str | None) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export Fixer to TorchScript (trace)")
    # model
    p.add_argument("--model", type=str, required=True, help="path to pretrained Fixer checkpoint (.pkl)")
    p.add_argument("--timestep", type=int, default=250)
    p.add_argument("--vae-skip-connection", action="store_true")
    # io
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--name", type=str, default="fixer")
    # shape/dtype/device
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", type=str, default=None, help="e.g., cuda, cuda:0, cpu")
    # warmup runs
    p.add_argument("--warmup-iters", type=int, default=5)
    return p.parse_args()


def str_to_dtype(s: str) -> torch.dtype:
    if s == "fp32":
        return torch.float32
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {s}")


class _FixerForward(nn.Module):
    """Thin wrapper around Pix2Pix_Turbo.forward for tracing end-to-end."""

    def __init__(self, core: Pix2Pix_Turbo):
        super().__init__()
        self.core = core

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


def build_core(pretrained_path: str, timestep: int, vae_skip_connection: bool,
               device: torch.device, dtype: torch.dtype) -> Pix2Pix_Turbo:
    core = Pix2Pix_Turbo(
        pretrained_path=pretrained_path,
        timestep=timestep,
        vae_skip_connection=vae_skip_connection,
        batch_size=1,
    )
    core.set_eval()
    core.to(device=device, dtype=dtype)
    # Align Cosmos tokenizer's internal dtype attribute (used to cast inputs)
    try:
        # Some Cosmos components store a `dtype` attribute (not the torch.nn.Module property)
        # that they use to cast runtime tensors. Keep it in sync with parameter dtype.
        if hasattr(core.vae, "dtype"):
            setattr(core.vae, "dtype", dtype)
    except Exception:
        pass
    # Ensure sigma_data uses the same dtype/device as the VAE/UNet to avoid upcasting
    try:
        core.sigma_data = torch.as_tensor(core.sigma_data, dtype=dtype, device=device)
    except Exception:
        pass
    return core


def dummy_input(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(1, 3, h, w, device=device, dtype=dtype)


def trace_and_save(module: nn.Module, example_inputs, out_path: str, warmup_iters: int = 5):
    module.eval()
    with torch.no_grad():
        # light warmup
        for _ in range(max(0, warmup_iters)):
            if isinstance(example_inputs, (tuple, list)):
                module(*example_inputs)
            else:
                module(example_inputs)
        # trace
        traced = torch.jit.trace(module, example_inputs, strict=False)
        torch.jit.save(traced, out_path)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = select_device(args.device)
    dtype = str_to_dtype(args.dtype)

    # 1) Build core model (loads UNet, VAE, condition, etc.)
    core = build_core(args.model, args.timestep, args.vae_skip_connection, device, dtype)

    # 2) End-to-end export (Fixer) — always export
    fixer_wrap = _FixerForward(core)
    ex = dummy_input(args.height, args.width, device, dtype)
    out_path = os.path.join(args.outdir, f"{args.name}.ts")
    trace_and_save(fixer_wrap, ex, out_path, warmup_iters=args.warmup_iters)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
