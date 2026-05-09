"""Diagnose texture_baker + CUDA.

Unbuffered prints; script sets OMP_NUM_THREADS=1 to reduce rare Windows hangs.
若控制台乱码，请用 UTF-8 终端或 chcp 65001。
"""
from __future__ import annotations

import os
import sys

# Avoid rare hangs: OpenMP inside PyTorch + extension kernels on Windows
os.environ.setdefault("OMP_NUM_THREADS", "1")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    log("1) import torch ...")
    import torch

    log(f"2) torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        log("   CUDA not available - skipping GPU test (install cu124 torch if you need GPU).")
        return 0

    log("3) import TextureBaker ...")
    from texture_baker import TextureBaker

    log(f"4) baker module: {TextureBaker.__module__!r}")
    import texture_baker.baker as baker_mod

    log(f"   baker.py path: {baker_mod.__file__}")

    log("5) build tensors on cuda:0 ...")
    b = TextureBaker()
    u = torch.zeros(4, 2, device="cuda")
    f = torch.zeros(2, 3, dtype=torch.int32, device="cuda")

    log("6) rasterize (CPU extension path copies to/from GPU; first call may be slow) ...")
    r = b.rasterize(u, f, 32)
    log("6b) rasterize returned OK")
    log(f"7) result device = {r.device} shape = {tuple(r.shape)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
