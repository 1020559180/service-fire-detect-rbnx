#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""best.pt compatibility conversion -- old PyTorch serialization format -> current loadable format.

Usage:
    python3 scripts/convert_model.py [best.pt] [output_dir]

Behavior:
  1. Load the old weights with the current ultralytics (auto-migrates architecture / weights).
  2. Re-save() in the current format (fire_smoke.pt, torch 2.x new serialization).
  3. Export ONNX (fire_smoke.onnx), cross-framework / cross-version, ready for Jetson TensorRT.
"""
from __future__ import annotations

import os
import sys

from ultralytics import YOLO

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best.pt"
)
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(SRC)


def main() -> None:
    print(f"[convert] loading: {SRC}")
    model = YOLO(SRC)

    # print model info and confirm the classes are correct
    names = model.names
    print(f"[convert] classes: {len(names)} -> {names}")
    print(f"[convert] task: {model.task}")

    # 1) re-save in the current ultralytics/torch format
    pt_out = os.path.join(OUT_DIR, "fire_smoke.pt")
    model.save(pt_out)
    print(f"[convert] ✅ saved current-format .pt: {pt_out}")

    # 2) export ONNX (cross-version, for Jetson TensorRT)
    onnx_out = os.path.join(OUT_DIR, "fire_smoke.onnx")
    try:
        model.export(format="onnx", imgsz=640, opset=17, simplify=False, half=False)
        # ultralytics exports a same-named .onnx next to SRC by default; rename it manually
        default_onnx = os.path.splitext(SRC)[0] + ".onnx"
        if os.path.exists(default_onnx) and os.path.abspath(default_onnx) != os.path.abspath(onnx_out):
            os.replace(default_onnx, onnx_out)
        print(f"[convert] ✅ exported ONNX: {onnx_out}")
    except Exception as e:  # noqa: BLE001
        print(f"[convert] ⚠ ONNX export failed (safe to ignore, .pt is ready): {e}")

    print("[convert] done.")


if __name__ == "__main__":
    main()
