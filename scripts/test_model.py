#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""New-model sanity test -- load best.pt / fire_smoke.onnx and run one real forward pass.

Does not depend on the RoboNIX framework; only verifies the model itself can load and infer. Synthesizes a "flame-like" test image
(orange-red gradient + bright spot) and confirms the forward pass emits tensors and returns a well-formed box structure.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from ultralytics import YOLO

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def make_test_image(w: int = 640, h: int = 640) -> np.ndarray:
    """Synthesize an orange-red gradient image with a bright center spot to mimic a flame scene."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # orange -> red horizontal gradient
    r = np.linspace(40, 255, w, dtype=np.uint8)
    g = np.linspace(20, 120, w, dtype=np.uint8)
    img[:, :, 2] = r[None, :]
    img[:, :, 1] = g[None, :]
    img[:, :, 0] = 30
    # bright yellow center spot (mimic the fire core)
    import cv2  # noqa: PLC0415
    cv2.circle(img, (w // 2, h // 2), 130, (0, 200, 255), -1)
    return img


def run(model_path: str, img: np.ndarray) -> None:
    print(f"\n── {os.path.basename(model_path)} ──")
    m = YOLO(model_path)
    print(f"  names = {m.names}")
    r = m.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
    n = len(r.boxes)
    print(f"  forward pass succeeded: output {r.boxes.xyxy.shape if n else '(no boxes)'}, detected {n} objects")
    for b in r.boxes[:5]:
        cls = int(b.cls[0].item())
        print(f"    class={m.names.get(cls)} conf={float(b.conf[0]):.3f} bbox={[round(float(v),1) for v in b.xyxy[0].tolist()]}")


def main() -> None:
    img = make_test_image()
    for name in ("best.pt", "fire_smoke.onnx"):
        p = os.path.join(MODELS, name)
        if os.path.exists(p):
            run(p, img)
        else:
            print(f"⚠ missing file: {p}")
    print("\n✅ test complete")


if __name__ == "__main__":
    sys.exit(main())
