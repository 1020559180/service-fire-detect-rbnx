#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""新模型可运行性测试 —— 加载 best.pt / fire_smoke.onnx，跑一次真实前向推理。

不依赖 RoboNIX 框架，只验证模型本身能加载并推理。合成一张「火焰样」测试图
（橙红渐变 + 亮斑），确认前向传播出张量、返回框结构正常。
"""
from __future__ import annotations

import os
import sys

import numpy as np
from ultralytics import YOLO

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def make_test_image(w: int = 640, h: int = 640) -> np.ndarray:
    """合成一张橙红渐变 + 中心亮斑的图，模拟火焰画面。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 橙色→红色水平渐变
    r = np.linspace(40, 255, w, dtype=np.uint8)
    g = np.linspace(20, 120, w, dtype=np.uint8)
    img[:, :, 2] = r[None, :]
    img[:, :, 1] = g[None, :]
    img[:, :, 0] = 30
    # 中心亮黄斑（模拟火核）
    import cv2  # noqa: PLC0415
    cv2.circle(img, (w // 2, h // 2), 130, (0, 200, 255), -1)
    return img


def run(model_path: str, img: np.ndarray) -> None:
    print(f"\n── {os.path.basename(model_path)} ──")
    m = YOLO(model_path)
    print(f"  names = {m.names}")
    r = m.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
    n = len(r.boxes)
    print(f"  前向推理成功：输出 {r.boxes.xyxy.shape if n else '(无框)'}，检测到 {n} 个目标")
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
            print(f"⚠ 缺文件: {p}")
    print("\n✅ 测试完成")


if __name__ == "__main__":
    sys.exit(main())
