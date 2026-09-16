#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""best.pt 兼容性转换 —— 旧版 PyTorch 序列化格式 → 当前版本可加载格式。

用法:
    python3 scripts/convert_model.py [best.pt] [输出目录]

行为:
  1. 用当前 ultralytics 加载旧权重（自动迁移架构/权重）。
  2. 以当前格式重新 save()（fire_smoke.pt，torch 2.x 新序列化）。
  3. 导出 ONNX（fire_smoke.onnx），跨框架/跨版本，可直接供 Jetson TensorRT 用。
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
    print(f"[convert] 加载: {SRC}")
    model = YOLO(SRC)

    # 打印模型信息，确认类别正确
    names = model.names
    print(f"[convert] 类别数: {len(names)} -> {names}")
    print(f"[convert] 任务: {model.task}")

    # 1) 以当前 ultralytics/torch 格式重新保存
    pt_out = os.path.join(OUT_DIR, "fire_smoke.pt")
    model.save(pt_out)
    print(f"[convert] ✅ 已保存当前格式 .pt: {pt_out}")

    # 2) 导出 ONNX（跨版本，供 Jetson TensorRT）
    onnx_out = os.path.join(OUT_DIR, "fire_smoke.onnx")
    try:
        model.export(format="onnx", imgsz=640, opset=17, simplify=False, half=False)
        # ultralytics 默认导出到 SRC 同目录同名 .onnx，手动改名
        default_onnx = os.path.splitext(SRC)[0] + ".onnx"
        if os.path.exists(default_onnx) and os.path.abspath(default_onnx) != os.path.abspath(onnx_out):
            os.replace(default_onnx, onnx_out)
        print(f"[convert] ✅ 已导出 ONNX: {onnx_out}")
    except Exception as e:  # noqa: BLE001
        print(f"[convert] ⚠ ONNX 导出失败（可忽略，.pt 已就绪）: {e}")

    print("[convert] 完成。")


if __name__ == "__main__":
    main()
