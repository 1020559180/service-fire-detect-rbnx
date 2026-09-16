#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""FireDetectBackend —— 烟火检测后端（拉流 + YOLO 推理 + GPS 附带）。

SDK 无关：只依赖 DJI 桥接 APK 的两个 HTTP 端点：

    GET  /api/video         MJPEG 视频流（multipart/x-mixed-replace; boundary=FRAME）
    POST /api/capture_gps   当前 GPS 坐标（命中烟火时附带）

数据流：

    RoboNIX (driver.py, MCP)
        │ 语义方法
        ▼
    FireDetectBackend (本模块)
        │ 拉一帧 MJPEG → cv2 解码 → YOLO 推理
        │ 命中烟火 → /api/capture_gps 取坐标
        ▼
    返回 dict {success, detections, alert, gps, ...}

换摄像头/换 SDK 只需改本后端（例如 PSDK/OSDK 的视频流与坐标来源），
driver.py 与契约层不变。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests

DEFAULT_HOST = "172.20.10.3"
DEFAULT_PORT = 8080
DEFAULT_MIN_CONF = 0.4

# 火焰/烟雾类别关键词（命中即视为告警）
_FIRE_KEYS = ("fire", "flame", "火焰")
_SMOKE_KEYS = ("smoke", "烟雾", "烟")

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))   # fire_detect/fire_detect
_ROOT_DIR = os.path.dirname(_PKG_DIR)                    # fire_detect（包根目录，含 models/）


class FireDetectBackend:
    """烟火检测后端：从 DJI 桥接 APK 拉流推理，命中时附带 GPS。"""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 model_path: str = "", min_confidence: float = DEFAULT_MIN_CONF):
        # host 为 "auto"/"" 时自动发现 APK IP（找不到则回退默认地址）
        if host in ("auto", "", None):
            from .discover import discover_apk
            host = discover_apk(port=port) or DEFAULT_HOST
        self.host = host
        self.port = port
        self.video_url = f"http://{host}:{port}/api/video"
        self.gps_url = f"http://{host}:{port}/api/capture_gps"
        self.model_path = model_path or ""
        self.min_confidence = float(min_confidence)
        self.timeout = 5.0
        self._model = None
        self._model_loaded = False
        self._last_error: Optional[str] = None
        # ── 实时监控状态 ──
        self._monitor_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._latest: Dict[str, Any] = {
            "success": False, "alert": False, "detections": [], "count": 0,
            "timestamp": 0, "gps": None, "running": False,
        }
        self._latest_lock = threading.Lock()

    # ── 连接探测 ───────────────────────────────────────────
    def ping(self) -> Dict[str, Any]:
        try:
            r = requests.get(f"http://{self.host}:{self.port}/api/status", timeout=self.timeout)
            return {"success": r.status_code == 200}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "message": f"RC Pro 不可达 ({e})"}

    def rc_state(self) -> Dict[str, Any]:
        """遥控器/无人机完整状态：/api/status + /api/capture_gps 合并，附带来源地址。"""
        try:
            r = requests.get(f"http://{self.host}:{self.port}/api/status", timeout=self.timeout)
            if r.status_code != 200:
                return {"success": False, "message": f"status HTTP {r.status_code}",
                        "_source": f"{self.host}:{self.port}"}
            try:
                data = r.json()
            except ValueError:
                data = {"raw": r.text[:300]}
            if not isinstance(data, dict):
                data = {"raw": data}
            data["success"] = True
            data["_source"] = f"{self.host}:{self.port}"
            try:
                g = requests.post(f"http://{self.host}:{self.port}/api/capture_gps",
                                  json={}, timeout=self.timeout)
                if g.status_code == 200:
                    gd = g.json()
                    if isinstance(gd, dict) and "latitude" in gd:
                        data["latitude"] = gd.get("latitude")
                        data["longitude"] = gd.get("longitude")
            except Exception:  # noqa: BLE001
                pass
            return data
        except Exception as e:  # noqa: BLE001
            return {"success": False, "message": f"RC Pro 不可达 ({e})",
                    "_source": f"{self.host}:{self.port}"}

    # ── 语义方法 ───────────────────────────────────────────
    def detect(self, min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """拉取一帧画面做烟火检测。命中烟火时附带当前 GPS。"""
        conf = float(min_confidence) if (min_confidence is not None and min_confidence > 0) else self.min_confidence

        frame = self._grab_frame()
        if frame is None:
            return {"success": False, "message": self._last_error or "拉取视频帧失败"}

        model = self._load_model()
        if model is None:
            return {"success": False, "message": self._last_error or "模型未加载"}

        detections = self._infer(frame, model, conf)
        alert = any(d["kind"] in ("fire", "smoke") for d in detections)

        result: Dict[str, Any] = {
            "success": True,
            "alert": alert,
            "detections": detections,
            "count": len(detections),
            "timestamp": int(time.time()),
            "gps": None,
        }
        # 命中烟火 → 实时取 GPS 坐标，供告警原语/上层使用
        if alert:
            result["gps"] = self._gps()
        return result

    # ── 拉流：从 MJPEG 读一帧 JPEG 字节 ────────────────────
    def _grab_frame(self, max_bytes: int = 5 * 1024 * 1024) -> Optional[bytes]:
        try:
            r = requests.get(self.video_url, stream=True, timeout=(3.0, 8.0))
            buf = b""
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    break
                buf += chunk
                soi = buf.find(b"\xff\xd8")
                if soi != -1:
                    eoi = buf.find(b"\xff\xd9", soi + 2)
                    if eoi != -1:
                        return buf[soi:eoi + 2]
                if len(buf) > max_bytes:
                    break
            self._last_error = "视频流中未找到完整 JPEG 帧"
        except Exception as e:  # noqa: BLE001
            self._last_error = f"拉取视频流失败: {e}"
        return None

    # ── 模型加载（懒加载 + 缓存） ──────────────────────────
    def _load_model(self):
        if self._model_loaded:
            return self._model
        self._model_loaded = True
        try:
            from ultralytics import YOLO
        except ImportError:
            self._last_error = "缺少 ultralytics 依赖（pip install ultralytics）"
            return None

        weights = self._resolve_model()
        try:
            self._model = YOLO(weights) if weights else YOLO("yolov8n.pt")
            return self._model
        except Exception as e:  # noqa: BLE001
            self._last_error = f"模型加载失败: {e}"
            self._model = None
            return None

    def _resolve_model(self) -> Optional[str]:
        candidates: List[str] = []
        if self.model_path:
            candidates.append(self.model_path)
        env = os.environ.get("FIRE_MODEL_PATH", "")
        if env:
            candidates.append(env)
        models_dir = os.path.join(_ROOT_DIR, "models")
        if os.path.isdir(models_dir):
            for f in sorted(os.listdir(models_dir)):
                if f.endswith((".pt", ".engine", ".onnx")):
                    candidates.append(os.path.join(models_dir, f))
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    # ── 推理 ──────────────────────────────────────────────
    def _infer(self, frame: bytes, model, conf: float) -> List[Dict[str, Any]]:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self._last_error = "缺少 opencv-python 依赖"
            return []

        arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            self._last_error = "JPEG 解码失败"
            return []

        results = model(img, conf=conf, verbose=False)
        detections: List[Dict[str, Any]] = []
        for r in results:
            names = r.names  # {cls_idx: name}
            for box in r.boxes:
                cls = int(box.cls[0].item())
                name = str(names.get(cls, cls)).lower()
                detections.append({
                    "kind": self._classify(name),
                    "class": name,
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                })
        return detections

    @staticmethod
    def _classify(name: str) -> str:
        if any(k in name for k in _FIRE_KEYS):
            return "fire"
        if any(k in name for k in _SMOKE_KEYS):
            return "smoke"
        return "other"

    # ── GPS ───────────────────────────────────────────────
    def _gps(self) -> Optional[Dict[str, Any]]:
        try:
            r = requests.post(self.gps_url, json={}, timeout=self.timeout)
            if r.status_code != 200:
                return None
            data = r.json()
            if data.get("success") is not False and "latitude" in data:
                return {
                    "latitude": data.get("latitude"),
                    "longitude": data.get("longitude"),
                }
        except Exception:  # noqa: BLE001
            return None
        return None

    # ── 实时监控（后台线程持续拉帧推理，轮询取最新结果） ──
    def start_monitor(self, min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """启动后台检测线程，持续拉帧推理，最新结果由 monitor_state() 轮询读取。"""
        with self._latest_lock:
            if self._monitor_running:
                return {"success": True, "running": True, "message": "监控已在运行"}
            self._monitor_running = True
        conf = float(min_confidence) if (min_confidence is not None and min_confidence > 0) else self.min_confidence

        # 预加载模型（失败则立即返回，不起线程）
        model = self._load_model()
        if model is None:
            with self._latest_lock:
                self._monitor_running = False
            return {"success": False, "running": False, "message": self._last_error or "模型未加载"}

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(model, conf), daemon=True, name="fire-detect-monitor"
        )
        self._monitor_thread.start()
        return {"success": True, "running": True}

    def stop_monitor(self) -> Dict[str, Any]:
        """停止后台检测线程。"""
        with self._latest_lock:
            self._monitor_running = False
            self._latest["running"] = False
        # 不 join：线程会在下一个循环起点发现 running=False 退出；join 会阻塞 MCP handler。
        return {"success": True, "running": False}

    def monitor_state(self) -> Dict[str, Any]:
        """返回最新一次检测结果（非阻塞，供外部循环轮询）。"""
        with self._latest_lock:
            return dict(self._latest)

    def _monitor_loop(self, model, conf: float) -> None:
        """后台循环：拉帧 → 推理 → 更新 _latest。"""
        while True:
            with self._latest_lock:
                if not self._monitor_running:
                    break
            frame = self._grab_frame()
            if frame is None:
                with self._latest_lock:
                    self._latest.update({
                        "success": False, "alert": False, "detections": [],
                        "count": 0, "timestamp": int(time.time()),
                        "gps": None, "running": True,
                        "message": self._last_error or "拉取视频帧失败",
                    })
                time.sleep(0.5)  # 拉流失败退避，避免空转打爆 HTTP
                continue
            detections = self._infer(frame, model, conf)
            alert = any(d["kind"] in ("fire", "smoke") for d in detections)
            gps = self._gps() if alert else None
            with self._latest_lock:
                self._latest.update({
                    "success": True, "alert": alert, "detections": detections,
                    "count": len(detections), "timestamp": int(time.time()),
                    "gps": gps, "running": True,
                })


if __name__ == "__main__":
    import json
    b = FireDetectBackend()
    print(json.dumps(b.ping(), ensure_ascii=False))
    print(json.dumps(b.detect(), ensure_ascii=False))
