#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""FireDetectBackend -- fire & smoke detection backend (stream pull + YOLO inference + GPS attach).

SDK-agnostic: depends only on two HTTP endpoints of the DJI bridge APK:

    GET  /api/video         MJPEG video stream (multipart/x-mixed-replace; boundary=FRAME)
    POST /api/capture_gps   current GPS coordinates (attached on a fire & smoke hit)

Data flow:

    RoboNIX (driver.py, MCP)
        │ semantic methods
        ▼
    FireDetectBackend (this module)
        │ grab one MJPEG frame -> cv2 decode -> YOLO inference
        │ on a fire & smoke hit -> fetch coordinates via /api/capture_gps
        ▼
    returns dict {success, detections, alert, gps, ...}

Swapping camera / SDK only requires changing this backend (e.g. the video stream and coordinate source for PSDK/OSDK);
driver.py and the contract layer stay unchanged.
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

# flame / smoke class keywords (a hit on any of these is treated as an alert)
_FIRE_KEYS = ("fire", "flame")
_SMOKE_KEYS = ("smoke",)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))   # fire_detect/fire_detect
_ROOT_DIR = os.path.dirname(_PKG_DIR)                    # fire_detect (package root, contains models/)


class FireDetectBackend:
    """Fire & smoke detection backend: pulls the stream from the DJI bridge APK and infers, attaching GPS on a hit."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 model_path: str = "", min_confidence: float = DEFAULT_MIN_CONF):
        # when host is "auto"/"" auto-discover the APK IP (fall back to the default address if not found)
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
        # -- real-time monitor state --
        self._monitor_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._latest: Dict[str, Any] = {
            "success": False, "alert": False, "detections": [], "count": 0,
            "timestamp": 0, "gps": None, "running": False,
        }
        self._latest_lock = threading.Lock()

    # -- connection probe --
    def ping(self) -> Dict[str, Any]:
        try:
            r = requests.get(f"http://{self.host}:{self.port}/api/status", timeout=self.timeout)
            return {"success": r.status_code == 200}
        except Exception as e:  # noqa: BLE001
            return {"success": False, "message": f"RC Pro unreachable ({e})"}

    def rc_state(self) -> Dict[str, Any]:
        """Full RC / drone status: merged /api/status + /api/capture_gps, with the source address attached."""
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
            return {"success": False, "message": f"RC Pro unreachable ({e})",
                    "_source": f"{self.host}:{self.port}"}

    # -- semantic methods --
    def detect(self, min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """Grab one frame and run fire & smoke detection. Attach the current GPS on a hit."""
        conf = float(min_confidence) if (min_confidence is not None and min_confidence > 0) else self.min_confidence

        frame = self._grab_frame()
        if frame is None:
            return {"success": False, "message": self._last_error or "failed to grab a video frame"}

        model = self._load_model()
        if model is None:
            return {"success": False, "message": self._last_error or "model not loaded"}

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
        # on a fire & smoke hit -> fetch live GPS coordinates, for alert primitives / upper layers
        if alert:
            result["gps"] = self._gps()
        return result

    # -- stream pull: read one frame of JPEG bytes from MJPEG --
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
            self._last_error = "no complete JPEG frame found in the video stream"
        except Exception as e:  # noqa: BLE001
            self._last_error = f"failed to pull the video stream: {e}"
        return None

    # -- model loading (lazy + cached) --
    def _load_model(self):
        if self._model_loaded:
            return self._model
        self._model_loaded = True
        try:
            from ultralytics import YOLO
        except ImportError:
            self._last_error = "missing ultralytics dependency (pip install ultralytics)"
            return None

        weights = self._resolve_model()
        try:
            self._model = YOLO(weights) if weights else YOLO("yolov8n.pt")
            return self._model
        except Exception as e:  # noqa: BLE001
            self._last_error = f"model load failed: {e}"
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

    # -- inference --
    def _infer(self, frame: bytes, model, conf: float) -> List[Dict[str, Any]]:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self._last_error = "missing opencv-python dependency"
            return []

        arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            self._last_error = "JPEG decode failed"
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

    # -- real-time monitor (background thread keeps grabbing frames and inferring; poll for the latest result) --
    def start_monitor(self, min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """Start the background detection thread, which keeps grabbing frames and inferring; the latest result is read by polling monitor_state()."""
        with self._latest_lock:
            if self._monitor_running:
                return {"success": True, "running": True, "message": "monitor already running"}
            self._monitor_running = True
        conf = float(min_confidence) if (min_confidence is not None and min_confidence > 0) else self.min_confidence

        # preload the model (return immediately on failure, don't start the thread)
        model = self._load_model()
        if model is None:
            with self._latest_lock:
                self._monitor_running = False
            return {"success": False, "running": False, "message": self._last_error or "model not loaded"}

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(model, conf), daemon=True, name="fire-detect-monitor"
        )
        self._monitor_thread.start()
        return {"success": True, "running": True}

    def stop_monitor(self) -> Dict[str, Any]:
        """Stop the background detection thread."""
        with self._latest_lock:
            self._monitor_running = False
            self._latest["running"] = False
        # don't join: the thread exits at the start of its next loop once it sees running=False; joining would block the MCP handler.
        return {"success": True, "running": False}

    def monitor_state(self) -> Dict[str, Any]:
        """Return the latest detection result (non-blocking, for polling by an external loop)."""
        with self._latest_lock:
            return dict(self._latest)

    def _monitor_loop(self, model, conf: float) -> None:
        """Background loop: grab a frame -> infer -> update _latest."""
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
                        "message": self._last_error or "failed to grab a video frame",
                    })
                time.sleep(0.5)  # back off on stream-pull failure to avoid hammering HTTP while spinning
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
