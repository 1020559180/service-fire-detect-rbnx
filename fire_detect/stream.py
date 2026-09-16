#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""fire_detect real-time visualization -- MJPEG annotated-stream service with detection boxes / results.

Two ways to use it:
  1) Standalone: RC_PRO_IP=<ip> python3 -m fire_detect.stream [--port 8081] [--conf 0.25]
  2) Start with the primitive: in driver.py's on_init, construct AnnotatedStreamer(backend) and start(),
     so when rbnx boot starts fire_detect the web visualization comes up automatically (drawing boxes as it detects).

Web UI:  http://localhost:8081/
Stream:  http://<host IP>:8081/stream   fire-detection annotated stream (MJPEG)
Fire:    http://localhost:8081/state   latest fire-detection result (JSON)
RC:      http://localhost:8081/rc      full RC / drone status (JSON, including the auto-discovered APK address)
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .backend import FireDetectBackend

_FIRE_COLOR = (0, 0, 255)     # BGR red -- fire
_SMOKE_COLOR = (0, 165, 255)  # BGR orange -- smoke

_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drone real-time feedback</title>
<style>
  body{background:#111;color:#eee;font-family:Consolas,Menlo,monospace;margin:0;padding:16px}
  h1{font-size:16px;font-weight:normal;color:#ffc;margin:0 0 4px}
  #conn{font-size:12px;color:#8f8;margin-bottom:10px}
  .cols{display:flex;gap:16px;flex-wrap:wrap}
  .col{flex:1;min-width:320px}
  h2{font-size:13px;color:#9cf;margin:12px 0 6px;border-bottom:1px solid #333;padding-bottom:2px}
  pre{background:#000;border:1px solid #333;padding:8px;font-size:11px;max-height:320px;overflow:auto;color:#ada}
  #vid{width:100%;border:2px solid #333;background:#000}
  #st{margin-top:6px;font-size:14px;color:#ccc;min-height:20px}
</style>
</head>
<body>
  <h1>🛸 Drone real-time feedback</h1>
  <div id="conn">Connecting…</div>

  <div class="cols">
    <div class="col">
      <h2>RC / Drone status</h2>
      <pre id="rc">—</pre>
    </div>
    <div class="col">
      <h2>Fire & smoke detection</h2>
      <img id="vid" src="/stream" alt="detection stream">
      <div id="st">Waiting for detection…</div>
    </div>
  </div>
<script>
setInterval(async()=>{
  // RC / drone status (including the auto-discovered APK address)
  try{
    const rc=await(await fetch('/rc')).json();
    document.getElementById('rc').textContent = JSON.stringify(rc,null,2);
    const src=rc._source?(' @ '+rc._source):'';
    document.getElementById('conn').textContent = rc.success
      ? ('RC Pro connected'+src+(rc.latitude?('   gps=('+rc.latitude+','+rc.longitude+')'):''))
      : ('RC Pro not connected'+src);
    document.getElementById('conn').style.color = rc.success ? '#8f8' : '#f88';
  }catch(e){}
  // fire-detection status
  try{
    const s=await(await fetch('/state')).json();
    const el=document.getElementById('st');
    if(s.alert){
      const d=s.detections.map(x=>`${x.class}@${x.confidence}`).join(', ');
      const g=s.gps?` gps=(${s.gps.latitude},${s.gps.longitude})`:'';
      el.textContent=`[frame ${s.frame}] ALERT ${s.count} -> ${d}${g}`;
      el.style.color='#f55';
    }else{
      el.textContent=`[frame ${s.frame}] normal, no fire/smoke`;
      el.style.color='#5f5';
    }
  }catch(e){}
},1000);
</script>
</body>
</html>
"""


class AnnotatedStreamer:
    """Continuously pull stream -> infer -> draw boxes -> encode; a built-in HTTP service serves the MJPEG annotated stream + JSON status.

    Reuses the externally-provided FireDetectBackend (shares the same backend and model with the primitive),
    without pulling the stream separately. Call start()/stop() to manage the background thread and HTTP service lifecycle.
    """

    def __init__(self, backend: FireDetectBackend, conf: float = 0.25, port: int = 8081):
        self.backend = backend
        self.conf = conf
        self.port = port
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._state = {"alert": False, "detections": [], "count": 0,
                       "timestamp": 0, "gps": None, "frame": 0}
        self._running = False
        self._loop_thread = None
        self._http_thread = None
        self._http_server = None

    # -- lifecycle --
    def start(self) -> bool:
        """Start the annotation loop thread + HTTP service. Returns whether the web server bound its port successfully."""
        if self._running:
            return self._http_server is not None
        self._running = True
        self._loop_thread = threading.Thread(target=self._loop, daemon=True, name="fire-stream-loop")
        self._loop_thread.start()
        return self._start_http()

    def stop(self) -> None:
        """Stop the annotation loop and shut down the HTTP service."""
        self._running = False
        if self._http_server is not None:
            try:
                self._http_server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._http_server.server_close()
            self._http_server = None

    def is_running(self) -> bool:
        return self._running

    def _start_http(self) -> bool:
        _Handler.streamer = self
        try:
            self._http_server = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
        except OSError as e:
            print(f"[fire_stream] ⚠ port {self.port} is in use, web visualization not started: {e}", flush=True)
            return False
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever, daemon=True, name="fire-stream-http"
        )
        self._http_thread.start()
        print(f"[fire_stream] ✅ web visualization: http://localhost:{self.port}/", flush=True)
        return True

    # -- read by the HTTP handlers --
    def latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def latest_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def rc_state(self) -> dict:
        """Full RC / drone status (/api/status + GPS), served by the /rc endpoint."""
        return self.backend.rc_state()

    # -- background loop --
    def _loop(self) -> None:
        model = self.backend._load_model()
        if model is None:
            with self._lock:
                self._state["message"] = self.backend._last_error or "model load failed"
            return

        n = 0
        while self._running:
            n += 1
            frame = self.backend._grab_frame()
            if frame is None:
                time.sleep(0.3)
                continue
            img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                time.sleep(0.3)
                continue

            dets = self.backend._infer(frame, model, self.conf)
            fires = [d for d in dets if d["kind"] in ("fire", "smoke")]
            alert = bool(fires)
            gps = self.backend._gps() if alert else None

            self._draw(img, fires, alert, n)

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    self._state = {
                        "alert": alert, "detections": fires, "count": len(fires),
                        "timestamp": int(time.time()), "gps": gps, "frame": n,
                    }
            time.sleep(0.03)

    def _draw(self, img: np.ndarray, fires: list, alert: bool, n: int) -> None:
        h, w = img.shape[:2]
        if alert:
            cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 200), -1)
            text = f"ALERT  fire/smoke={len(fires)}  frame {n}  {time.strftime('%H:%M:%S')}"
            color = (255, 255, 255)
        else:
            cv2.rectangle(img, (0, 0), (w, 34), (35, 35, 35), -1)
            text = f"NORMAL  frame {n}  {time.strftime('%H:%M:%S')}"
            color = (200, 200, 200)
        cv2.putText(img, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        for d in fires:
            color = _FIRE_COLOR if d["kind"] == "fire" else _SMOKE_COLOR
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{d['class']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            top = max(y1 - th - 8, 36)
            cv2.rectangle(img, (x1, top), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


class _Handler(BaseHTTPRequestHandler):
    streamer = None

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._html()
        elif self.path == "/stream":
            self._stream()
        elif self.path == "/state":
            self._json()
        elif self.path == "/rc":
            self._json(self.streamer.rc_state())
        else:
            self.send_error(404)

    def _html(self) -> None:
        body = _HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj=None) -> None:
        data = obj if obj is not None else self.streamer.latest_state()
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while self.streamer.is_running():
                jpeg = self.streamer.latest_jpeg()
                if jpeg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="fire_detect real-time annotated stream service")
    ap.add_argument("--rc-host", default=os.environ.get("RC_PRO_IP", "172.20.10.3"))
    ap.add_argument("--rc-port", type=int, default=int(os.environ.get("RC_PRO_PORT", "8080")))
    ap.add_argument("--model", default=os.environ.get("FIRE_MODEL_PATH", ""))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--port", type=int, default=int(os.environ.get("FIRE_STREAM_PORT", "8081")))
    args = ap.parse_args()

    backend = FireDetectBackend(args.rc_host, args.rc_port, model_path=args.model, min_confidence=args.conf)
    streamer = AnnotatedStreamer(backend, conf=args.conf, port=args.port)
    streamer.start()
    print(f"[fire_stream] pulling stream http://{args.rc_host}:{args.rc_port}/api/video  conf={args.conf}")
    print("[fire_stream] Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
