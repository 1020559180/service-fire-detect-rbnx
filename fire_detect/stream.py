#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""fire_detect 实时可视化 —— 带检测框/结果的 MJPEG 标注流服务。

两种用法：
  1) 独立运行： RC_PRO_IP=<ip> python3 -m fire_detect.stream [--port 8081] [--conf 0.25]
  2) 随原语启动： driver.py 的 on_init 里 new AnnotatedStreamer(backend) 并 start()，
     这样 rbnx boot 启动 fire_detect 时，web 可视化自动一并起来（边检测边画框）。

Web 端:  http://localhost:8081/
流:      http://<本机IP>:8081/stream   火检标注流 (MJPEG)
火检:    http://localhost:8081/state   最新火检结果 (JSON)
遥控器:  http://localhost:8081/rc      遥控器/无人机完整状态 (JSON，含自动发现的 APK 地址)
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

_FIRE_COLOR = (0, 0, 255)     # BGR 红 —— fire
_SMOKE_COLOR = (0, 165, 255)  # BGR 橙 —— smoke

_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>无人机实时反馈</title>
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
  <h1>🛸 无人机实时反馈</h1>
  <div id="conn">连接中…</div>

  <div class="cols">
    <div class="col">
      <h2>遥控器 / 无人机状态</h2>
      <pre id="rc">—</pre>
    </div>
    <div class="col">
      <h2>烟火检测</h2>
      <img id="vid" src="/stream" alt="detection stream">
      <div id="st">等待检测…</div>
    </div>
  </div>
<script>
setInterval(async()=>{
  // 遥控器 / 无人机状态（含自动发现的 APK 地址）
  try{
    const rc=await(await fetch('/rc')).json();
    document.getElementById('rc').textContent = JSON.stringify(rc,null,2);
    const src=rc._source?(' @ '+rc._source):'';
    document.getElementById('conn').textContent = rc.success
      ? ('RC Pro 已连接'+src+(rc.latitude?('   gps=('+rc.latitude+','+rc.longitude+')'):''))
      : ('RC Pro 未连接'+src);
    document.getElementById('conn').style.color = rc.success ? '#8f8' : '#f88';
  }catch(e){}
  // 火检状态
  try{
    const s=await(await fetch('/state')).json();
    const el=document.getElementById('st');
    if(s.alert){
      const d=s.detections.map(x=>`${x.class}@${x.confidence}`).join(', ');
      const g=s.gps?` gps=(${s.gps.latitude},${s.gps.longitude})`:'';
      el.textContent=`[frame ${s.frame}] ALERT ${s.count} -> ${d}${g}`;
      el.style.color='#f55';
    }else{
      el.textContent=`[frame ${s.frame}] 正常，无火烟`;
      el.style.color='#5f5';
    }
  }catch(e){}
},1000);
</script>
</body>
</html>
"""


class AnnotatedStreamer:
    """持续拉流→推理→画框→编码；内置 HTTP 服务输出 MJPEG 标注流 + JSON 状态。

    复用外部传入的 FireDetectBackend（与原语共享同一个 backend 与模型），
    不单独拉流。调用 start()/stop() 管理后台线程与 HTTP 服务生命周期。
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

    # ── 生命周期 ───────────────────────────────────────────
    def start(self) -> bool:
        """启动标注循环线程 + HTTP 服务。返回 web 是否成功绑定端口。"""
        if self._running:
            return self._http_server is not None
        self._running = True
        self._loop_thread = threading.Thread(target=self._loop, daemon=True, name="fire-stream-loop")
        self._loop_thread.start()
        return self._start_http()

    def stop(self) -> None:
        """停止标注循环，关闭 HTTP 服务。"""
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
            print(f"[fire_stream] ⚠ 端口 {self.port} 被占用，web 可视化未启动: {e}", flush=True)
            return False
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever, daemon=True, name="fire-stream-http"
        )
        self._http_thread.start()
        print(f"[fire_stream] ✅ web 可视化: http://localhost:{self.port}/", flush=True)
        return True

    # ── 供 HTTP handler 读取 ───────────────────────────────
    def latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def latest_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def rc_state(self) -> dict:
        """遥控器/无人机完整状态（/api/status + GPS），供 /rc 端点。"""
        return self.backend.rc_state()

    # ── 后台循环 ───────────────────────────────────────────
    def _loop(self) -> None:
        model = self.backend._load_model()
        if model is None:
            with self._lock:
                self._state["message"] = self.backend._last_error or "模型加载失败"
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
    ap = argparse.ArgumentParser(description="fire_detect 实时标注流服务")
    ap.add_argument("--rc-host", default=os.environ.get("RC_PRO_IP", "172.20.10.3"))
    ap.add_argument("--rc-port", type=int, default=int(os.environ.get("RC_PRO_PORT", "8080")))
    ap.add_argument("--model", default=os.environ.get("FIRE_MODEL_PATH", ""))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--port", type=int, default=int(os.environ.get("FIRE_STREAM_PORT", "8081")))
    args = ap.parse_args()

    backend = FireDetectBackend(args.rc_host, args.rc_port, model_path=args.model, min_confidence=args.conf)
    streamer = AnnotatedStreamer(backend, conf=args.conf, port=args.port)
    streamer.start()
    print(f"[fire_stream] 拉流 http://{args.rc_host}:{args.rc_port}/api/video  conf={args.conf}")
    print("[fire_stream] Ctrl+C 停止")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
