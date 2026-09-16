#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""fire_detect standalone REPL -- framework-free manual backend debugging entry point.

The driver layer (framework dispatch) lives in ``driver.py`` (the `@vision.mcp` handler). This file is a **standalone test
REPL** that drives ``FireDetectBackend`` directly to grab frames and infer, for validating fire & smoke detection without the framework.

Run:
    RC_PRO_IP=<ip> FIRE_MODEL_PATH=<weights> python3 -m fire_detect.main
"""
from __future__ import annotations

import json
import logging
import os

from .backend import FireDetectBackend

logging.basicConfig(level=logging.INFO, format="[fire_detect] %(message)s")
log = logging.getLogger("fire_detect.repl")

_HOST = os.environ.get("RC_PRO_IP", "172.20.10.3")
_PORT = int(os.environ.get("RC_PRO_PORT", "8080"))
_MODEL = os.environ.get("FIRE_MODEL_PATH", "")


def main() -> None:
    backend = FireDetectBackend(_HOST, _PORT, model_path=_MODEL)
    print(f"[fire_detect] target: {backend.video_url}")
    print("commands: ping | detect [conf] | start [conf] | state | stop | q/quit/exit\n")

    while True:
        try:
            raw = input("fire> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()
        if cmd in ("q", "quit", "exit"):
            break
        if cmd == "ping":
            print(json.dumps(backend.ping(), ensure_ascii=False))
            continue
        if cmd == "detect":
            conf = float(parts[1]) if len(parts) > 1 else None
            print(json.dumps(backend.detect(min_confidence=conf), ensure_ascii=False, indent=2))
            continue
        if cmd == "start":
            conf = float(parts[1]) if len(parts) > 1 else None
            print(json.dumps(backend.start_monitor(min_confidence=conf), ensure_ascii=False, indent=2))
            continue
        if cmd == "state":
            print(json.dumps(backend.monitor_state(), ensure_ascii=False, indent=2))
            continue
        if cmd == "stop":
            print(json.dumps(backend.stop_monitor(), ensure_ascii=False, indent=2))
            continue
        print("unknown command (ping | detect [conf] | start [conf] | state | stop)")


if __name__ == "__main__":
    main()
