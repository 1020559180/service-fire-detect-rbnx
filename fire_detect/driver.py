#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""fire_detect 驱动 —— 烟火检测 RoboNIX 原语（MCP handler 层）。

能力通过 **MCP** 暴露，类型使用 ``rbnx codegen --mcp`` 生成的
``fire_detect_mcp`` / ``std_msgs_mcp`` dataclass。

Capability surface:
  robonix/service/fire_detect/fire_detect    rpc  检测画面中的烟火（命中附带 GPS）单帧
  robonix/service/fire_detect/start_monitor  rpc  启动实时监控（后台持续拉帧推理）
  robonix/service/fire_detect/stop_monitor   rpc  停止实时监控
  robonix/service/fire_detect/monitor_state  rpc  读取监控最新结果（供外部循环轮询）

本文件 SDK 无关：只调 ``fire_detect.backend.FireDetectBackend`` 的语义方法，
不出现任何 MSDK/HTTP 术语。换摄像头/SDK 只换后端。
"""
from __future__ import annotations

import json
import os
from typing import Any

from robonix_api import Service, Ok  # type: ignore

from .backend import FireDetectBackend
from .stream import AnnotatedStreamer

# ── 导入 codegen 生成的 MCP dataclass ──
# 由 `rbnx codegen -p . --out-dir rbnx-build/codegen --mcp` 生成到
# rbnx-build/codegen/robonix_mcp_types/ 下。
import fire_detect_mcp  # noqa: E402
import std_msgs_mcp  # noqa: E402

detector = Service(id="fire_detect", namespace="robonix/service/fire_detect")

_backend: FireDetectBackend | None = None
_streamer: AnnotatedStreamer | None = None


def _get_backend() -> FireDetectBackend:
    global _backend
    if _backend is None:
        host = os.environ.get("RC_PRO_IP", "172.20.10.3")
        port = int(os.environ.get("RC_PRO_PORT", "8080"))
        _backend = FireDetectBackend(host, port)
    return _backend


def _status(result) -> std_msgs_mcp.String:
    """把 dict 结果序列化为 std_msgs/String 响应字段。"""
    return std_msgs_mcp.String(data=json.dumps(result, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 烟火检测
# ═══════════════════════════════════════════════════════════════════════════════

@detector.mcp("robonix/service/fire_detect/fire_detect")
def fire_detect(req: fire_detect_mcp.FireDetect_Request) -> fire_detect_mcp.FireDetect_Response:
    """检测当前画面中的烟火（火焰/烟雾），命中时附带当前 GPS 坐标。"""
    conf = float(req.min_confidence) if req.min_confidence > 0 else None
    result = _get_backend().detect(min_confidence=conf)
    return fire_detect_mcp.FireDetect_Response(status=_status(result))


# ═══════════════════════════════════════════════════════════════════════════════
# 实时监控（后台持续拉帧推理，外部循环轮询）
# ═══════════════════════════════════════════════════════════════════════════════

@detector.mcp("robonix/service/fire_detect/start_monitor")
def start_monitor(_req: fire_detect_mcp.StartMonitor_Request) -> fire_detect_mcp.StartMonitor_Response:
    """启动烟火实时监控：后台线程持续拉帧推理，最新结果由 monitor_state 轮询。"""
    result = _get_backend().start_monitor()
    return fire_detect_mcp.StartMonitor_Response(status=_status(result))


@detector.mcp("robonix/service/fire_detect/stop_monitor")
def stop_monitor(_req: fire_detect_mcp.StopMonitor_Request) -> fire_detect_mcp.StopMonitor_Response:
    """停止烟火实时监控。"""
    result = _get_backend().stop_monitor()
    return fire_detect_mcp.StopMonitor_Response(status=_status(result))


@detector.mcp("robonix/service/fire_detect/monitor_state")
def monitor_state(_req: fire_detect_mcp.MonitorState_Request) -> fire_detect_mcp.MonitorState_Response:
    """读取实时监控最新结果（alert/detections/gps/timestamp）。"""
    result = _get_backend().monitor_state()
    return fire_detect_mcp.MonitorState_Response(status=_status(result))


# ═══════════════════════════════════════════════════════════════════════════════
# Lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

@detector.on_init
def init(config: dict | None) -> Any:
    """启动时初始化后端并探测 RC Pro 连接。"""
    global _backend, _streamer
    cfg = config or {}
    host = cfg.get("rc_pro_ip") or os.environ.get("RC_PRO_IP", "172.20.10.3")
    port = int(cfg.get("rc_pro_port") or os.environ.get("RC_PRO_PORT", "8080"))
    model_path = cfg.get("model_path", "") or os.environ.get("FIRE_MODEL_PATH", "")
    min_conf = float(cfg.get("min_confidence", 0.4))
    _backend = FireDetectBackend(host, port, model_path=model_path, min_confidence=min_conf)
    if not _backend.ping().get("success"):
        print(f"[fire_detect] ⚠ 无法连接到 RC Pro ({host}:{port})，将继续注册但调用可能失败", flush=True)
    else:
        print(f"[fire_detect] ✅ 已连接到 RC Pro ({host}:{port})", flush=True)

    # 实时可视化：随原语一起启动 web 标注流（边检测边画框，供 web/app 观看）
    stream_port = int(cfg.get("stream_port", 8081))
    _streamer = AnnotatedStreamer(_backend, conf=min_conf, port=stream_port)
    _streamer.start()
    return Ok()


@detector.on_shutdown
def shutdown() -> Any:
    """关闭时清理：先停监控线程，再清后端。"""
    global _backend, _streamer
    if _streamer is not None:
        _streamer.stop()
        _streamer = None
    if _backend is not None:
        _backend.stop_monitor()
    _backend = None
    return Ok()


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    detector.run()
