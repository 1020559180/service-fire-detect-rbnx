# fire_detect —— 无人机烟火检测原语

独立于 `dji_msdk` 的 RoboNIX 原语包，命名空间 `robonix/service/fire_detect/*`。
与 `dji_msdk`（`robonix/primitive/drone/*`）并存，可被 executor **同时调用**：

```bash
# 无人机起飞 / 状态
rbnx call robonix/primitive/drone/takeoff '{"altitude": 5.0}'
rbnx call robonix/primitive/drone/state

# 烟火检测（单帧：拉一帧画面推理；命中时返回 GPS）
rbnx call robonix/service/fire_detect/fire_detect '{"min_confidence": 0.4}'

# 烟火实时监控（后台持续拉帧推理，外部循环轮询）
rbnx call robonix/service/fire_detect/start_monitor
rbnx call robonix/service/fire_detect/monitor_state      # 反复轮询，读最新 alert/gps
rbnx call robonix/service/fire_detect/stop_monitor
```

## 架构

```
RoboNIX executor
   ├─ robonix/primitive/drone/*   → dji_msdk（无人机控制）
   └─ robonix/service/fire_detect/*  → fire_detect（烟火检测）
          │ MJPEG 拉流 + YOLO 推理
          ▼
   DJI 桥接 APK :8080/api/video + /api/capture_gps
          │
          ▼
   无人机镜头画面
```

烟火检测**读流**，不控制飞机；GPS 通过 `/api/capture_gps` 实时获取，仅在命中
烟火时附带返回，供告警原语/上层使用。

## 目录

```
fire_detect/
  package_manifest.yaml            # 包元数据 + capabilities + config
  capabilities/                    # 契约（Schema A，由 rbnx codegen --mcp 消费）
  fire_detect/
    __init__.py
    backend.py                     # 拉帧 + 推理 + GPS（SDK 无关，纯 Python）
    driver.py                      # @vision.mcp(...) 分发层
    main.py                        # standalone REPL（无框架联调）
  models/                          # 烟火 YOLO 权重（默认回退 ultralytics yolov8n）
  scripts/build.sh / start.sh
  requirements.txt
```

## 模型

`backend.py` 的加载优先级：

1. `config.model_path` 指定权重（`.pt` / `.engine` / `.onnx`）
2. 环境变量 `FIRE_MODEL_PATH`
3. `models/` 目录下第一个权重文件
4. 回退 ultralytics 官方 `yolov8n`（无烟火专项类别，仅验证链路）

生产建议：用自训的 fire/smoke 二分类 YOLO 权重；Jetson 上导出 TensorRT `.engine`
（FP16）实时推理。

## 依赖

- `requests`（拉流/GPS）
- `opencv-python`（MJPEG 解码）
- `ultralytics`（YOLO 推理；TensorRT 部署则只需 `torch` + engine）
