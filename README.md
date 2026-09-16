# fire_detect -- drone fire & smoke detection primitive

A RoboNIX primitive package independent of `dji_msdk`, in the `robonix/service/fire_detect/*` namespace.
It coexists with `dji_msdk` (`robonix/primitive/drone/*`) and can be called by the executor **simultaneously**:

```bash
# Drone takeoff / state
rbnx call robonix/primitive/drone/takeoff '{"altitude": 5.0}'
rbnx call robonix/primitive/drone/state

# Fire & smoke detection (single frame: grab one frame and infer; returns GPS on a hit)
rbnx call robonix/service/fire_detect/fire_detect '{"min_confidence": 0.4}'

# Real-time fire & smoke monitoring (background continuous frame inference, polled by an external loop)
rbnx call robonix/service/fire_detect/start_monitor
rbnx call robonix/service/fire_detect/monitor_state      # poll repeatedly to read the latest alert/gps
rbnx call robonix/service/fire_detect/stop_monitor
```

## Architecture

```
RoboNIX executor
   ├─ robonix/primitive/drone/*   → dji_msdk (drone control)
   └─ robonix/service/fire_detect/*  → fire_detect (fire & smoke detection)
          │ MJPEG stream pull + YOLO inference
          ▼
   DJI bridge APK :8080/api/video + /api/capture_gps
          │
          ▼
   drone camera frame
```

Fire & smoke detection **reads the stream** and does not control the aircraft; GPS is fetched live via `/api/capture_gps` and only attached on a
hit, for use by alert primitives / upper layers.

## Layout

```
fire_detect/
  package_manifest.yaml            # package metadata + capabilities + config
  capabilities/                    # contracts (Schema A, consumed by rbnx codegen --mcp)
  fire_detect/
    __init__.py
    backend.py                     # frame grab + inference + GPS (SDK-agnostic, pure Python)
    driver.py                      # @vision.mcp(...) dispatch layer
    main.py                        # standalone REPL (framework-free debugging)
  models/                          # fire & smoke YOLO weights (default fallback ultralytics yolov8n)
  scripts/build.sh / start.sh
  requirements.txt
```

## Model

Model loading priority in `backend.py`:

1. weights specified by `config.model_path` (`.pt` / `.engine` / `.onnx`)
2. the `FIRE_MODEL_PATH` environment variable
3. the first weights file in the `models/` directory
4. fallback to the official ultralytics `yolov8n` (no fire/smoke-specific classes; only validates the pipeline)

Production tip: use a self-trained fire/smoke two-class YOLO weights file; export a TensorRT `.engine` on Jetson
(FP16) for real-time inference.

## Dependencies

- `requests` (stream pull / GPS)
- `opencv-python` (MJPEG decoding)
- `ultralytics` (YOLO inference; for TensorRT deployment only `torch` + engine are needed)
