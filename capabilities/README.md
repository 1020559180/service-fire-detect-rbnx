# capabilities -- fire_detect fire & smoke detection primitive contract surface (Schema A)

A primitive package independent of `dji_msdk`, using the `fire_detect` service namespace; it coexists with the drone primitives and can be
called by the executor **simultaneously** (`rbnx call robonix/primitive/drone/*` + `rbnx call
robonix/service/fire_detect/*`).

- `lib/vision/srv/*.srv` -- ROS2-style service definitions (request params + `std_msgs/String status`)
- `service/fire_detect/*.toml` -- primitive metadata (semantic id / version / idl / description)

## .srv convention

```
float64 min_confidence   # request: confidence threshold (0~1)
---
std_msgs/String status   # response: JSON string
```

## .toml convention (Schema A)

```toml
[contract]
id      = "robonix/service/fire_detect/fire_detect"
version = "1"
kind    = "primitive"
idl     = "vision/srv/FireDetect.srv"
description = "..."

[mode]
type = "rpc"
```

## Primitive set

- **core**: `fire_detect` -- grabs one video frame, detects flame/smoke with YOLO, attaches GPS on a hit
- **lifecycle**: `driver` -- started by `rbnx boot` via `Driver(CMD_INIT)`
