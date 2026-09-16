# capabilities — fire_detect 烟火检测原语契约面（Schema A）

独立于 `dji_msdk` 的原语包，使用 `fire_detect` 服务命名空间，与无人机原语并存、可被
executor **同时调用**（`rbnx call robonix/primitive/drone/*` + `rbnx call
robonix/service/fire_detect/*`）。

- `lib/vision/srv/*.srv` —— ROS2 风格服务定义（请求参数 + `std_msgs/String status`）
- `service/fire_detect/*.toml` —— 原语元数据（语义 id / version / idl / description）

## .srv 约定

```
float64 min_confidence   # 请求：置信度阈值（0~1）
---
std_msgs/String status   # 响应：JSON 字符串
```

## .toml 约定（Schema A）

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

## 原语集

- **核心**：`fire_detect` —— 拉取一帧视频画面，YOLO 识别火焰/烟雾，命中时附带 GPS
- **生命周期**：`driver` —— `rbnx boot` 通过 `Driver(CMD_INIT)` 拉起
