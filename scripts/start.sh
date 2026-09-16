#!/usr/bin/env bash
# fire_detect 启动脚本 —— 由 rbnx boot 调用，启动烟火检测原语驱动。
#
# 环境变量：
#   RBNX_ATLAS            — Atlas gRPC 地址（rbnx 自动注入）
#   RBNX_CAP_CONFIG_JSON  — 原语配置 JSON（从 manifest 的 config 注入）
#   RC_PRO_IP / RC_PRO_PORT — DJI 桥接 APK 地址（优先于 config）
#   FIRE_MODEL_PATH       — 烟火权重路径（优先于 config）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# ============================================================
# 1. 配置 Robonix Python 环境
# ============================================================
ROBONIX_PYLIB="${ROBONIX_PYLIB:-$HOME/robonix/pylib/robonix-api}"
CODEGEN_DIR="$PROJECT_DIR/rbnx-build/codegen/proto_gen"
MCP_TYPES_DIR="$PROJECT_DIR/rbnx-build/codegen/robonix_mcp_types"
export PYTHONPATH="$PROJECT_DIR:$ROBONIX_PYLIB:$CODEGEN_DIR:$MCP_TYPES_DIR:${PYTHONPATH:-}"

# ============================================================
# 2. 解析地址/权重（manifest config 优先，环境变量覆盖）
# ============================================================
if [ -n "${RBNX_CAP_CONFIG_JSON:-}" ]; then
    RC_IP=$(python3 -c "import json,sys; c=json.loads(sys.argv[1]); print(c.get('rc_pro_ip','172.20.10.3'))" "$RBNX_CAP_CONFIG_JSON" 2>/dev/null || echo "172.20.10.3")
    RC_PORT=$(python3 -c "import json,sys; c=json.loads(sys.argv[1]); print(c.get('rc_pro_port',8080))" "$RBNX_CAP_CONFIG_JSON" 2>/dev/null || echo "8080")
    export RC_PRO_IP="${RC_PRO_IP:-$RC_IP}"
    export RC_PRO_PORT="${RC_PRO_PORT:-$RC_PORT}"
else
    export RC_PRO_IP="${RC_PRO_IP:-172.20.10.3}"
    export RC_PRO_PORT="${RC_PRO_PORT:-8080}"
fi

echo "[fire_detect] RC Pro: ${RC_PRO_IP}:${RC_PRO_PORT}"
echo "[fire_detect] Atlas:  ${RBNX_ATLAS:-auto}"

# ============================================================
# 3. 安装运行时依赖
# ============================================================
pip3 install -q requests opencv-python onnxruntime 2>/dev/null || true
pip3 install -q ultralytics 2>/dev/null || true   # 注意：会拉 torch 依赖；Jetson 请先用 JetPack/NVIDIA 装好 torch
pip3 install -q "mcp>=1.0,<2" 2>/dev/null || true

# ============================================================
# 4. 启动烟火检测原语驱动
# ============================================================
exec python3 -u -m fire_detect.driver
