#!/usr/bin/env bash
# fire_detect start script -- called by rbnx boot to start the fire & smoke detection primitive driver.
#
# Environment variables:
#   RBNX_ATLAS            -- Atlas gRPC address (injected automatically by rbnx)
#   RBNX_CAP_CONFIG_JSON  -- primitive config JSON (injected from the manifest's config)
#   RC_PRO_IP / RC_PRO_PORT -- DJI bridge APK address (overrides config)
#   FIRE_MODEL_PATH       -- fire & smoke weights path (overrides config)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# ============================================================
# 1. Configure the Robonix Python environment
# ============================================================
ROBONIX_PYLIB="${ROBONIX_PYLIB:-$HOME/robonix/pylib/robonix-api}"
CODEGEN_DIR="$PROJECT_DIR/rbnx-build/codegen/proto_gen"
MCP_TYPES_DIR="$PROJECT_DIR/rbnx-build/codegen/robonix_mcp_types"
export PYTHONPATH="$PROJECT_DIR:$ROBONIX_PYLIB:$CODEGEN_DIR:$MCP_TYPES_DIR:${PYTHONPATH:-}"

# ============================================================
# 2. Resolve address / weights (manifest config first, env vars override)
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
# 3. Install runtime dependencies
# ============================================================
pip3 install -q requests opencv-python onnxruntime 2>/dev/null || true
pip3 install -q ultralytics 2>/dev/null || true   # note: pulls in the torch dependency; on Jetson install torch first via JetPack/NVIDIA
pip3 install -q "mcp>=1.0,<2" 2>/dev/null || true

# ============================================================
# 4. Start the fire & smoke detection primitive driver
# ============================================================
exec python3 -u -m fire_detect.driver
