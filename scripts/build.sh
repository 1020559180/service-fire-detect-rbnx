#!/usr/bin/env bash
# fire_detect 构建脚本 —— 生成 MCP dataclass（fire_detect_mcp / std_msgs_mcp）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo "[fire_detect] rbnx codegen..."

if command -v rbnx &>/dev/null; then
    cd "$PROJECT_DIR"
    rbnx codegen -p . --out-dir rbnx-build/codegen --mcp
    echo "[fire_detect] codegen 完成"
else
    echo "[fire_detect] ⚠ rbnx 未安装，跳过 codegen"
    echo "                 安装: cargo install --git https://github.com/syswonder/robonix rbnx"
    mkdir -p "$PROJECT_DIR/rbnx-build"
    touch "$PROJECT_DIR/rbnx-build/.rbnx-built"
fi
