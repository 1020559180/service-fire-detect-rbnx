#!/usr/bin/env bash
# fire_detect build script -- generate MCP dataclasses (fire_detect_mcp / std_msgs_mcp).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

echo "[fire_detect] rbnx codegen..."

if command -v rbnx &>/dev/null; then
    cd "$PROJECT_DIR"
    rbnx codegen -p . --out-dir rbnx-build/codegen --mcp
    echo "[fire_detect] codegen complete"
else
    echo "[fire_detect] ⚠ rbnx not installed, skipping codegen"
    echo "                 install: cargo install --git https://github.com/syswonder/robonix rbnx"
    mkdir -p "$PROJECT_DIR/rbnx-build"
    touch "$PROJECT_DIR/rbnx-build/.rbnx-built"
fi
