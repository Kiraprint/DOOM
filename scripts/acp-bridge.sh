#!/usr/bin/env bash
# ACP Bridge: opencode-ai ACP → WebSocket → marimo
# Run this BEFORE launching marimo to enable AI assistant in notebooks.
set -euo pipefail

PORT="${1:-3023}"

echo "Starting ACP bridge on ws://127.0.0.1:${PORT} ..."
echo "Marimo will connect to this as its AI assistant backend."
echo "Press Ctrl+C to stop."

# stdio-to-ws bridges opencode-ai's ACP (stdio) to WebSocket
exec npx stdio-to-ws "npx opencode-ai acp" --port "${PORT}"
