#!/usr/bin/env bash
# Launch NanoChat Web Console
cd "$(dirname "$0")"

PYTHON="./.venv/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "=========================================="
echo "🚀 启动 NanoChat 交互式 Web 控制台"
echo "🌐 访问地址: http://127.0.0.1:8080"
echo "=========================================="

$PYTHON -m scripts.chat_server
