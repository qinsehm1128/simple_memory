#!/bin/bash
set -e

echo "==================================="
echo "Simple Memory - Starting services"
echo "==================================="

# Start web interface in background
echo "Starting Web interface on port 8765..."
simple-memory web --host 0.0.0.0 --port 8765 &
WEB_PID=$!

# Wait a moment for web to initialize
sleep 3

# Start MCP SSE server
echo "Starting MCP SSE server on port 8766..."
simple-memory serve --host 0.0.0.0 --port 8766 &
MCP_PID=$!

echo "==================================="
echo "Services started!"
echo "Web UI: http://localhost:8765"
echo "MCP SSE: http://localhost:8766/sse"
echo "==================================="

# Handle shutdown
trap "kill $WEB_PID $MCP_PID 2>/dev/null; exit 0" SIGTERM SIGINT

# Wait for any process to exit
wait -n

# If one exits, kill the other
kill $WEB_PID $MCP_PID 2>/dev/null
exit 1
