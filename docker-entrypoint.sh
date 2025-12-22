#!/bin/bash
set -e

# Check CPU compatibility
check_cpu() {
    if grep -q avx2 /proc/cpuinfo 2>/dev/null; then
        echo "CPU supports AVX2"
        return 0
    elif grep -q sse4 /proc/cpuinfo 2>/dev/null; then
        echo "CPU supports SSE4 (may have limited performance)"
        return 0
    else
        echo "WARNING: CPU may not support required instructions for LanceDB"
        echo "If you see 'Illegal instruction' errors, your CPU is not compatible"
        return 1
    fi
}

# Set environment for compatibility
export OPENBLAS_CORETYPE=${OPENBLAS_CORETYPE:-ARMV8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

echo "==================================="
echo "Simple Memory - Starting services"
echo "==================================="

# Check CPU
check_cpu || true

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
