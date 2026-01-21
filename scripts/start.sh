#!/bin/bash
set -e

# Production startup script for ShotGraph
# This script handles initialization and starts the application

echo "Starting ShotGraph Production Server..."

# Wait for GPU to be available (if enabled)
if [ "$GPU_ENABLED" = "true" ]; then
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi || echo "Warning: nvidia-smi failed, but continuing..."
    else
        echo "Warning: nvidia-smi not found, GPU may not be available"
    fi
fi

# Verify required directories exist
mkdir -p "${STORAGE_PATH:-/app/output}"
mkdir -p "${ASSETS_PATH:-/app/assets}"

# Set default values if not provided
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export WORKERS="${WORKERS:-1}"

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Execution Profile: ${EXECUTION_PROFILE:-prod_gpu}"
echo "  GPU Enabled: ${GPU_ENABLED:-true}"
echo "  Storage Path: ${STORAGE_PATH:-/app/output}"

# Run database migrations or initialization if needed
# python -m app.migrate

# Start the application
exec uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --no-access-log \
    --log-level info
