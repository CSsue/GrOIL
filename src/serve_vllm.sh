#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
ENV_PATH="$PROJECT_ROOT/.env"

if [ -f "$ENV_PATH" ]; then
    set -a; source "$ENV_PATH"; set +a
    echo "✅ Loaded environment from $ENV_PATH"
else
    echo "❌ Error: .env not found"
    exit 1
fi

EXTRACTED_PORT=$(echo "$API_BASE" | grep -oE '[0-9]+' | head -n 1)
PORT=${EXTRACTED_PORT:-8000} 

echo "🚀 FORCING $MODEL_NAME on port $PORT ..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --served-model-name "$MODEL_NAME" \
    --port "$PORT" \
    --max-model-len 160000 \
    --gpu-memory-utilization 0.6 \
    --max-num-seqs 64     