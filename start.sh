#!/bin/bash
set -e

MODEL_DIR="${SAM3_MODEL_DIR:-/runpod-volume/sam3_models}"
CHECKPOINT="${MODEL_DIR}/sam3.pt"

# Download SAM3 model if not already cached (e.g. on Network Volume)
if [ ! -f "$CHECKPOINT" ]; then
    echo "⏳ SAM3 model not found at $CHECKPOINT — downloading..."
    mkdir -p "$MODEL_DIR"
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='1038lab/sam3', filename='sam3.pt', local_dir='$MODEL_DIR'); print('✅ SAM3 model downloaded!')"
else
    echo "✅ SAM3 model found at $CHECKPOINT (cached)"
fi

# Set env for handler
export SAM3_CHECKPOINT="$CHECKPOINT"

# Start handler
exec python -u handler.py
