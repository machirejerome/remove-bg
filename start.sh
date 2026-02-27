#!/bin/bash
set -e

# ─── Find SAM3 model checkpoint ──────────────────────────────────────────────
# Search order:
#   1. Existing ComfyUI path on Network Volume
#   2. Custom SAM3_CHECKPOINT env var
#   3. Default download location
#   4. Download from HuggingFace if not found anywhere

COMFYUI_PATH="/runpod-volume/runpod-slim/ComfyUI/models/sam3/sam3.pt"
DEFAULT_PATH="/runpod-volume/sam3_models/sam3.pt"
CHECKPOINT="${SAM3_CHECKPOINT:-}"

if [ -f "$COMFYUI_PATH" ]; then
    CHECKPOINT="$COMFYUI_PATH"
    echo "✅ SAM3 found at ComfyUI path: $CHECKPOINT"
elif [ -n "$CHECKPOINT" ] && [ -f "$CHECKPOINT" ]; then
    echo "✅ SAM3 found at custom path: $CHECKPOINT"
elif [ -f "$DEFAULT_PATH" ]; then
    CHECKPOINT="$DEFAULT_PATH"
    echo "✅ SAM3 found at default path: $CHECKPOINT"
else
    echo "⏳ SAM3 model not found — downloading to $DEFAULT_PATH ..."
    mkdir -p "$(dirname "$DEFAULT_PATH")"
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='1038lab/sam3', filename='sam3.pt', local_dir='/runpod-volume/sam3_models'); print('✅ SAM3 downloaded!')"
    CHECKPOINT="$DEFAULT_PATH"
fi

export SAM3_CHECKPOINT="$CHECKPOINT"
echo "🚀 Starting handler with SAM3_CHECKPOINT=$CHECKPOINT"

exec python -u handler.py
