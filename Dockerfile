# =============================================================================
# Remove BG – RunPod Serverless Endpoint
# SAM3 (native Python library from comfyui-rmbg) — exact ComfyUI replica
# =============================================================================
#
# Build:
#   docker build -t remove-bg .
#
# The SAM3 model (sam3.pt, ~3.3GB) is downloaded from the public 1038lab/sam3
# mirror on HuggingFace — NO access request needed.
# =============================================================================

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── SAM3 Library ───────────────────────────────────────────────────────────
# Clone the comfyui-rmbg repo to get the SAM3 Python library
# We only need the models/sam3/ directory (the native SAM3 code)
RUN git clone --depth 1 https://github.com/1038lab/ComfyUI-RMBG.git /tmp/comfyui-rmbg && \
    mkdir -p /app/sam3_lib && \
    cp -r /tmp/comfyui-rmbg/models/sam3/* /app/sam3_lib/ && \
    rm -rf /tmp/comfyui-rmbg

# ─── SAM3 Model Checkpoint ─────────────────────────────────────────────────
# Download sam3.pt from public 1038lab/sam3 mirror (no access request needed)
RUN mkdir -p /app/sam3_models && \
    python -c "from huggingface_hub import hf_hub_download; path = hf_hub_download(repo_id='1038lab/sam3', filename='sam3.pt', local_dir='/app/sam3_models'); print(f'Downloaded SAM3 checkpoint: {path}')"

# Copy handler
COPY handler.py .

# Environment
ENV SAM3_LIB_DIR=/app/sam3_lib
ENV SAM3_CHECKPOINT=/app/sam3_models/sam3.pt
ENV SAM3_BPE_PATH=/app/sam3_lib/assets/bpe_simple_vocab_16e6.txt.gz

# RunPod entrypoint
CMD ["python", "-u", "handler.py"]
