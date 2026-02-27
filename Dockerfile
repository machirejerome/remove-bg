# =============================================================================
# Remove BG – RunPod Serverless Endpoint
# SAM3 (native Python library from comfyui-rmbg) — exact ComfyUI replica
# =============================================================================
#
# Build:  docker build -t remove-bg .
#
# The Docker image is kept LIGHTWEIGHT (~3GB). The SAM3 model (3.3GB) is
# downloaded at first startup and cached on the RunPod Network Volume.
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

# SAM3 Python library (code only, ~34MB — NOT the model)
RUN git clone --depth 1 https://github.com/1038lab/ComfyUI-RMBG.git /tmp/comfyui-rmbg && \
    mkdir -p /app/sam3_lib && \
    cp -r /tmp/comfyui-rmbg/models/sam3/* /app/sam3_lib/ && \
    rm -rf /tmp/comfyui-rmbg

# Copy handler + startup script
COPY handler.py .
COPY start.sh .
RUN chmod +x start.sh

# Environment
ENV SAM3_LIB_DIR=/app/sam3_lib
ENV SAM3_BPE_PATH=/app/sam3_lib/assets/bpe_simple_vocab_16e6.txt.gz
ENV SAM3_MODEL_DIR=/runpod-volume/sam3_models

# Start: downloads model if needed, then runs handler
CMD ["./start.sh"]
