# =============================================================================
# RunPod Serverless Worker: Qwen Camera Control (Multi-Angle)
# =============================================================================
# LIGHTWEIGHT image (~5GB) — model weights are stored on RunPod Network Volume.
# First cold start downloads ~25GB to /runpod-volume, then cached for reuse.
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY handler.py .
COPY qwenimage/ qwenimage/

# Network Volume mount point (configured in RunPod endpoint settings)
ENV VOLUME_PATH=/runpod-volume

CMD ["python", "-u", "handler.py"]
