# =============================================================================
# RunPod Serverless Worker: Qwen Camera Control (Multi-Angle)
# =============================================================================
# Models are baked into the image at build time (~30GB total).
# Cold starts load from local disk in ~30s.
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

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY handler.py .
COPY qwenimage/ qwenimage/
COPY download_models.py .

# Pre-download all model weights into HF cache at build time
RUN python download_models.py

CMD ["python", "-u", "handler.py"]
