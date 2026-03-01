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

# Pre-download all model weights into HF cache at build time
# This makes the image ~30GB but eliminates download time at runtime
RUN python -c "
import torch

# Try built-in diffusers first
try:
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.models import QwenImageTransformer2DModel
except ImportError:
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

print('[build] Downloading transformer...')
transformer = QwenImageTransformer2DModel.from_pretrained(
    'linoyts/Qwen-Image-Edit-Rapid-AIO',
    subfolder='transformer',
    torch_dtype=torch.bfloat16,
)

print('[build] Downloading base pipeline...')
pipe = QwenImageEditPlusPipeline.from_pretrained(
    'Qwen/Qwen-Image-Edit-2509',
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

print('[build] Downloading LoRA...')
pipe.load_lora_weights(
    'dx8152/Qwen-Edit-2509-Multiple-angles',
    weight_name='镜头转换.safetensors',
    adapter_name='angles',
)

print('[build] All models cached. Done!')
"

CMD ["python", "-u", "handler.py"]
