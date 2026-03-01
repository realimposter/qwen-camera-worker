"""
Download and cache all model weights at Docker build time.
This runs once during `docker build` so models are baked into the image.
"""

import torch

# Try built-in diffusers first
try:
    from diffusers import QwenImageEditPlusPipeline
    from diffusers.models import QwenImageTransformer2DModel
    print("[build] Using built-in diffusers QwenImage classes")
except ImportError:
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
    print("[build] Using vendored qwenimage classes")

print("[build] Downloading transformer: linoyts/Qwen-Image-Edit-Rapid-AIO ...")
transformer = QwenImageTransformer2DModel.from_pretrained(
    "linoyts/Qwen-Image-Edit-Rapid-AIO",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

print("[build] Downloading base pipeline: Qwen/Qwen-Image-Edit-2509 ...")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

print("[build] Downloading LoRA: dx8152/Qwen-Edit-2509-Multiple-angles ...")
pipe.load_lora_weights(
    "dx8152/Qwen-Edit-2509-Multiple-angles",
    weight_name="镜头转换.safetensors",
    adapter_name="angles",
)

print("[build] All models cached. Done!")
