"""
RunPod Serverless Handler: Qwen Camera Control (Multi-Angle)

Replicates the exact HF Space setup from linoyts/Qwen-Image-Edit-Angles:
  - Base pipeline: Qwen/Qwen-Image-Edit-2509
  - Transformer: linoyts/Qwen-Image-Edit-Rapid-AIO (Lightning-distilled, 4-step)
  - LoRA: dx8152/Qwen-Edit-2509-Multiple-angles (fused at scale 1.25)
  - Bilingual Chinese+English camera prompt construction

Models are cached on RunPod Network Volume at /runpod-volume/models/
to avoid re-downloading ~25GB on every cold start.
"""

import os
import io
import base64
import random
import time
import requests
import torch
import numpy as np
import runpod
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

# RunPod Network Volume mount point
VOLUME_PATH = os.environ.get("VOLUME_PATH", "/runpod-volume")
MODEL_CACHE_DIR = os.path.join(VOLUME_PATH, "models", "qwen-camera-control")

# HuggingFace cache directory → point to network volume
os.environ["HF_HOME"] = os.path.join(VOLUME_PATH, "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(VOLUME_PATH, "huggingface")

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
TRANSFORMER_MODEL = "linoyts/Qwen-Image-Edit-Rapid-AIO"
LORA_MODEL = "dx8152/Qwen-Edit-2509-Multiple-angles"
LORA_WEIGHT_NAME = "镜头转换.safetensors"
LORA_SCALE = 1.25  # Matches HF Space: pipe.fuse_lora(lora_scale=1.25)

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = np.iinfo(np.int32).max

# =============================================================================
# Model Loading (at module level for warm starts)
# =============================================================================

def load_pipeline():
    """Load the pipeline, downloading models to network volume if needed."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    # Try importing from diffusers first (latest git version)
    try:
        from diffusers import QwenImageEditPlusPipeline
        from diffusers.models import QwenImageTransformer2DModel
        print("[startup] Using built-in diffusers QwenImage classes")
    except ImportError:
        from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
        from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
        print("[startup] Using vendored qwenimage classes")

    # Ensure cache directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    print(f"[startup] HF_HOME = {os.environ['HF_HOME']}")
    print(f"[startup] Loading transformer from {TRANSFORMER_MODEL} ...")
    start = time.time()

    transformer = QwenImageTransformer2DModel.from_pretrained(
        TRANSFORMER_MODEL,
        subfolder="transformer",
        torch_dtype=DTYPE,
    )
    print(f"[startup] Transformer loaded in {time.time() - start:.1f}s")

    print(f"[startup] Loading pipeline from {BASE_MODEL} ...")
    start = time.time()

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL,
        transformer=transformer,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    print(f"[startup] Pipeline loaded in {time.time() - start:.1f}s. Loading LoRA ...")
    start = time.time()

    # Load and fuse the multi-angle LoRA at scale 1.25 (matching HF Space exactly)
    pipe.load_lora_weights(
        LORA_MODEL,
        weight_name=LORA_WEIGHT_NAME,
        adapter_name="angles",
    )
    pipe.set_adapters(["angles"], adapter_weights=[1.0])
    pipe.fuse_lora(adapter_names=["angles"], lora_scale=LORA_SCALE)
    pipe.unload_lora_weights()

    print(f"[startup] LoRA fused (scale={LORA_SCALE}) in {time.time() - start:.1f}s. Ready!")
    return pipe


print("[startup] Initializing Qwen Camera Control pipeline ...")
total_start = time.time()
pipe = load_pipeline()
print(f"[startup] Total init time: {time.time() - total_start:.1f}s")


# =============================================================================
# Bilingual Camera Prompt Construction (matches HF Space build_camera_prompt)
# =============================================================================

def build_camera_prompt(
    rotate_deg: float = 0.0,
    move_forward: float = 0.0,
    vertical_tilt: float = 0.0,
    use_wide_angle: bool = False,
) -> str:
    """Build a bilingual Chinese+English camera movement prompt.

    This exactly matches the HF Space's build_camera_prompt() function.
    The bilingual format is critical because the LoRA was trained on these
    specific Chinese+English prompt patterns.
    """
    prompt_parts = []

    if rotate_deg != 0:
        direction = "left" if rotate_deg > 0 else "right"
        if direction == "left":
            prompt_parts.append(
                f"将镜头向左旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the left."
            )
        else:
            prompt_parts.append(
                f"将镜头向右旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the right."
            )

    if move_forward > 5:
        prompt_parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
    elif move_forward >= 1:
        prompt_parts.append("将镜头向前移动 Move the camera forward.")

    if vertical_tilt <= -1:
        prompt_parts.append("将相机转向鸟瞰视角 Turn the camera to a bird's-eye view.")
    elif vertical_tilt >= 1:
        prompt_parts.append("将相机切换到仰视视角 Turn the camera to a worm's-eye view.")

    if use_wide_angle:
        prompt_parts.append("将镜头转为广角镜头 Turn the camera to a wide-angle lens.")

    final_prompt = " ".join(prompt_parts).strip()
    return final_prompt if final_prompt else "no camera movement"


# =============================================================================
# Image Utilities
# =============================================================================

def load_image_from_input(image_input: str) -> Image.Image:
    """Load an image from a URL or base64 string."""
    if image_input.startswith("http://") or image_input.startswith("https://"):
        response = requests.get(image_input, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        # Assume base64
        if "," in image_input:  # data:image/...;base64,XXXX
            image_input = image_input.split(",", 1)[1]
        image_bytes = base64.b64decode(image_input)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def image_to_base64(image: Image.Image, format: str = "WEBP", quality: int = 95) -> str:
    """Convert a PIL Image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =============================================================================
# RunPod Handler
# =============================================================================

def handler(job):
    """Process a camera control image edit job.

    Input schema:
        - image (str, required): URL or base64-encoded image
        - rotate_degrees (float, optional): Camera rotation in degrees. Default: 0
        - move_forward (float, optional): Zoom/dolly. 0=none, 5=forward, 10=close-up. Default: 0
        - vertical_tilt (float, optional): -1=bird's-eye, 0=level, 1=worm's-eye. Default: 0
        - use_wide_angle (bool, optional): Wide-angle lens effect. Default: false
        - prompt (str, optional): Additional styling prompt appended after camera directive
        - seed (int, optional): RNG seed for reproducibility. Default: random
        - num_inference_steps (int, optional): Denoising steps. Default: 4
        - true_guidance_scale (float, optional): CFG scale. Default: 1.0
        - output_format (str, optional): "webp", "png", or "jpeg". Default: "webp"
        - output_quality (int, optional): Quality for lossy formats. Default: 95

    Output:
        - image_base64 (str): Base64-encoded output image
        - seed (int): Seed used for generation
        - prompt (str): Full prompt sent to the model (for debugging)
        - format (str): Output format used
    """
    job_input = job["input"]

    # --- Parse inputs ---
    image_input = job_input.get("image")
    if not image_input:
        return {"error": "Missing required 'image' input (URL or base64)"}

    rotate_degrees = float(job_input.get("rotate_degrees", 0))
    move_forward = float(job_input.get("move_forward", 0))
    vertical_tilt = float(job_input.get("vertical_tilt", 0))
    use_wide_angle = bool(job_input.get("use_wide_angle", False))
    extra_prompt = job_input.get("prompt", "")
    seed = job_input.get("seed", None)
    randomize_seed = seed is None
    num_inference_steps = int(job_input.get("num_inference_steps", 4))
    true_guidance_scale = float(job_input.get("true_guidance_scale", 1.0))
    output_format = job_input.get("output_format", "webp").upper()
    output_quality = int(job_input.get("output_quality", 95))

    # Map format names
    format_map = {"WEBP": "WEBP", "PNG": "PNG", "JPEG": "JPEG", "JPG": "JPEG"}
    output_format = format_map.get(output_format, "WEBP")

    # --- Build prompt ---
    camera_prompt = build_camera_prompt(rotate_degrees, move_forward, vertical_tilt, use_wide_angle)

    if camera_prompt == "no camera movement" and not extra_prompt:
        return {"error": "No camera movement or prompt specified. Nothing to do."}

    # Combine camera prompt with optional extra styling prompt
    if extra_prompt:
        full_prompt = f"{camera_prompt} {extra_prompt}" if camera_prompt != "no camera movement" else extra_prompt
    else:
        full_prompt = camera_prompt

    print(f"[handler] Prompt: {full_prompt}")
    print(f"[handler] Params: rotate={rotate_degrees}, forward={move_forward}, tilt={vertical_tilt}, wide={use_wide_angle}")

    # --- Load image ---
    try:
        input_image = load_image_from_input(image_input)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}

    # --- Seed ---
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    # --- Inference ---
    try:
        result = pipe(
            image=[input_image],
            prompt=full_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=1,
        ).images[0]
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    # --- Encode output ---
    image_b64 = image_to_base64(result, format=output_format, quality=output_quality)

    return {
        "image_base64": image_b64,
        "seed": seed,
        "prompt": full_prompt,
        "format": output_format.lower(),
    }


# =============================================================================
# Entry Point
# =============================================================================

runpod.serverless.start({"handler": handler})
