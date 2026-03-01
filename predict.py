"""
Replicate Cog Predictor: Qwen Camera Control (Multi-Angle)

Replicates the exact HF Space setup from linoyts/Qwen-Image-Edit-Angles:
  - Base pipeline: Qwen/Qwen-Image-Edit-2509
  - Transformer: linoyts/Qwen-Image-Edit-Rapid-AIO (Lightning-distilled, 4-step)
  - LoRA: dx8152/Qwen-Edit-2509-Multiple-angles (fused at scale 1.25)
  - Bilingual Chinese+English camera prompt construction
"""

import io
import random
import time
import tempfile

import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, Path

MAX_SEED = np.iinfo(np.int32).max

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
        if rotate_deg > 0:
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


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory for fast predictions."""
        try:
            from diffusers import QwenImageEditPlusPipeline
            from diffusers.models import QwenImageTransformer2DModel
        except ImportError:
            from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
            from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel

        print("[setup] Loading transformer...")
        start = time.time()
        transformer = QwenImageTransformer2DModel.from_pretrained(
            "linoyts/Qwen-Image-Edit-Rapid-AIO",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        print(f"[setup] Loading pipeline... ({time.time() - start:.1f}s)")
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        print("[setup] Loading and fusing LoRA at scale 1.25...")
        self.pipe.load_lora_weights(
            "dx8152/Qwen-Edit-2509-Multiple-angles",
            weight_name="镜头转换.safetensors",
            adapter_name="angles",
        )
        self.pipe.set_adapters(["angles"], adapter_weights=[1.0])
        self.pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
        self.pipe.unload_lora_weights()

        print(f"[setup] Ready! Total: {time.time() - start:.1f}s")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        rotate_degrees: float = Input(
            description="Camera rotation in degrees. Positive=left, negative=right.",
            default=0,
            ge=-90,
            le=90,
        ),
        move_forward: float = Input(
            description="Zoom/dolly. 0=none, 5=forward, 10=close-up.",
            default=0,
            ge=0,
            le=10,
        ),
        vertical_tilt: float = Input(
            description="Vertical tilt. -1=bird's-eye, 0=level, 1=worm's-eye.",
            default=0,
            ge=-1,
            le=1,
        ),
        use_wide_angle: bool = Input(
            description="Apply wide-angle lens effect.",
            default=False,
        ),
        prompt: str = Input(
            description="Optional extra styling prompt appended after camera directive.",
            default="",
        ),
        seed: int = Input(
            description="Random seed. Set to 0 for random.",
            default=0,
            ge=0,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps.",
            default=4,
            ge=1,
            le=50,
        ),
    ) -> Path:
        """Run camera control image editing."""

        # Load input image
        input_image = Image.open(str(image)).convert("RGB")

        # Build bilingual camera prompt
        camera_prompt = build_camera_prompt(
            rotate_degrees, move_forward, vertical_tilt, use_wide_angle
        )

        if camera_prompt == "no camera movement" and not prompt:
            # Just return the original image if nothing to do
            camera_prompt = "keep the same camera angle"

        if prompt:
            full_prompt = (
                f"{camera_prompt} {prompt}"
                if camera_prompt != "no camera movement"
                else prompt
            )
        else:
            full_prompt = camera_prompt

        print(f"[predict] Prompt: {full_prompt}")

        # Seed
        if seed == 0:
            seed = random.randint(1, MAX_SEED)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Run inference
        result = self.pipe(
            image=[input_image],
            prompt=full_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=1.0,
            num_images_per_prompt=1,
        ).images[0]

        # Save output
        output_path = Path(tempfile.mktemp(suffix=".webp"))
        result.save(str(output_path), format="WEBP", quality=95)
        return output_path
