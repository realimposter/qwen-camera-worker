# Qwen Camera Control

Camera-aware image editing with precise angle control. Rotate, zoom, tilt, and apply wide-angle effects to any image using 4-step Lightning inference.

![Camera Control Examples](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles/resolve/main/qwen_angles.png)

## What It Does

This model takes an input image and applies camera movements — rotation, zoom, tilt, or wide-angle lens effects — while preserving the scene content. It uses bilingual (Chinese + English) prompts matching the LoRA's training data for maximum accuracy.

### Supported Camera Movements

| Movement | Parameter | Range | Example |
|----------|-----------|-------|---------|
| **Rotation** | `rotate_degrees` | -90° to +90° | `45` = rotate 45° left |
| **Zoom/Dolly** | `move_forward` | 0 to 10 | `5` = move forward, `10` = close-up |
| **Vertical Tilt** | `vertical_tilt` | -1 to 1 | `-1` = bird's-eye, `1` = worm's-eye |
| **Wide-Angle** | `use_wide_angle` | true/false | Adds wide-angle lens distortion |

## Model Architecture

| Component | Source | Purpose |
|-----------|--------|---------|
| Base Pipeline | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | VAE, tokenizer, Qwen2.5-VL text encoder |
| Transformer | [linoyts/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/linoyts/Qwen-Image-Edit-Rapid-AIO) | Lightning-distilled for 4-step inference |
| LoRA | [dx8152/Qwen-Edit-2509-Multiple-angles](https://huggingface.co/dx8152/Qwen-Edit-2509-Multiple-angles) | Multi-angle camera control, fused at **1.25x scale** |

This is an exact replica of the [linoyts/Qwen-Image-Edit-Angles](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles) HuggingFace Space, packaged for API use.

## API Usage

```python
import replicate

output = replicate.run(
    "realimposter/qwen-camera-control",
    input={
        "image": "https://example.com/photo.jpg",
        "rotate_degrees": 45,
        "move_forward": 0,
        "vertical_tilt": 0,
        "use_wide_angle": False,
        "num_inference_steps": 4,
    }
)
print(output)  # URL to the result image
```

```bash
curl -s -X POST \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"version":"<VERSION_ID>","input":{"image":"https://example.com/photo.jpg","rotate_degrees":45}}' \
  https://api.replicate.com/v1/predictions
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | file | **required** | Input image (URL or upload) |
| `rotate_degrees` | float | 0 | Camera rotation. Positive = left, negative = right |
| `move_forward` | float | 0 | Zoom/dolly forward (0–10) |
| `vertical_tilt` | float | 0 | Tilt: -1 = bird's-eye, 0 = level, 1 = worm's-eye |
| `use_wide_angle` | bool | false | Apply wide-angle lens effect |
| `prompt` | string | "" | Optional extra styling prompt |
| `seed` | int | 0 | RNG seed (0 = random) |
| `num_inference_steps` | int | 4 | Denoising steps (4 recommended for Rapid-AIO) |

## Performance

- **Inference**: ~3-5 seconds per image (4 steps on L40S)
- **Cold start**: ~2-3 minutes (model loading + LoRA fusion)
- **Output**: WebP format, 95% quality

## Credits

- [Qwen Team](https://huggingface.co/Qwen) — Base image editing model
- [dx8152](https://huggingface.co/dx8152) — Multi-angle camera LoRA
- [Phr00t](https://huggingface.co/Phr00t) / [linoyts](https://huggingface.co/linoyts) — Rapid-AIO Lightning transformer
- [linoyts](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles) — Original HF Space demo
