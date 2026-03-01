# Qwen Camera Control — Replicate Model

Replicates the [HF Space](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles) for camera-aware image editing using the exact same models and settings.

## Models

| Component | Source | Note |
|-----------|--------|------|
| Base pipeline | `Qwen/Qwen-Image-Edit-2509` | VAE, tokenizer, text encoder |
| Transformer | `linoyts/Qwen-Image-Edit-Rapid-AIO` | Lightning-distilled, 4-step |
| LoRA | `dx8152/Qwen-Edit-2509-Multiple-angles` | Fused at **scale 1.25** |

## Deploy to Replicate

### 1. Install Cog

```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog
```

### 2. Create a model on Replicate

Go to [replicate.com/create](https://replicate.com/create) and create a new model (e.g. `realimposter/qwen-camera-control`).

### 3. Push

```bash
cog push r8.im/realimposter/qwen-camera-control
```

Replicate builds the image on their servers (~15-20 min for first push). No local GPU needed.

## API

### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | file | **required** | Input image |
| `rotate_degrees` | float | 0 | Positive=left, negative=right (±90°) |
| `move_forward` | float | 0 | 0=none, 5=forward, 10=close-up |
| `vertical_tilt` | float | 0 | -1=bird's-eye, 0=level, 1=worm's-eye |
| `use_wide_angle` | bool | false | Wide-angle lens effect |
| `prompt` | string | "" | Extra styling prompt |
| `seed` | int | 0 | RNG seed (0=random) |
| `num_inference_steps` | int | 4 | Denoising steps |

### Output

Returns a URL to the output image (WebP format).
