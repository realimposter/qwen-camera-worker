# Qwen Camera Control — RunPod Serverless Worker

Replicates the [HF Space](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles) for camera-aware image editing using the exact same models and settings.

## Models

| Component | Source | Note |
|-----------|--------|------|
| Base pipeline | `Qwen/Qwen-Image-Edit-2509` | VAE, tokenizer, text encoder |
| Transformer | `linoyts/Qwen-Image-Edit-Rapid-AIO` | Lightning-distilled, 4-step |
| LoRA | `dx8152/Qwen-Edit-2509-Multiple-angles` | Fused at **scale 1.25** |

## How It Works

- **Docker image** includes all model weights (~30GB) baked in at build time
- **GitHub Action** auto-builds and pushes to `ghcr.io` on every push to `main`
- **Cold starts** load models from local disk in ~30s (no network download)

## Deploy

### 1. Push to GitHub

The GitHub Action auto-builds and pushes to:
```
ghcr.io/realimposter/qwen-camera-worker:latest
```

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Console → Serverless](https://www.runpod.io/console/serverless)
2. **New Endpoint** → Image: `ghcr.io/realimposter/qwen-camera-worker:latest`
3. **GPU**: A6000 or L40S (48GB VRAM)
4. **Active Workers**: 0, **Max Workers**: 1+
5. **Container Disk**: 50GB

### 3. Test

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/portrait.jpg",
      "rotate_degrees": 45
    }
  }'
```

## API

### Input

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | string | **required** | URL or base64 image |
| `rotate_degrees` | float | 0 | Positive=left, negative=right (±90°) |
| `move_forward` | float | 0 | 0=none, 5=forward, 10=close-up |
| `vertical_tilt` | float | 0 | -1=bird's-eye, 0=level, 1=worm's-eye |
| `use_wide_angle` | bool | false | Wide-angle lens effect |
| `prompt` | string | "" | Extra styling prompt |
| `seed` | int | random | RNG seed |
| `num_inference_steps` | int | 4 | Denoising steps |

### Output

```json
{
  "image_base64": "...",
  "seed": 42,
  "prompt": "将镜头向左旋转45度 Rotate the camera 45 degrees to the left.",
  "format": "webp"
}
```
