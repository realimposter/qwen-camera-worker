# Qwen Camera Control — RunPod Serverless Worker

Replicates the [HF Space](https://huggingface.co/spaces/linoyts/Qwen-Image-Edit-Angles) for camera-aware image editing using the exact same models and settings.

## Models

| Component | Source | Note |
|-----------|--------|------|
| Base pipeline | `Qwen/Qwen-Image-Edit-2509` | VAE, tokenizer, text encoder |
| Transformer | `linoyts/Qwen-Image-Edit-Rapid-AIO` | Lightning-distilled, 4-step |
| LoRA | `dx8152/Qwen-Edit-2509-Multiple-angles` | Fused at **scale 1.25** |

## Architecture

- **Docker image** is lightweight (~5GB) — just code + Python deps
- **Model weights** (~25GB) are downloaded on first cold start and cached on a **RunPod Network Volume**
- **GitHub Action** auto-builds and pushes to `ghcr.io` on every push to `main`

## Deploy

### 1. Push to GitHub

Commit and push. The GitHub Action (`.github/workflows/build-runpod-qwen-camera.yml`) will auto-build and push the image to `ghcr.io/YOUR_ORG/qwen-camera-control:latest`.

### 2. Create a RunPod Network Volume

1. Go to [RunPod Console → Storage](https://www.runpod.io/console/user/storage)
2. Create a **Network Volume** (50GB minimum, same region as your endpoint)
3. Note the volume ID

### 3. Create a RunPod Serverless Endpoint

1. Go to [RunPod Console → Serverless](https://www.runpod.io/console/serverless)
2. **New Endpoint** → Container Image: `ghcr.io/YOUR_ORG/qwen-camera-control:latest`
3. **GPU**: A6000 (48GB) or L40S
4. **Network Volume**: Attach the volume you created
5. **Workers**: Active = 0, Max = 1+
6. **Container Disk**: 20GB (for pip packages)

> ⚠️ First cold start will take 5-10 min while models download to the volume. Subsequent starts are fast (~30s).

### 4. Test

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/portrait.jpg",
      "rotate_degrees": 45,
      "move_forward": 0,
      "vertical_tilt": 0,
      "use_wide_angle": false
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
| `output_format` | string | "webp" | "webp", "png", or "jpeg" |

### Output

```json
{
  "image_base64": "...",
  "seed": 42,
  "prompt": "将镜头向左旋转45度 Rotate the camera 45 degrees to the left.",
  "format": "webp"
}
```
