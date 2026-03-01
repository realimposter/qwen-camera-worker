"""
Download model weights to HF cache at Docker build time.
Uses snapshot_download() to ONLY download files to disk — does NOT load
models into memory, so this works within GitHub Actions' 7GB RAM limit.
"""

from huggingface_hub import snapshot_download

print("[build] Downloading base pipeline: Qwen/Qwen-Image-Edit-2509 ...")
snapshot_download("Qwen/Qwen-Image-Edit-2509")

print("[build] Downloading transformer: linoyts/Qwen-Image-Edit-Rapid-AIO ...")
snapshot_download("linoyts/Qwen-Image-Edit-Rapid-AIO")

print("[build] Downloading LoRA: dx8152/Qwen-Edit-2509-Multiple-angles ...")
snapshot_download("dx8152/Qwen-Edit-2509-Multiple-angles")

print("[build] All models downloaded to HF cache. Done!")
