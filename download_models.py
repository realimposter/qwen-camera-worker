"""
Download model weights to local directories at Docker build time.
Uses snapshot_download() to ONLY download files to disk — does NOT load
models into memory, so this works within GitHub Actions' 7GB RAM limit.
Clears HF cache to save disk space.
"""

import os
import shutil
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
CACHE_DIR = "/src/hf_cache"

print("[build] Downloading base pipeline: Qwen/Qwen-Image-Edit-2509 ...")
snapshot_download("Qwen/Qwen-Image-Edit-2509", cache_dir=CACHE_DIR, local_dir="/src/weights/base", local_dir_use_symlinks=False, ignore_patterns=["transformer/*"])

print("[build] Downloading transformer: linoyts/Qwen-Image-Edit-Rapid-AIO ...")
snapshot_download("linoyts/Qwen-Image-Edit-Rapid-AIO", cache_dir=CACHE_DIR, local_dir="/src/weights/transformer", local_dir_use_symlinks=False, ignore_patterns=["*transformer_weights.safetensors"])

print("[build] Downloading LoRA: dx8152/Qwen-Edit-2509-Multiple-angles ...")
snapshot_download("dx8152/Qwen-Edit-2509-Multiple-angles", cache_dir=CACHE_DIR, local_dir="/src/weights/lora", local_dir_use_symlinks=False)

print("[build] Cleaning up HF cache to save disk space...")
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

print("[build] All models downloaded to local dir. Done!")
