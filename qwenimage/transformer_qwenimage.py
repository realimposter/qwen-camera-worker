"""
Re-export QwenImageTransformer2DModel from diffusers.

The HF Space vendors its own copy, but the latest diffusers (installed
from git) includes it natively. We re-export here so handler.py's
imports match the HF Space's import paths.
"""

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

__all__ = ["QwenImageTransformer2DModel"]
