from __future__ import annotations

"""
Shared constants for ComfyUI-based Nodetool nodes.

This module centralises recommended model lists so that node modules
(`text_to_image`, `image_to_image`, etc.) stay focused on workflow
logic rather than asset declarations.
"""

from nodetool.metadata.types import (
    HFCLIP,
    HFVAE,
    HFControlNet,
    HFTextToImage,
    HFUnet,
)
from nodetool.nodes.huggingface.stable_diffusion_base import (
    HF_CONTROLNET_MODELS as _HF_CONTROLNET_MODELS,
    HF_STABLE_DIFFUSION_MODELS as _HF_STABLE_DIFFUSION_MODELS,
    HF_STABLE_DIFFUSION_XL_MODELS as _HF_STABLE_DIFFUSION_XL_MODELS,
    HF_CONTROLNET_XL_MODELS as _HF_CONTROLNET_XL_MODELS,
)


# Stable Diffusion text-to-image base models
HF_STABLE_DIFFUSION_MODELS: list[HFTextToImage] = _HF_STABLE_DIFFUSION_MODELS
HF_STABLE_DIFFUSION_XL_MODELS: list[HFTextToImage] = _HF_STABLE_DIFFUSION_XL_MODELS

# ControlNet models
HF_CONTROLNET_MODELS: list[HFControlNet] = _HF_CONTROLNET_MODELS
HF_CONTROLNET_XL_MODELS: list[HFControlNet] = _HF_CONTROLNET_XL_MODELS


FLUX_DEV_MODELS: list[HFTextToImage] = [
    HFTextToImage(
        repo_id="Comfy-Org/flux1-dev",
        path="flux1-dev-fp8.safetensors",
    ),
    HFTextToImage(
        repo_id="Comfy-Org/flux1-dev",
        path="flux1-dev.safetensors",
    ),
]

FLUX_SCHNELL_MODELS: list[HFTextToImage] = [
    HFTextToImage(
        repo_id="Comfy-Org/flux1-schnell",
        path="flux1-schnell.safetensors",
    ),
    HFTextToImage(
        repo_id="Comfy-Org/flux1-schnell",
        path="flux1-schnell-fp8.safetensors",
    ),
]

FLUX_VAE = HFVAE(
    repo_id="ffxvs/vae-flux",
    path="ae.safetensors",
)

FLUX_CLIP_L = HFCLIP(
    repo_id="Comfy-Org/stable-diffusion-3.5-fp8",
    path="text_encoders/clip_l.safetensors",
)

FLUX_CLIP_T5XXL = HFCLIP(
    repo_id="Comfy-Org/stable-diffusion-3.5-fp8",
    path="text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors",
)
