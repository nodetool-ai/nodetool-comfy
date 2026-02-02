"""
ComfyUI Type Wrappers

This module provides wrapper types for ComfyUI's internal types (MODEL, CLIP, VAE, etc.)
These wrappers allow ComfyUI objects to be passed between nodes in the nodetool system
while maintaining type safety.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ComfyUIType(BaseModel):
    """Base class for ComfyUI type wrappers."""
    
    # Store the actual ComfyUI object
    _value: Any = None
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, value: Any = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_value', value)
    
    @property
    def value(self) -> Any:
        """Get the wrapped ComfyUI object."""
        return self._value
    
    def __bool__(self) -> bool:
        """Check if the wrapper contains a value."""
        return self._value is not None


class Model(ComfyUIType):
    """
    Wrapper for ComfyUI MODEL type (model patcher).
    Represents a diffusion model used for denoising.
    """
    pass


class Clip(ComfyUIType):
    """
    Wrapper for ComfyUI CLIP type.
    Represents a CLIP text encoder model.
    """
    pass


class Vae(ComfyUIType):
    """
    Wrapper for ComfyUI VAE type.
    Represents a Variational Autoencoder for encoding/decoding images.
    """
    pass


class Conditioning(ComfyUIType):
    """
    Wrapper for ComfyUI CONDITIONING type.
    Represents conditioning tensors (text embeddings) for guiding generation.
    """
    pass


class Latent(ComfyUIType):
    """
    Wrapper for ComfyUI LATENT type.
    Represents latent image tensors (dict with 'samples' key).
    """
    pass


class Mask(ComfyUIType):
    """
    Wrapper for ComfyUI MASK type.
    Represents mask tensors for inpainting.
    """
    pass


class ControlNet(ComfyUIType):
    """
    Wrapper for ComfyUI CONTROL_NET type.
    Represents a ControlNet model for guided generation.
    """
    pass


class StyleModel(ComfyUIType):
    """
    Wrapper for ComfyUI STYLE_MODEL type.
    Represents a style transfer model.
    """
    pass


class Gligen(ComfyUIType):
    """
    Wrapper for ComfyUI GLIGEN type.
    Represents a GLIGEN model for layout-guided generation.
    """
    pass


class UpscaleModel(ComfyUIType):
    """
    Wrapper for ComfyUI UPSCALE_MODEL type.
    Represents an upscaling model.
    """
    pass


class Sampler(ComfyUIType):
    """
    Wrapper for ComfyUI SAMPLER type.
    Represents a sampler configuration.
    """
    pass


class Sigmas(ComfyUIType):
    """
    Wrapper for ComfyUI SIGMAS type.
    Represents sigma schedule for sampling.
    """
    pass


class Noise(ComfyUIType):
    """
    Wrapper for ComfyUI NOISE type.
    Represents noise configuration.
    """
    pass


class Guider(ComfyUIType):
    """
    Wrapper for ComfyUI GUIDER type.
    Represents a sampling guider.
    """
    pass


class Audio(ComfyUIType):
    """
    Wrapper for ComfyUI AUDIO type.
    Represents audio tensors.
    """
    pass


# Type mapping for code generation
COMFY_TYPE_MAP = {
    "MODEL": Model,
    "CLIP": Clip,
    "VAE": Vae,
    "CONDITIONING": Conditioning,
    "LATENT": Latent,
    "MASK": Mask,
    "CONTROL_NET": ControlNet,
    "STYLE_MODEL": StyleModel,
    "GLIGEN": Gligen,
    "UPSCALE_MODEL": UpscaleModel,
    "SAMPLER": Sampler,
    "SIGMAS": Sigmas,
    "NOISE": Noise,
    "GUIDER": Guider,
    "AUDIO": Audio,
}


__all__ = [
    "ComfyUIType",
    "Model",
    "Clip",
    "Vae",
    "Conditioning",
    "Latent",
    "Mask",
    "ControlNet",
    "StyleModel",
    "Gligen",
    "UpscaleModel",
    "Sampler",
    "Sigmas",
    "Noise",
    "Guider",
    "Audio",
    "COMFY_TYPE_MAP",
]
