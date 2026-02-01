"""
Auto-generated ComfyUI nodes for Nodetool

This file was generated from comfy_nodes_metadata.json
DO NOT EDIT MANUALLY - regenerate using scripts/generate_comfy_nodes.py
"""

from __future__ import annotations

from typing import Any, Optional

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


class KSampler(BaseNode):
    """
    Uses the provided model, positive and negative conditioning to denoise the latent image.
    
    Category: sampling
    ComfyUI Node ID: KSampler
    """

    model: Any = Field(default=None, description="The model used for denoising the input latent.")
    seed: int = Field(default=0, description="The random seed used for creating the noise.", ge=0, le=18446744073709551615)
    steps: int = Field(default=20, description="The number of steps used in the denoising process.", ge=1, le=10000)
    cfg: float = Field(default=8.0, description="The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.", ge=0.0, le=100.0)
    sampler_name: str = Field(default=None, description="The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.")
    scheduler: str = Field(default=None, description="The scheduler controls how noise is gradually removed to form the image.")
    positive: Any = Field(default=None, description="The conditioning describing the attributes you want to include in the image.")
    negative: Any = Field(default=None, description="The conditioning describing the attributes you want to exclude from the image.")
    latent_image: Any = Field(default=None, description="The latent image to denoise.")
    denoise: float = Field(default=1.0, description="The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.", ge=0.0, le=1.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the KSampler node."""
        # Import the ComfyUI node class
        from nodes import KSampler

        # Create node instance
        node = KSampler()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["seed"] = self.seed
        kwargs["steps"] = self.steps
        kwargs["cfg"] = self.cfg
        kwargs["sampler_name"] = self.sampler_name
        kwargs["scheduler"] = self.scheduler
        kwargs["positive"] = self.positive
        kwargs["negative"] = self.negative
        kwargs["latent_image"] = self.latent_image
        kwargs["denoise"] = self.denoise

        # Call the node function
        result = node.sample(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class CheckpointLoaderSimple(BaseNode):
    """
    Loads a diffusion model checkpoint, diffusion models are used to denoise latents.
    
    Category: loaders
    ComfyUI Node ID: CheckpointLoaderSimple
    """

    ckpt_name: Any = Field(default=None, description="The name of the checkpoint (model) to load.")

    async def process(self, context: ProcessingContext) -> tuple[Any, Any, Any]:
        """Process the CheckpointLoaderSimple node."""
        # Import the ComfyUI node class
        from nodes import CheckpointLoaderSimple

        # Create node instance
        node = CheckpointLoaderSimple()

        # Prepare inputs
        kwargs = {}
        kwargs["ckpt_name"] = self.ckpt_name

        # Call the node function
        result = node.load_checkpoint(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class CLIPTextEncode(BaseNode):
    """
    Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images.
    
    Category: conditioning
    ComfyUI Node ID: CLIPTextEncode
    """

    text: Any = Field(default=None, description="The text to be encoded.")
    clip: Any = Field(default=None, description="The CLIP model used for encoding the text.")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncode node."""
        # Import the ComfyUI node class
        from nodes import CLIPTextEncode

        # Create node instance
        node = CLIPTextEncode()

        # Prepare inputs
        kwargs = {}
        kwargs["text"] = self.text
        kwargs["clip"] = self.clip

        # Call the node function
        result = node.encode(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class CLIPSetLastLayer(BaseNode):
    """CLIPSetLastLayer node from ComfyUI (category: conditioning)"""

    clip: Any = Field(default=None, description="clip parameter")
    stop_at_clip_layer: int = Field(default=0, description="stop_at_clip_layer parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPSetLastLayer node."""
        # Import the ComfyUI node class
        from nodes import CLIPSetLastLayer

        # Create node instance
        node = CLIPSetLastLayer()

        # Prepare inputs
        kwargs = {}
        kwargs["clip"] = self.clip
        kwargs["stop_at_clip_layer"] = self.stop_at_clip_layer

        # Call the node function
        result = node.set_last_layer(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class VAEDecode(BaseNode):
    """
    Decodes latent images back into pixel space images.
    
    Category: latent
    ComfyUI Node ID: VAEDecode
    """

    samples: Any = Field(default=None, description="The latent to be decoded.")
    vae: Any = Field(default=None, description="The VAE model used for decoding the latent.")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAEDecode node."""
        # Import the ComfyUI node class
        from nodes import VAEDecode

        # Create node instance
        node = VAEDecode()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["vae"] = self.vae

        # Call the node function
        result = node.decode(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class VAEEncode(BaseNode):
    """VAEEncode node from ComfyUI (category: latent)"""

    pixels: Any = Field(default=None, description="pixels parameter")
    vae: Any = Field(default=None, description="vae parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAEEncode node."""
        # Import the ComfyUI node class
        from nodes import VAEEncode

        # Create node instance
        node = VAEEncode()

        # Prepare inputs
        kwargs = {}
        kwargs["pixels"] = self.pixels
        kwargs["vae"] = self.vae

        # Call the node function
        result = node.encode(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class VAEEncodeForInpaint(BaseNode):
    """VAEEncodeForInpaint node from ComfyUI (category: latent/inpaint)"""

    pixels: Any = Field(default=None, description="pixels parameter")
    vae: Any = Field(default=None, description="vae parameter")
    mask: Any = Field(default=None, description="mask parameter")
    grow_mask_by: int = Field(default=6, description="grow_mask_by parameter", ge=0, le=64)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAEEncodeForInpaint node."""
        # Import the ComfyUI node class
        from nodes import VAEEncodeForInpaint

        # Create node instance
        node = VAEEncodeForInpaint()

        # Prepare inputs
        kwargs = {}
        kwargs["pixels"] = self.pixels
        kwargs["vae"] = self.vae
        kwargs["mask"] = self.mask
        kwargs["grow_mask_by"] = self.grow_mask_by

        # Call the node function
        result = node.encode(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class VAELoader(BaseNode):
    """VAELoader node from ComfyUI (category: loaders)"""

    vae_name: Any = Field(default=None, description="vae_name parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAELoader node."""
        # Import the ComfyUI node class
        from nodes import VAELoader

        # Create node instance
        node = VAELoader()

        # Prepare inputs
        kwargs = {}
        kwargs["vae_name"] = self.vae_name

        # Call the node function
        result = node.load_vae(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class EmptyLatentImage(BaseNode):
    """
    Create a new batch of empty latent images to be denoised via sampling.
    
    Category: latent
    ComfyUI Node ID: EmptyLatentImage
    """

    width: int = Field(default=512, description="The width of the latent images in pixels.", ge=16, le={'_ref': 'MAX_RESOLUTION'})
    height: int = Field(default=512, description="The height of the latent images in pixels.", ge=16, le={'_ref': 'MAX_RESOLUTION'})
    batch_size: int = Field(default=1, description="The number of latent images in the batch.", ge=1, le=4096)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyLatentImage node."""
        # Import the ComfyUI node class
        from nodes import EmptyLatentImage

        # Create node instance
        node = EmptyLatentImage()

        # Prepare inputs
        kwargs = {}
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["batch_size"] = self.batch_size

        # Call the node function
        result = node.generate(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LatentUpscale(BaseNode):
    """LatentUpscale node from ComfyUI (category: latent)"""

    samples: Any = Field(default=None, description="samples parameter")
    upscale_method: Any = Field(default=None, description="upscale_method parameter")
    width: int = Field(default=512, description="width parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    height: int = Field(default=512, description="height parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    crop: Any = Field(default=None, description="crop parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentUpscale node."""
        # Import the ComfyUI node class
        from nodes import LatentUpscale

        # Create node instance
        node = LatentUpscale()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["upscale_method"] = self.upscale_method
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["crop"] = self.crop

        # Call the node function
        result = node.upscale(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LatentUpscaleBy(BaseNode):
    """LatentUpscaleBy node from ComfyUI (category: latent)"""

    samples: Any = Field(default=None, description="samples parameter")
    upscale_method: Any = Field(default=None, description="upscale_method parameter")
    scale_by: float = Field(default=1.5, description="scale_by parameter", ge=0.01, le=8.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentUpscaleBy node."""
        # Import the ComfyUI node class
        from nodes import LatentUpscaleBy

        # Create node instance
        node = LatentUpscaleBy()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["upscale_method"] = self.upscale_method
        kwargs["scale_by"] = self.scale_by

        # Call the node function
        result = node.upscale(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LatentFromBatch(BaseNode):
    """LatentFromBatch node from ComfyUI (category: latent/batch)"""

    samples: Any = Field(default=None, description="samples parameter")
    batch_index: int = Field(default=0, description="batch_index parameter", ge=0, le=63)
    length: int = Field(default=1, description="length parameter", ge=1, le=64)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentFromBatch node."""
        # Import the ComfyUI node class
        from nodes import LatentFromBatch

        # Create node instance
        node = LatentFromBatch()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["batch_index"] = self.batch_index
        kwargs["length"] = self.length

        # Call the node function
        result = node.frombatch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class RepeatLatentBatch(BaseNode):
    """RepeatLatentBatch node from ComfyUI (category: latent/batch)"""

    samples: Any = Field(default=None, description="samples parameter")
    amount: int = Field(default=1, description="amount parameter", ge=1, le=64)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the RepeatLatentBatch node."""
        # Import the ComfyUI node class
        from nodes import RepeatLatentBatch

        # Create node instance
        node = RepeatLatentBatch()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["amount"] = self.amount

        # Call the node function
        result = node.repeat(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class SaveImage(BaseNode):
    """
    Saves the input images to your ComfyUI output directory.
    
    Category: image
    ComfyUI Node ID: SaveImage
    """

    images: Any = Field(default=None, description="The images to save.")
    filename_prefix: str = Field(default="ComfyUI", description="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveImage node."""
        # Import the ComfyUI node class
        from nodes import SaveImage

        # Create node instance
        node = SaveImage()

        # Prepare inputs
        kwargs = {}
        kwargs["images"] = self.images
        kwargs["filename_prefix"] = self.filename_prefix

        # Call the node function
        result = node.save_images(**kwargs)

        # Return result
        return result


class PreviewImage(BaseNode):
    """PreviewImage node from ComfyUI (category: uncategorized)"""

    images: Any = Field(default=None, description="images parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the PreviewImage node."""
        # Import the ComfyUI node class
        from nodes import PreviewImage

        # Create node instance
        node = PreviewImage()

        # Prepare inputs
        kwargs = {}
        kwargs["images"] = self.images

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class LoadImage(BaseNode):
    """LoadImage node from ComfyUI (category: image)"""

    image: Any = Field(default=None, description="image parameter")

    async def process(self, context: ProcessingContext) -> tuple[Any, Any]:
        """Process the LoadImage node."""
        # Import the ComfyUI node class
        from nodes import LoadImage

        # Create node instance
        node = LoadImage()

        # Prepare inputs
        kwargs = {}
        kwargs["image"] = self.image

        # Call the node function
        result = node.load_image(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class LoadImageMask(BaseNode):
    """LoadImageMask node from ComfyUI (category: mask)"""

    image: Any = Field(default=None, description="image parameter")
    channel: Any = Field(default=None, description="channel parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoadImageMask node."""
        # Import the ComfyUI node class
        from nodes import LoadImageMask

        # Create node instance
        node = LoadImageMask()

        # Prepare inputs
        kwargs = {}
        kwargs["image"] = self.image
        kwargs["channel"] = self.channel

        # Call the node function
        result = node.load_image(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LoadImageOutput(BaseNode):
    """
    Load an image from the output folder. When the refresh button is clicked, the node will update the image list and automatically select the first image, allowing for easy iteration.
    
    Category: uncategorized
    ComfyUI Node ID: LoadImageOutput
    """

    image: str = Field(default=None, description="image parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoadImageOutput node."""
        # Import the ComfyUI node class
        from nodes import LoadImageOutput

        # Create node instance
        node = LoadImageOutput()

        # Prepare inputs
        kwargs = {}
        kwargs["image"] = self.image

        # Call the node function
        result = node.load_image(**kwargs)

        # Return result
        return result


class ImageScale(BaseNode):
    """ImageScale node from ComfyUI (category: image/upscaling)"""

    image: Any = Field(default=None, description="image parameter")
    upscale_method: Any = Field(default=None, description="upscale_method parameter")
    width: int = Field(default=512, description="width parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    height: int = Field(default=512, description="height parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    crop: Any = Field(default=None, description="crop parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageScale node."""
        # Import the ComfyUI node class
        from nodes import ImageScale

        # Create node instance
        node = ImageScale()

        # Prepare inputs
        kwargs = {}
        kwargs["image"] = self.image
        kwargs["upscale_method"] = self.upscale_method
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["crop"] = self.crop

        # Call the node function
        result = node.upscale(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ImageScaleBy(BaseNode):
    """ImageScaleBy node from ComfyUI (category: image/upscaling)"""

    image: Any = Field(default=None, description="image parameter")
    upscale_method: Any = Field(default=None, description="upscale_method parameter")
    scale_by: float = Field(default=1.0, description="scale_by parameter", ge=0.01, le=8.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageScaleBy node."""
        # Import the ComfyUI node class
        from nodes import ImageScaleBy

        # Create node instance
        node = ImageScaleBy()

        # Prepare inputs
        kwargs = {}
        kwargs["image"] = self.image
        kwargs["upscale_method"] = self.upscale_method
        kwargs["scale_by"] = self.scale_by

        # Call the node function
        result = node.upscale(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ImageInvert(BaseNode):
    """ImageInvert node from ComfyUI (category: image)"""

    image: Any = Field(default=None, description="image parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageInvert node."""
        # Import the ComfyUI node class
        from nodes import ImageInvert

        # Create node instance
        node = ImageInvert()

        # Prepare inputs
        kwargs = {}
        kwargs["image"] = self.image

        # Call the node function
        result = node.invert(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ImagePadForOutpaint(BaseNode):
    """ImagePadForOutpaint node from ComfyUI (category: image)"""

    image: Any = Field(default=None, description="image parameter")
    left: int = Field(default=0, description="left parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    top: int = Field(default=0, description="top parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    right: int = Field(default=0, description="right parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    bottom: int = Field(default=0, description="bottom parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    feathering: int = Field(default=40, description="feathering parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})

    async def process(self, context: ProcessingContext) -> tuple[Any, Any]:
        """Process the ImagePadForOutpaint node."""
        # Import the ComfyUI node class
        from nodes import ImagePadForOutpaint

        # Create node instance
        node = ImagePadForOutpaint()

        # Prepare inputs
        kwargs = {}
        kwargs["image"] = self.image
        kwargs["left"] = self.left
        kwargs["top"] = self.top
        kwargs["right"] = self.right
        kwargs["bottom"] = self.bottom
        kwargs["feathering"] = self.feathering

        # Call the node function
        result = node.expand_image(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class EmptyImage(BaseNode):
    """EmptyImage node from ComfyUI (category: image)"""

    width: int = Field(default=512, description="width parameter", ge=1, le={'_ref': 'MAX_RESOLUTION'})
    height: int = Field(default=512, description="height parameter", ge=1, le={'_ref': 'MAX_RESOLUTION'})
    batch_size: int = Field(default=1, description="batch_size parameter", ge=1, le=4096)
    color: int = Field(default=0, description="color parameter", ge=0, le=16777215)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyImage node."""
        # Import the ComfyUI node class
        from nodes import EmptyImage

        # Create node instance
        node = EmptyImage()

        # Prepare inputs
        kwargs = {}
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["batch_size"] = self.batch_size
        kwargs["color"] = self.color

        # Call the node function
        result = node.generate(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ConditioningAverage(BaseNode):
    """ConditioningAverage node from ComfyUI (category: conditioning)"""

    conditioning_to: Any = Field(default=None, description="conditioning_to parameter")
    conditioning_from: Any = Field(default=None, description="conditioning_from parameter")
    conditioning_to_strength: float = Field(default=1.0, description="conditioning_to_strength parameter", ge=0.0, le=1.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningAverage node."""
        # Import the ComfyUI node class
        from nodes import ConditioningAverage

        # Create node instance
        node = ConditioningAverage()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning_to"] = self.conditioning_to
        kwargs["conditioning_from"] = self.conditioning_from
        kwargs["conditioning_to_strength"] = self.conditioning_to_strength

        # Call the node function
        result = node.addWeighted(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ConditioningCombine(BaseNode):
    """ConditioningCombine node from ComfyUI (category: conditioning)"""

    conditioning_1: Any = Field(default=None, description="conditioning_1 parameter")
    conditioning_2: Any = Field(default=None, description="conditioning_2 parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningCombine node."""
        # Import the ComfyUI node class
        from nodes import ConditioningCombine

        # Create node instance
        node = ConditioningCombine()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning_1"] = self.conditioning_1
        kwargs["conditioning_2"] = self.conditioning_2

        # Call the node function
        result = node.combine(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ConditioningConcat(BaseNode):
    """ConditioningConcat node from ComfyUI (category: conditioning)"""

    conditioning_to: Any = Field(default=None, description="conditioning_to parameter")
    conditioning_from: Any = Field(default=None, description="conditioning_from parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningConcat node."""
        # Import the ComfyUI node class
        from nodes import ConditioningConcat

        # Create node instance
        node = ConditioningConcat()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning_to"] = self.conditioning_to
        kwargs["conditioning_from"] = self.conditioning_from

        # Call the node function
        result = node.concat(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ConditioningSetArea(BaseNode):
    """ConditioningSetArea node from ComfyUI (category: conditioning)"""

    conditioning: Any = Field(default=None, description="conditioning parameter")
    width: int = Field(default=64, description="width parameter", ge=64, le={'_ref': 'MAX_RESOLUTION'})
    height: int = Field(default=64, description="height parameter", ge=64, le={'_ref': 'MAX_RESOLUTION'})
    x: int = Field(default=0, description="x parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    y: int = Field(default=0, description="y parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    strength: float = Field(default=1.0, description="strength parameter", ge=0.0, le=10.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningSetArea node."""
        # Import the ComfyUI node class
        from nodes import ConditioningSetArea

        # Create node instance
        node = ConditioningSetArea()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning"] = self.conditioning
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["x"] = self.x
        kwargs["y"] = self.y
        kwargs["strength"] = self.strength

        # Call the node function
        result = node.append(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ConditioningSetAreaPercentage(BaseNode):
    """ConditioningSetAreaPercentage node from ComfyUI (category: conditioning)"""

    conditioning: Any = Field(default=None, description="conditioning parameter")
    width: float = Field(default=1.0, description="width parameter", ge=0, le=1.0)
    height: float = Field(default=1.0, description="height parameter", ge=0, le=1.0)
    x: float = Field(default=0, description="x parameter", ge=0, le=1.0)
    y: float = Field(default=0, description="y parameter", ge=0, le=1.0)
    strength: float = Field(default=1.0, description="strength parameter", ge=0.0, le=10.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningSetAreaPercentage node."""
        # Import the ComfyUI node class
        from nodes import ConditioningSetAreaPercentage

        # Create node instance
        node = ConditioningSetAreaPercentage()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning"] = self.conditioning
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["x"] = self.x
        kwargs["y"] = self.y
        kwargs["strength"] = self.strength

        # Call the node function
        result = node.append(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ConditioningSetAreaStrength(BaseNode):
    """ConditioningSetAreaStrength node from ComfyUI (category: conditioning)"""

    conditioning: Any = Field(default=None, description="conditioning parameter")
    strength: float = Field(default=1.0, description="strength parameter", ge=0.0, le=10.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningSetAreaStrength node."""
        # Import the ComfyUI node class
        from nodes import ConditioningSetAreaStrength

        # Create node instance
        node = ConditioningSetAreaStrength()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning"] = self.conditioning
        kwargs["strength"] = self.strength

        # Call the node function
        result = node.append(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ConditioningSetMask(BaseNode):
    """ConditioningSetMask node from ComfyUI (category: conditioning)"""

    conditioning: Any = Field(default=None, description="conditioning parameter")
    mask: Any = Field(default=None, description="mask parameter")
    strength: float = Field(default=1.0, description="strength parameter", ge=0.0, le=10.0)
    set_cond_area: str = Field(default=None, description="set_cond_area parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningSetMask node."""
        # Import the ComfyUI node class
        from nodes import ConditioningSetMask

        # Create node instance
        node = ConditioningSetMask()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning"] = self.conditioning
        kwargs["mask"] = self.mask
        kwargs["strength"] = self.strength
        kwargs["set_cond_area"] = self.set_cond_area

        # Call the node function
        result = node.append(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class KSamplerAdvanced(BaseNode):
    """KSamplerAdvanced node from ComfyUI (category: sampling)"""

    model: Any = Field(default=None, description="model parameter")
    add_noise: str = Field(default=None, description="add_noise parameter")
    noise_seed: int = Field(default=0, description="noise_seed parameter", ge=0, le=18446744073709551615)
    steps: int = Field(default=20, description="steps parameter", ge=1, le=10000)
    cfg: float = Field(default=8.0, description="cfg parameter", ge=0.0, le=100.0)
    sampler_name: str = Field(default=None, description="sampler_name parameter")
    scheduler: str = Field(default=None, description="scheduler parameter")
    positive: Any = Field(default=None, description="positive parameter")
    negative: Any = Field(default=None, description="negative parameter")
    latent_image: Any = Field(default=None, description="latent_image parameter")
    start_at_step: int = Field(default=0, description="start_at_step parameter", ge=0, le=10000)
    end_at_step: int = Field(default=10000, description="end_at_step parameter", ge=0, le=10000)
    return_with_leftover_noise: str = Field(default=None, description="return_with_leftover_noise parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the KSamplerAdvanced node."""
        # Import the ComfyUI node class
        from nodes import KSamplerAdvanced

        # Create node instance
        node = KSamplerAdvanced()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["add_noise"] = self.add_noise
        kwargs["noise_seed"] = self.noise_seed
        kwargs["steps"] = self.steps
        kwargs["cfg"] = self.cfg
        kwargs["sampler_name"] = self.sampler_name
        kwargs["scheduler"] = self.scheduler
        kwargs["positive"] = self.positive
        kwargs["negative"] = self.negative
        kwargs["latent_image"] = self.latent_image
        kwargs["start_at_step"] = self.start_at_step
        kwargs["end_at_step"] = self.end_at_step
        kwargs["return_with_leftover_noise"] = self.return_with_leftover_noise

        # Call the node function
        result = node.sample(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class SetLatentNoiseMask(BaseNode):
    """SetLatentNoiseMask node from ComfyUI (category: latent/inpaint)"""

    samples: Any = Field(default=None, description="samples parameter")
    mask: Any = Field(default=None, description="mask parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the SetLatentNoiseMask node."""
        # Import the ComfyUI node class
        from nodes import SetLatentNoiseMask

        # Create node instance
        node = SetLatentNoiseMask()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["mask"] = self.mask

        # Call the node function
        result = node.set_mask(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LatentComposite(BaseNode):
    """LatentComposite node from ComfyUI (category: latent)"""

    samples_to: Any = Field(default=None, description="samples_to parameter")
    samples_from: Any = Field(default=None, description="samples_from parameter")
    x: int = Field(default=0, description="x parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    y: int = Field(default=0, description="y parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    feather: int = Field(default=0, description="feather parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentComposite node."""
        # Import the ComfyUI node class
        from nodes import LatentComposite

        # Create node instance
        node = LatentComposite()

        # Prepare inputs
        kwargs = {}
        kwargs["samples_to"] = self.samples_to
        kwargs["samples_from"] = self.samples_from
        kwargs["x"] = self.x
        kwargs["y"] = self.y
        kwargs["feather"] = self.feather

        # Call the node function
        result = node.composite(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LatentBlend(BaseNode):
    """LatentBlend node from ComfyUI (category: _for_testing)"""

    samples1: Any = Field(default=None, description="samples1 parameter")
    samples2: Any = Field(default=None, description="samples2 parameter")
    blend_factor: float = Field(default=0.5, description="blend_factor parameter", ge=0, le=1)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentBlend node."""
        # Import the ComfyUI node class
        from nodes import LatentBlend

        # Create node instance
        node = LatentBlend()

        # Prepare inputs
        kwargs = {}
        kwargs["samples1"] = self.samples1
        kwargs["samples2"] = self.samples2
        kwargs["blend_factor"] = self.blend_factor

        # Call the node function
        result = node.blend(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LatentRotate(BaseNode):
    """LatentRotate node from ComfyUI (category: latent/transform)"""

    samples: Any = Field(default=None, description="samples parameter")
    rotation: str = Field(default=None, description="rotation parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentRotate node."""
        # Import the ComfyUI node class
        from nodes import LatentRotate

        # Create node instance
        node = LatentRotate()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["rotation"] = self.rotation

        # Call the node function
        result = node.rotate(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LatentFlip(BaseNode):
    """LatentFlip node from ComfyUI (category: latent/transform)"""

    samples: Any = Field(default=None, description="samples parameter")
    flip_method: str = Field(default=None, description="flip_method parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentFlip node."""
        # Import the ComfyUI node class
        from nodes import LatentFlip

        # Create node instance
        node = LatentFlip()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["flip_method"] = self.flip_method

        # Call the node function
        result = node.flip(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LatentCrop(BaseNode):
    """LatentCrop node from ComfyUI (category: latent/transform)"""

    samples: Any = Field(default=None, description="samples parameter")
    width: int = Field(default=512, description="width parameter", ge=64, le={'_ref': 'MAX_RESOLUTION'})
    height: int = Field(default=512, description="height parameter", ge=64, le={'_ref': 'MAX_RESOLUTION'})
    x: int = Field(default=0, description="x parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    y: int = Field(default=0, description="y parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentCrop node."""
        # Import the ComfyUI node class
        from nodes import LatentCrop

        # Create node instance
        node = LatentCrop()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["x"] = self.x
        kwargs["y"] = self.y

        # Call the node function
        result = node.crop(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LoraLoader(BaseNode):
    """
    LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together.
    
    Category: loaders
    ComfyUI Node ID: LoraLoader
    """

    model: Any = Field(default=None, description="The diffusion model the LoRA will be applied to.")
    clip: Any = Field(default=None, description="The CLIP model the LoRA will be applied to.")
    lora_name: Any = Field(default=None, description="The name of the LoRA.")
    strength_model: float = Field(default=1.0, description="How strongly to modify the diffusion model. This value can be negative.", le=100.0)
    strength_clip: float = Field(default=1.0, description="How strongly to modify the CLIP model. This value can be negative.", le=100.0)

    async def process(self, context: ProcessingContext) -> tuple[Any, Any]:
        """Process the LoraLoader node."""
        # Import the ComfyUI node class
        from nodes import LoraLoader

        # Create node instance
        node = LoraLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["clip"] = self.clip
        kwargs["lora_name"] = self.lora_name
        kwargs["strength_model"] = self.strength_model
        kwargs["strength_clip"] = self.strength_clip

        # Call the node function
        result = node.load_lora(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class CLIPLoader(BaseNode):
    """
    [Recipes]

stable_diffusion: clip-l
stable_cascade: clip-g
sd3: t5 xxl/ clip-g / clip-l
stable_audio: t5 base
mochi: t5 xxl
cosmos: old t5 xxl
lumina2: gemma 2 2B
wan: umt5 xxl
 hidream: llama-3.1 (Recommend) or t5
omnigen2: qwen vl 2.5 3B
    
    Category: advanced/loaders
    ComfyUI Node ID: CLIPLoader
    """

    clip_name: Any = Field(default=None, description="clip_name parameter")
    type: str = Field(default=None, description="type parameter")
    device: Optional[str] = Field(default=None, description="device parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPLoader node."""
        # Import the ComfyUI node class
        from nodes import CLIPLoader

        # Create node instance
        node = CLIPLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["clip_name"] = self.clip_name
        kwargs["type"] = self.type
        if self.device is not None:
            kwargs["device"] = self.device

        # Call the node function
        result = node.load_clip(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class UNETLoader(BaseNode):
    """UNETLoader node from ComfyUI (category: advanced/loaders)"""

    unet_name: Any = Field(default=None, description="unet_name parameter")
    weight_dtype: str = Field(default=None, description="weight_dtype parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the UNETLoader node."""
        # Import the ComfyUI node class
        from nodes import UNETLoader

        # Create node instance
        node = UNETLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["unet_name"] = self.unet_name
        kwargs["weight_dtype"] = self.weight_dtype

        # Call the node function
        result = node.load_unet(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class DualCLIPLoader(BaseNode):
    """
    [Recipes]

sdxl: clip-l, clip-g
sd3: clip-l, clip-g / clip-l, t5 / clip-g, t5
flux: clip-l, t5
hidream: at least one of t5 or llama, recommended t5 and llama
hunyuan_image: qwen2.5vl 7b and byt5 small
newbie: gemma-3-4b-it, jina clip v2
    
    Category: advanced/loaders
    ComfyUI Node ID: DualCLIPLoader
    """

    clip_name1: Any = Field(default=None, description="clip_name1 parameter")
    clip_name2: Any = Field(default=None, description="clip_name2 parameter")
    type: str = Field(default=None, description="type parameter")
    device: Optional[str] = Field(default=None, description="device parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the DualCLIPLoader node."""
        # Import the ComfyUI node class
        from nodes import DualCLIPLoader

        # Create node instance
        node = DualCLIPLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["clip_name1"] = self.clip_name1
        kwargs["clip_name2"] = self.clip_name2
        kwargs["type"] = self.type
        if self.device is not None:
            kwargs["device"] = self.device

        # Call the node function
        result = node.load_clip(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class CLIPVisionEncode(BaseNode):
    """CLIPVisionEncode node from ComfyUI (category: conditioning)"""

    clip_vision: Any = Field(default=None, description="clip_vision parameter")
    image: Any = Field(default=None, description="image parameter")
    crop: str = Field(default=None, description="crop parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPVisionEncode node."""
        # Import the ComfyUI node class
        from nodes import CLIPVisionEncode

        # Create node instance
        node = CLIPVisionEncode()

        # Prepare inputs
        kwargs = {}
        kwargs["clip_vision"] = self.clip_vision
        kwargs["image"] = self.image
        kwargs["crop"] = self.crop

        # Call the node function
        result = node.encode(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class StyleModelApply(BaseNode):
    """StyleModelApply node from ComfyUI (category: conditioning/style_model)"""

    conditioning: Any = Field(default=None, description="conditioning parameter")
    style_model: Any = Field(default=None, description="style_model parameter")
    clip_vision_output: Any = Field(default=None, description="clip_vision_output parameter")
    strength: float = Field(default=1.0, description="strength parameter", ge=0.0, le=10.0)
    strength_type: str = Field(default=None, description="strength_type parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the StyleModelApply node."""
        # Import the ComfyUI node class
        from nodes import StyleModelApply

        # Create node instance
        node = StyleModelApply()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning"] = self.conditioning
        kwargs["style_model"] = self.style_model
        kwargs["clip_vision_output"] = self.clip_vision_output
        kwargs["strength"] = self.strength
        kwargs["strength_type"] = self.strength_type

        # Call the node function
        result = node.apply_stylemodel(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class unCLIPConditioning(BaseNode):
    """unCLIPConditioning node from ComfyUI (category: conditioning)"""

    conditioning: Any = Field(default=None, description="conditioning parameter")
    clip_vision_output: Any = Field(default=None, description="clip_vision_output parameter")
    strength: float = Field(default=1.0, description="strength parameter", le=10.0)
    noise_augmentation: float = Field(default=0.0, description="noise_augmentation parameter", ge=0.0, le=1.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the unCLIPConditioning node."""
        # Import the ComfyUI node class
        from nodes import unCLIPConditioning

        # Create node instance
        node = unCLIPConditioning()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning"] = self.conditioning
        kwargs["clip_vision_output"] = self.clip_vision_output
        kwargs["strength"] = self.strength
        kwargs["noise_augmentation"] = self.noise_augmentation

        # Call the node function
        result = node.apply_adm(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ControlNetApplyAdvanced(BaseNode):
    """ControlNetApplyAdvanced node from ComfyUI (category: conditioning/controlnet)"""

    positive: Any = Field(default=None, description="positive parameter")
    negative: Any = Field(default=None, description="negative parameter")
    control_net: Any = Field(default=None, description="control_net parameter")
    image: Any = Field(default=None, description="image parameter")
    strength: float = Field(default=1.0, description="strength parameter", ge=0.0, le=10.0)
    start_percent: float = Field(default=0.0, description="start_percent parameter", ge=0.0, le=1.0)
    end_percent: float = Field(default=1.0, description="end_percent parameter", ge=0.0, le=1.0)
    vae: Optional[Any] = Field(default=None, description="vae parameter")

    async def process(self, context: ProcessingContext) -> tuple[Any, Any]:
        """Process the ControlNetApplyAdvanced node."""
        # Import the ComfyUI node class
        from nodes import ControlNetApplyAdvanced

        # Create node instance
        node = ControlNetApplyAdvanced()

        # Prepare inputs
        kwargs = {}
        kwargs["positive"] = self.positive
        kwargs["negative"] = self.negative
        kwargs["control_net"] = self.control_net
        kwargs["image"] = self.image
        kwargs["strength"] = self.strength
        kwargs["start_percent"] = self.start_percent
        kwargs["end_percent"] = self.end_percent
        if self.vae is not None:
            kwargs["vae"] = self.vae

        # Call the node function
        result = node.apply_controlnet(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class ControlNetLoader(BaseNode):
    """ControlNetLoader node from ComfyUI (category: loaders)"""

    control_net_name: Any = Field(default=None, description="control_net_name parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ControlNetLoader node."""
        # Import the ComfyUI node class
        from nodes import ControlNetLoader

        # Create node instance
        node = ControlNetLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["control_net_name"] = self.control_net_name

        # Call the node function
        result = node.load_controlnet(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class DiffControlNetLoader(BaseNode):
    """DiffControlNetLoader node from ComfyUI (category: loaders)"""

    model: Any = Field(default=None, description="model parameter")
    control_net_name: Any = Field(default=None, description="control_net_name parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the DiffControlNetLoader node."""
        # Import the ComfyUI node class
        from nodes import DiffControlNetLoader

        # Create node instance
        node = DiffControlNetLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["control_net_name"] = self.control_net_name

        # Call the node function
        result = node.load_controlnet(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class StyleModelLoader(BaseNode):
    """StyleModelLoader node from ComfyUI (category: loaders)"""

    style_model_name: Any = Field(default=None, description="style_model_name parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the StyleModelLoader node."""
        # Import the ComfyUI node class
        from nodes import StyleModelLoader

        # Create node instance
        node = StyleModelLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["style_model_name"] = self.style_model_name

        # Call the node function
        result = node.load_style_model(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class CLIPVisionLoader(BaseNode):
    """CLIPVisionLoader node from ComfyUI (category: loaders)"""

    clip_name: Any = Field(default=None, description="clip_name parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPVisionLoader node."""
        # Import the ComfyUI node class
        from nodes import CLIPVisionLoader

        # Create node instance
        node = CLIPVisionLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["clip_name"] = self.clip_name

        # Call the node function
        result = node.load_clip(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class VAEDecodeTiled(BaseNode):
    """VAEDecodeTiled node from ComfyUI (category: _for_testing)"""

    samples: Any = Field(default=None, description="samples parameter")
    vae: Any = Field(default=None, description="vae parameter")
    tile_size: int = Field(default=512, description="tile_size parameter", ge=64, le=4096)
    overlap: int = Field(default=64, description="overlap parameter", ge=0, le=4096)
    temporal_size: int = Field(default=64, description="Only used for video VAEs: Amount of frames to decode at a time.", ge=8, le=4096)
    temporal_overlap: int = Field(default=8, description="Only used for video VAEs: Amount of frames to overlap.", ge=4, le=4096)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAEDecodeTiled node."""
        # Import the ComfyUI node class
        from nodes import VAEDecodeTiled

        # Create node instance
        node = VAEDecodeTiled()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["vae"] = self.vae
        kwargs["tile_size"] = self.tile_size
        kwargs["overlap"] = self.overlap
        kwargs["temporal_size"] = self.temporal_size
        kwargs["temporal_overlap"] = self.temporal_overlap

        # Call the node function
        result = node.decode(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class VAEEncodeTiled(BaseNode):
    """VAEEncodeTiled node from ComfyUI (category: _for_testing)"""

    pixels: Any = Field(default=None, description="pixels parameter")
    vae: Any = Field(default=None, description="vae parameter")
    tile_size: int = Field(default=512, description="tile_size parameter", ge=64, le=4096)
    overlap: int = Field(default=64, description="overlap parameter", ge=0, le=4096)
    temporal_size: int = Field(default=64, description="Only used for video VAEs: Amount of frames to encode at a time.", ge=8, le=4096)
    temporal_overlap: int = Field(default=8, description="Only used for video VAEs: Amount of frames to overlap.", ge=4, le=4096)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAEEncodeTiled node."""
        # Import the ComfyUI node class
        from nodes import VAEEncodeTiled

        # Create node instance
        node = VAEEncodeTiled()

        # Prepare inputs
        kwargs = {}
        kwargs["pixels"] = self.pixels
        kwargs["vae"] = self.vae
        kwargs["tile_size"] = self.tile_size
        kwargs["overlap"] = self.overlap
        kwargs["temporal_size"] = self.temporal_size
        kwargs["temporal_overlap"] = self.temporal_overlap

        # Call the node function
        result = node.encode(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class unCLIPCheckpointLoader(BaseNode):
    """unCLIPCheckpointLoader node from ComfyUI (category: loaders)"""

    ckpt_name: Any = Field(default=None, description="ckpt_name parameter")

    async def process(self, context: ProcessingContext) -> tuple[Any, Any, Any, Any]:
        """Process the unCLIPCheckpointLoader node."""
        # Import the ComfyUI node class
        from nodes import unCLIPCheckpointLoader

        # Create node instance
        node = unCLIPCheckpointLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["ckpt_name"] = self.ckpt_name

        # Call the node function
        result = node.load_checkpoint(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class GLIGENLoader(BaseNode):
    """GLIGENLoader node from ComfyUI (category: loaders)"""

    gligen_name: Any = Field(default=None, description="gligen_name parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the GLIGENLoader node."""
        # Import the ComfyUI node class
        from nodes import GLIGENLoader

        # Create node instance
        node = GLIGENLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["gligen_name"] = self.gligen_name

        # Call the node function
        result = node.load_gligen(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class GLIGENTextBoxApply(BaseNode):
    """GLIGENTextBoxApply node from ComfyUI (category: conditioning/gligen)"""

    conditioning_to: Any = Field(default=None, description="conditioning_to parameter")
    clip: Any = Field(default=None, description="clip parameter")
    gligen_textbox_model: Any = Field(default=None, description="gligen_textbox_model parameter")
    text: str = Field(default="", description="text parameter")
    width: int = Field(default=64, description="width parameter", ge=8, le={'_ref': 'MAX_RESOLUTION'})
    height: int = Field(default=64, description="height parameter", ge=8, le={'_ref': 'MAX_RESOLUTION'})
    x: int = Field(default=0, description="x parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    y: int = Field(default=0, description="y parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})

    async def process(self, context: ProcessingContext) -> Any:
        """Process the GLIGENTextBoxApply node."""
        # Import the ComfyUI node class
        from nodes import GLIGENTextBoxApply

        # Create node instance
        node = GLIGENTextBoxApply()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning_to"] = self.conditioning_to
        kwargs["clip"] = self.clip
        kwargs["gligen_textbox_model"] = self.gligen_textbox_model
        kwargs["text"] = self.text
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["x"] = self.x
        kwargs["y"] = self.y

        # Call the node function
        result = node.append(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class InpaintModelConditioning(BaseNode):
    """InpaintModelConditioning node from ComfyUI (category: conditioning/inpaint)"""

    positive: Any = Field(default=None, description="positive parameter")
    negative: Any = Field(default=None, description="negative parameter")
    vae: Any = Field(default=None, description="vae parameter")
    pixels: Any = Field(default=None, description="pixels parameter")
    mask: Any = Field(default=None, description="mask parameter")
    noise_mask: bool = Field(default=True, description="Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model.")

    async def process(self, context: ProcessingContext) -> tuple[Any, Any, Any]:
        """Process the InpaintModelConditioning node."""
        # Import the ComfyUI node class
        from nodes import InpaintModelConditioning

        # Create node instance
        node = InpaintModelConditioning()

        # Prepare inputs
        kwargs = {}
        kwargs["positive"] = self.positive
        kwargs["negative"] = self.negative
        kwargs["vae"] = self.vae
        kwargs["pixels"] = self.pixels
        kwargs["mask"] = self.mask
        kwargs["noise_mask"] = self.noise_mask

        # Call the node function
        result = node.encode(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class DiffusersLoader(BaseNode):
    """DiffusersLoader node from ComfyUI (category: advanced/loaders/deprecated)"""

    model_path: Any = Field(default=None, description="model_path parameter")

    async def process(self, context: ProcessingContext) -> tuple[Any, Any, Any]:
        """Process the DiffusersLoader node."""
        # Import the ComfyUI node class
        from nodes import DiffusersLoader

        # Create node instance
        node = DiffusersLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["model_path"] = self.model_path

        # Call the node function
        result = node.load_checkpoint(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class LoadLatent(BaseNode):
    """LoadLatent node from ComfyUI (category: _for_testing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoadLatent node."""
        # Import the ComfyUI node class
        from nodes import LoadLatent

        # Create node instance
        node = LoadLatent()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.load(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class SaveLatent(BaseNode):
    """SaveLatent node from ComfyUI (category: _for_testing)"""

    samples: Any = Field(default=None, description="samples parameter")
    filename_prefix: str = Field(default="latents/ComfyUI", description="filename_prefix parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveLatent node."""
        # Import the ComfyUI node class
        from nodes import SaveLatent

        # Create node instance
        node = SaveLatent()

        # Prepare inputs
        kwargs = {}
        kwargs["samples"] = self.samples
        kwargs["filename_prefix"] = self.filename_prefix

        # Call the node function
        result = node.save(**kwargs)

        # Return result
        return result


class ConditioningZeroOut(BaseNode):
    """ConditioningZeroOut node from ComfyUI (category: advanced/conditioning)"""

    conditioning: Any = Field(default=None, description="conditioning parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningZeroOut node."""
        # Import the ComfyUI node class
        from nodes import ConditioningZeroOut

        # Create node instance
        node = ConditioningZeroOut()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning"] = self.conditioning

        # Call the node function
        result = node.zero_out(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ConditioningSetTimestepRange(BaseNode):
    """ConditioningSetTimestepRange node from ComfyUI (category: advanced/conditioning)"""

    conditioning: Any = Field(default=None, description="conditioning parameter")
    start: float = Field(default=0.0, description="start parameter", ge=0.0, le=1.0)
    end: float = Field(default=1.0, description="end parameter", ge=0.0, le=1.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningSetTimestepRange node."""
        # Import the ComfyUI node class
        from nodes import ConditioningSetTimestepRange

        # Create node instance
        node = ConditioningSetTimestepRange()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning"] = self.conditioning
        kwargs["start"] = self.start
        kwargs["end"] = self.end

        # Call the node function
        result = node.set_range(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class LoraLoaderModelOnly(BaseNode):
    """LoraLoaderModelOnly node from ComfyUI (category: uncategorized)"""

    model: Any = Field(default=None, description="model parameter")
    lora_name: Any = Field(default=None, description="lora_name parameter")
    strength_model: float = Field(default=1.0, description="strength_model parameter", le=100.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoraLoaderModelOnly node."""
        # Import the ComfyUI node class
        from nodes import LoraLoaderModelOnly

        # Create node instance
        node = LoraLoaderModelOnly()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["lora_name"] = self.lora_name
        kwargs["strength_model"] = self.strength_model

        # Call the node function
        result = node.load_lora_model_only(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class TextEncodeAceStepAudio(BaseNode):
    """TextEncodeAceStepAudio node from ComfyUI (category: conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TextEncodeAceStepAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_ace import TextEncodeAceStepAudio

        # Create node instance
        node = TextEncodeAceStepAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyAceStepLatentAudio(BaseNode):
    """EmptyAceStepLatentAudio node from ComfyUI (category: latent/audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyAceStepLatentAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_ace import EmptyAceStepLatentAudio

        # Create node instance
        node = EmptyAceStepLatentAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerLCMUpscale(BaseNode):
    """SamplerLCMUpscale node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerLCMUpscale node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_advanced_samplers import SamplerLCMUpscale

        # Create node instance
        node = SamplerLCMUpscale()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerEulerCFGpp(BaseNode):
    """SamplerEulerCFGpp node from ComfyUI (category: _for_testing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerEulerCFGpp node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_advanced_samplers import SamplerEulerCFGpp

        # Create node instance
        node = SamplerEulerCFGpp()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class AlignYourStepsScheduler(BaseNode):
    """AlignYourStepsScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the AlignYourStepsScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler

        # Create node instance
        node = AlignYourStepsScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class APG(BaseNode):
    """APG node from ComfyUI (category: sampling/custom_sampling)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the APG node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_apg import APG

        # Create node instance
        node = APG()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class UNetSelfAttentionMultiply(BaseNode):
    """UNetSelfAttentionMultiply node from ComfyUI (category: _for_testing/attention_experiments)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the UNetSelfAttentionMultiply node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_attention_multiply import UNetSelfAttentionMultiply

        # Create node instance
        node = UNetSelfAttentionMultiply()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class UNetCrossAttentionMultiply(BaseNode):
    """UNetCrossAttentionMultiply node from ComfyUI (category: _for_testing/attention_experiments)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the UNetCrossAttentionMultiply node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_attention_multiply import UNetCrossAttentionMultiply

        # Create node instance
        node = UNetCrossAttentionMultiply()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPAttentionMultiply(BaseNode):
    """CLIPAttentionMultiply node from ComfyUI (category: _for_testing/attention_experiments)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPAttentionMultiply node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_attention_multiply import CLIPAttentionMultiply

        # Create node instance
        node = CLIPAttentionMultiply()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class UNetTemporalAttentionMultiply(BaseNode):
    """UNetTemporalAttentionMultiply node from ComfyUI (category: _for_testing/attention_experiments)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the UNetTemporalAttentionMultiply node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_attention_multiply import UNetTemporalAttentionMultiply

        # Create node instance
        node = UNetTemporalAttentionMultiply()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyLatentAudio(BaseNode):
    """EmptyLatentAudio node from ComfyUI (category: latent/audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyLatentAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import EmptyLatentAudio

        # Create node instance
        node = EmptyLatentAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ConditioningStableAudio(BaseNode):
    """ConditioningStableAudio node from ComfyUI (category: conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningStableAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import ConditioningStableAudio

        # Create node instance
        node = ConditioningStableAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class VAEEncodeAudio(BaseNode):
    """VAEEncodeAudio node from ComfyUI (category: latent/audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAEEncodeAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import VAEEncodeAudio

        # Create node instance
        node = VAEEncodeAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class VAEDecodeAudio(BaseNode):
    """VAEDecodeAudio node from ComfyUI (category: latent/audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAEDecodeAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import VAEDecodeAudio

        # Create node instance
        node = VAEDecodeAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveAudio(BaseNode):
    """SaveAudio node from ComfyUI (category: audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import SaveAudio

        # Create node instance
        node = SaveAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveAudioMP3(BaseNode):
    """SaveAudioMP3 node from ComfyUI (category: audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveAudioMP3 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import SaveAudioMP3

        # Create node instance
        node = SaveAudioMP3()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveAudioOpus(BaseNode):
    """SaveAudioOpus node from ComfyUI (category: audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveAudioOpus node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import SaveAudioOpus

        # Create node instance
        node = SaveAudioOpus()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class PreviewAudio(BaseNode):
    """PreviewAudio node from ComfyUI (category: audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PreviewAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import PreviewAudio

        # Create node instance
        node = PreviewAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LoadAudio(BaseNode):
    """LoadAudio node from ComfyUI (category: audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoadAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import LoadAudio

        # Create node instance
        node = LoadAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class RecordAudio(BaseNode):
    """RecordAudio node from ComfyUI (category: audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the RecordAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import RecordAudio

        # Create node instance
        node = RecordAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TrimAudioDuration(BaseNode):
    """
    Trim audio tensor into chosen time range.
    
    Category: audio
    ComfyUI Node ID: TrimAudioDuration
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TrimAudioDuration node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import TrimAudioDuration

        # Create node instance
        node = TrimAudioDuration()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SplitAudioChannels(BaseNode):
    """
    Separates the audio into left and right channels.
    
    Category: audio
    ComfyUI Node ID: SplitAudioChannels
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SplitAudioChannels node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import SplitAudioChannels

        # Create node instance
        node = SplitAudioChannels()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class AudioConcat(BaseNode):
    """
    Concatenates the audio1 to audio2 in the specified direction.
    
    Category: audio
    ComfyUI Node ID: AudioConcat
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the AudioConcat node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import AudioConcat

        # Create node instance
        node = AudioConcat()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class AudioMerge(BaseNode):
    """
    Combine two audio tracks by overlaying their waveforms.
    
    Category: audio
    ComfyUI Node ID: AudioMerge
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the AudioMerge node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import AudioMerge

        # Create node instance
        node = AudioMerge()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class AudioAdjustVolume(BaseNode):
    """AudioAdjustVolume node from ComfyUI (category: audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the AudioAdjustVolume node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import AudioAdjustVolume

        # Create node instance
        node = AudioAdjustVolume()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyAudio(BaseNode):
    """EmptyAudio node from ComfyUI (category: audio)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyAudio node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio import EmptyAudio

        # Create node instance
        node = EmptyAudio()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class AudioEncoderLoader(BaseNode):
    """AudioEncoderLoader node from ComfyUI (category: loaders)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the AudioEncoderLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio_encoder import AudioEncoderLoader

        # Create node instance
        node = AudioEncoderLoader()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class AudioEncoderEncode(BaseNode):
    """AudioEncoderEncode node from ComfyUI (category: conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the AudioEncoderEncode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_audio_encoder import AudioEncoderEncode

        # Create node instance
        node = AudioEncoderEncode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanCameraEmbedding(BaseNode):
    """WanCameraEmbedding node from ComfyUI (category: camera)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanCameraEmbedding node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_camera_trajectory import WanCameraEmbedding

        # Create node instance
        node = WanCameraEmbedding()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Canny(BaseNode):
    """Canny node from ComfyUI (category: image/preprocessors)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Canny node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_canny import Canny

        # Create node instance
        node = Canny()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CFGZeroStar(BaseNode):
    """CFGZeroStar node from ComfyUI (category: advanced/guidance)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CFGZeroStar node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_cfg import CFGZeroStar

        # Create node instance
        node = CFGZeroStar()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CFGNorm(BaseNode):
    """CFGNorm node from ComfyUI (category: advanced/guidance)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CFGNorm node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_cfg import CFGNorm

        # Create node instance
        node = CFGNorm()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyChromaRadianceLatentImage(BaseNode):
    """EmptyChromaRadianceLatentImage node from ComfyUI (category: latent/chroma_radiance)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyChromaRadianceLatentImage node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_chroma_radiance import EmptyChromaRadianceLatentImage

        # Create node instance
        node = EmptyChromaRadianceLatentImage()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ChromaRadianceOptions(BaseNode):
    """
    Allows setting advanced options for the Chroma Radiance model.
    
    Category: model_patches/chroma_radiance
    ComfyUI Node ID: ChromaRadianceOptions
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ChromaRadianceOptions node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_chroma_radiance import ChromaRadianceOptions

        # Create node instance
        node = ChromaRadianceOptions()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodeSDXLRefiner(BaseNode):
    """CLIPTextEncodeSDXLRefiner node from ComfyUI (category: advanced/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodeSDXLRefiner node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXLRefiner

        # Create node instance
        node = CLIPTextEncodeSDXLRefiner()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodeSDXL(BaseNode):
    """CLIPTextEncodeSDXL node from ComfyUI (category: advanced/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodeSDXL node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL

        # Create node instance
        node = CLIPTextEncodeSDXL()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class PorterDuffImageComposite(BaseNode):
    """PorterDuffImageComposite node from ComfyUI (category: mask/compositing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PorterDuffImageComposite node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_compositing import PorterDuffImageComposite

        # Create node instance
        node = PorterDuffImageComposite()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SplitImageWithAlpha(BaseNode):
    """SplitImageWithAlpha node from ComfyUI (category: mask/compositing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SplitImageWithAlpha node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_compositing import SplitImageWithAlpha

        # Create node instance
        node = SplitImageWithAlpha()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class JoinImageWithAlpha(BaseNode):
    """JoinImageWithAlpha node from ComfyUI (category: mask/compositing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the JoinImageWithAlpha node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_compositing import JoinImageWithAlpha

        # Create node instance
        node = JoinImageWithAlpha()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodeControlnet(BaseNode):
    """CLIPTextEncodeControlnet node from ComfyUI (category: _for_testing/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodeControlnet node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_cond import CLIPTextEncodeControlnet

        # Create node instance
        node = CLIPTextEncodeControlnet()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class T5TokenizerOptions(BaseNode):
    """T5TokenizerOptions node from ComfyUI (category: _for_testing/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the T5TokenizerOptions node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_cond import T5TokenizerOptions

        # Create node instance
        node = T5TokenizerOptions()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ContextWindowsManualNode(BaseNode):
    """
    Manually set context windows.
    
    Category: context
    ComfyUI Node ID: ContextWindowsManual
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ContextWindowsManual node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_context_windows import ContextWindowsManualNode

        # Create node instance
        node = ContextWindowsManualNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SetUnionControlNetType(BaseNode):
    """SetUnionControlNetType node from ComfyUI (category: conditioning/controlnet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SetUnionControlNetType node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_controlnet import SetUnionControlNetType

        # Create node instance
        node = SetUnionControlNetType()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ControlNetInpaintingAliMamaApply(BaseNode):
    """ControlNetInpaintingAliMamaApply node from ComfyUI (category: conditioning/controlnet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ControlNetInpaintingAliMamaApply node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_controlnet import ControlNetInpaintingAliMamaApply

        # Create node instance
        node = ControlNetInpaintingAliMamaApply()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyCosmosLatentVideo(BaseNode):
    """EmptyCosmosLatentVideo node from ComfyUI (category: latent/video)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyCosmosLatentVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_cosmos import EmptyCosmosLatentVideo

        # Create node instance
        node = EmptyCosmosLatentVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CosmosImageToVideoLatent(BaseNode):
    """CosmosImageToVideoLatent node from ComfyUI (category: conditioning/inpaint)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CosmosImageToVideoLatent node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_cosmos import CosmosImageToVideoLatent

        # Create node instance
        node = CosmosImageToVideoLatent()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CosmosPredict2ImageToVideoLatent(BaseNode):
    """CosmosPredict2ImageToVideoLatent node from ComfyUI (category: conditioning/inpaint)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CosmosPredict2ImageToVideoLatent node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_cosmos import CosmosPredict2ImageToVideoLatent

        # Create node instance
        node = CosmosPredict2ImageToVideoLatent()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class BasicScheduler(BaseNode):
    """BasicScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the BasicScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import BasicScheduler

        # Create node instance
        node = BasicScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class KarrasScheduler(BaseNode):
    """KarrasScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the KarrasScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import KarrasScheduler

        # Create node instance
        node = KarrasScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ExponentialScheduler(BaseNode):
    """ExponentialScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ExponentialScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import ExponentialScheduler

        # Create node instance
        node = ExponentialScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class PolyexponentialScheduler(BaseNode):
    """PolyexponentialScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PolyexponentialScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import PolyexponentialScheduler

        # Create node instance
        node = PolyexponentialScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LaplaceScheduler(BaseNode):
    """LaplaceScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LaplaceScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import LaplaceScheduler

        # Create node instance
        node = LaplaceScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SDTurboScheduler(BaseNode):
    """SDTurboScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SDTurboScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SDTurboScheduler

        # Create node instance
        node = SDTurboScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class BetaSamplingScheduler(BaseNode):
    """BetaSamplingScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the BetaSamplingScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import BetaSamplingScheduler

        # Create node instance
        node = BetaSamplingScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class VPScheduler(BaseNode):
    """VPScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the VPScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import VPScheduler

        # Create node instance
        node = VPScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SplitSigmas(BaseNode):
    """SplitSigmas node from ComfyUI (category: sampling/custom_sampling/sigmas)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SplitSigmas node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SplitSigmas

        # Create node instance
        node = SplitSigmas()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SplitSigmasDenoise(BaseNode):
    """SplitSigmasDenoise node from ComfyUI (category: sampling/custom_sampling/sigmas)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SplitSigmasDenoise node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SplitSigmasDenoise

        # Create node instance
        node = SplitSigmasDenoise()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class FlipSigmas(BaseNode):
    """FlipSigmas node from ComfyUI (category: sampling/custom_sampling/sigmas)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the FlipSigmas node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import FlipSigmas

        # Create node instance
        node = FlipSigmas()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SetFirstSigma(BaseNode):
    """SetFirstSigma node from ComfyUI (category: sampling/custom_sampling/sigmas)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SetFirstSigma node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SetFirstSigma

        # Create node instance
        node = SetFirstSigma()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ExtendIntermediateSigmas(BaseNode):
    """ExtendIntermediateSigmas node from ComfyUI (category: sampling/custom_sampling/sigmas)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ExtendIntermediateSigmas node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import ExtendIntermediateSigmas

        # Create node instance
        node = ExtendIntermediateSigmas()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplingPercentToSigma(BaseNode):
    """SamplingPercentToSigma node from ComfyUI (category: sampling/custom_sampling/sigmas)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplingPercentToSigma node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplingPercentToSigma

        # Create node instance
        node = SamplingPercentToSigma()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class KSamplerSelect(BaseNode):
    """KSamplerSelect node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the KSamplerSelect node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import KSamplerSelect

        # Create node instance
        node = KSamplerSelect()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerDPMPP_3M_SDE(BaseNode):
    """SamplerDPMPP_3M_SDE node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerDPMPP_3M_SDE node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerDPMPP_3M_SDE

        # Create node instance
        node = SamplerDPMPP_3M_SDE()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerDPMPP_2M_SDE(BaseNode):
    """SamplerDPMPP_2M_SDE node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerDPMPP_2M_SDE node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerDPMPP_2M_SDE

        # Create node instance
        node = SamplerDPMPP_2M_SDE()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerDPMPP_SDE(BaseNode):
    """SamplerDPMPP_SDE node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerDPMPP_SDE node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerDPMPP_SDE

        # Create node instance
        node = SamplerDPMPP_SDE()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerDPMPP_2S_Ancestral(BaseNode):
    """SamplerDPMPP_2S_Ancestral node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerDPMPP_2S_Ancestral node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerDPMPP_2S_Ancestral

        # Create node instance
        node = SamplerDPMPP_2S_Ancestral()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerEulerAncestral(BaseNode):
    """SamplerEulerAncestral node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerEulerAncestral node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerEulerAncestral

        # Create node instance
        node = SamplerEulerAncestral()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerEulerAncestralCFGPP(BaseNode):
    """SamplerEulerAncestralCFGPP node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerEulerAncestralCFGPP node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerEulerAncestralCFGPP

        # Create node instance
        node = SamplerEulerAncestralCFGPP()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerLMS(BaseNode):
    """SamplerLMS node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerLMS node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerLMS

        # Create node instance
        node = SamplerLMS()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerDPMAdaptative(BaseNode):
    """SamplerDPMAdaptative node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerDPMAdaptative node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerDPMAdaptative

        # Create node instance
        node = SamplerDPMAdaptative()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerER_SDE(BaseNode):
    """SamplerER_SDE node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerER_SDE node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerER_SDE

        # Create node instance
        node = SamplerER_SDE()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerSASolver(BaseNode):
    """SamplerSASolver node from ComfyUI (category: sampling/custom_sampling/samplers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerSASolver node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerSASolver

        # Create node instance
        node = SamplerSASolver()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerSEEDS2(BaseNode):
    """
    This sampler node can represent multiple samplers:

seeds_2
- default setting

exp_heun_2_x0
- solver_type=phi_2, r=1.0, eta=0.0

exp_heun_2_x0_sde
- solver_type=phi_2, r=1.0, eta=1.0, s_noise=1.0
    
    Category: sampling/custom_sampling/samplers
    ComfyUI Node ID: SamplerSEEDS2
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerSEEDS2 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerSEEDS2

        # Create node instance
        node = SamplerSEEDS2()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerCustom(BaseNode):
    """SamplerCustom node from ComfyUI (category: sampling/custom_sampling)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerCustom node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerCustom

        # Create node instance
        node = SamplerCustom()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class BasicGuider(BaseNode):
    """BasicGuider node from ComfyUI (category: sampling/custom_sampling/guiders)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the BasicGuider node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import BasicGuider

        # Create node instance
        node = BasicGuider()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CFGGuider(BaseNode):
    """CFGGuider node from ComfyUI (category: sampling/custom_sampling/guiders)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CFGGuider node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import CFGGuider

        # Create node instance
        node = CFGGuider()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class DualCFGGuider(BaseNode):
    """DualCFGGuider node from ComfyUI (category: sampling/custom_sampling/guiders)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the DualCFGGuider node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import DualCFGGuider

        # Create node instance
        node = DualCFGGuider()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class DisableNoise(BaseNode):
    """DisableNoise node from ComfyUI (category: sampling/custom_sampling/noise)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the DisableNoise node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import DisableNoise

        # Create node instance
        node = DisableNoise()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class RandomNoise(BaseNode):
    """RandomNoise node from ComfyUI (category: sampling/custom_sampling/noise)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the RandomNoise node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import RandomNoise

        # Create node instance
        node = RandomNoise()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SamplerCustomAdvanced(BaseNode):
    """SamplerCustomAdvanced node from ComfyUI (category: sampling/custom_sampling)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SamplerCustomAdvanced node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

        # Create node instance
        node = SamplerCustomAdvanced()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class AddNoise(BaseNode):
    """AddNoise node from ComfyUI (category: _for_testing/custom_sampling/noise)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the AddNoise node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import AddNoise

        # Create node instance
        node = AddNoise()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ManualSigmas(BaseNode):
    """ManualSigmas node from ComfyUI (category: _for_testing/custom_sampling)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ManualSigmas node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_custom_sampler import ManualSigmas

        # Create node instance
        node = ManualSigmas()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LoadImageDataSetFromFolderNode(BaseNode):
    """LoadImageDataSetFromFolder node from ComfyUI (category: dataset)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoadImageDataSetFromFolder node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import LoadImageDataSetFromFolderNode

        # Create node instance
        node = LoadImageDataSetFromFolderNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LoadImageTextDataSetFromFolderNode(BaseNode):
    """LoadImageTextDataSetFromFolder node from ComfyUI (category: dataset)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoadImageTextDataSetFromFolder node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import LoadImageTextDataSetFromFolderNode

        # Create node instance
        node = LoadImageTextDataSetFromFolderNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveImageDataSetToFolderNode(BaseNode):
    """SaveImageDataSetToFolder node from ComfyUI (category: dataset)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveImageDataSetToFolder node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import SaveImageDataSetToFolderNode

        # Create node instance
        node = SaveImageDataSetToFolderNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveImageTextDataSetToFolderNode(BaseNode):
    """SaveImageTextDataSetToFolder node from ComfyUI (category: dataset)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveImageTextDataSetToFolder node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import SaveImageTextDataSetToFolderNode

        # Create node instance
        node = SaveImageTextDataSetToFolderNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageProcessingNode(BaseNode):
    """Unknown node from ComfyUI (category: dataset/image)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Unknown node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import ImageProcessingNode

        # Create node instance
        node = ImageProcessingNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TextProcessingNode(BaseNode):
    """Unknown node from ComfyUI (category: dataset/text)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Unknown node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import TextProcessingNode

        # Create node instance
        node = TextProcessingNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ShuffleImageTextDatasetNode(BaseNode):
    """ShuffleImageTextDataset node from ComfyUI (category: dataset/image)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ShuffleImageTextDataset node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import ShuffleImageTextDatasetNode

        # Create node instance
        node = ShuffleImageTextDatasetNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ResolutionBucket(BaseNode):
    """ResolutionBucket node from ComfyUI (category: dataset)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ResolutionBucket node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import ResolutionBucket

        # Create node instance
        node = ResolutionBucket()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class MakeTrainingDataset(BaseNode):
    """MakeTrainingDataset node from ComfyUI (category: dataset)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the MakeTrainingDataset node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import MakeTrainingDataset

        # Create node instance
        node = MakeTrainingDataset()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveTrainingDataset(BaseNode):
    """SaveTrainingDataset node from ComfyUI (category: dataset)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveTrainingDataset node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import SaveTrainingDataset

        # Create node instance
        node = SaveTrainingDataset()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LoadTrainingDataset(BaseNode):
    """LoadTrainingDataset node from ComfyUI (category: dataset)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoadTrainingDataset node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_dataset import LoadTrainingDataset

        # Create node instance
        node = LoadTrainingDataset()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class DifferentialDiffusion(BaseNode):
    """DifferentialDiffusion node from ComfyUI (category: _for_testing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the DifferentialDiffusion node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

        # Create node instance
        node = DifferentialDiffusion()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EasyCacheNode(BaseNode):
    """
    Native EasyCache implementation.
    
    Category: advanced/debug/model
    ComfyUI Node ID: EasyCache
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EasyCache node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_easycache import EasyCacheNode

        # Create node instance
        node = EasyCacheNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LazyCacheNode(BaseNode):
    """
    A homebrew version of EasyCache - even 'easier' version of EasyCache to implement. Overall works worse than EasyCache, but better in some rare cases AND universal compatibility with everything in ComfyUI.
    
    Category: advanced/debug/model
    ComfyUI Node ID: LazyCache
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LazyCache node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_easycache import LazyCacheNode

        # Create node instance
        node = LazyCacheNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ReferenceLatent(BaseNode):
    """
    This node sets the guiding latent for an edit model. If the model supports it you can chain multiple to set multiple reference images.
    
    Category: advanced/conditioning/edit_models
    ComfyUI Node ID: ReferenceLatent
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ReferenceLatent node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_edit_model import ReferenceLatent

        # Create node instance
        node = ReferenceLatent()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EpsilonScaling(BaseNode):
    """Epsilon Scaling node from ComfyUI (category: model_patches/unet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Epsilon Scaling node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_eps import EpsilonScaling

        # Create node instance
        node = EpsilonScaling()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TemporalScoreRescaling(BaseNode):
    """
    [Post-CFG Function]
TSR - Temporal Score Rescaling (2510.01184)

Rescaling the model's score or noise to steer the sampling diversity.

    
    Category: model_patches/unet
    ComfyUI Node ID: TemporalScoreRescaling
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TemporalScoreRescaling node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_eps import TemporalScoreRescaling

        # Create node instance
        node = TemporalScoreRescaling()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodeFlux(BaseNode):
    """CLIPTextEncodeFlux node from ComfyUI (category: advanced/conditioning/flux)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodeFlux node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_flux import CLIPTextEncodeFlux

        # Create node instance
        node = CLIPTextEncodeFlux()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyFlux2LatentImage(BaseNode):
    """EmptyFlux2LatentImage node from ComfyUI (category: latent)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyFlux2LatentImage node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_flux import EmptyFlux2LatentImage

        # Create node instance
        node = EmptyFlux2LatentImage()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class FluxGuidance(BaseNode):
    """FluxGuidance node from ComfyUI (category: advanced/conditioning/flux)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the FluxGuidance node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_flux import FluxGuidance

        # Create node instance
        node = FluxGuidance()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class FluxDisableGuidance(BaseNode):
    """
    This node completely disables the guidance embed on Flux and Flux like models
    
    Category: advanced/conditioning/flux
    ComfyUI Node ID: FluxDisableGuidance
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the FluxDisableGuidance node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_flux import FluxDisableGuidance

        # Create node instance
        node = FluxDisableGuidance()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class FluxKontextImageScale(BaseNode):
    """
    This node resizes the image to one that is more optimal for flux kontext.
    
    Category: advanced/conditioning/flux
    ComfyUI Node ID: FluxKontextImageScale
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the FluxKontextImageScale node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_flux import FluxKontextImageScale

        # Create node instance
        node = FluxKontextImageScale()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class FluxKontextMultiReferenceLatentMethod(BaseNode):
    """FluxKontextMultiReferenceLatentMethod node from ComfyUI (category: advanced/conditioning/flux)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the FluxKontextMultiReferenceLatentMethod node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_flux import FluxKontextMultiReferenceLatentMethod

        # Create node instance
        node = FluxKontextMultiReferenceLatentMethod()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Flux2Scheduler(BaseNode):
    """Flux2Scheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Flux2Scheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_flux import Flux2Scheduler

        # Create node instance
        node = Flux2Scheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class FreeU(BaseNode):
    """FreeU node from ComfyUI (category: model_patches/unet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the FreeU node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_freelunch import FreeU

        # Create node instance
        node = FreeU()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class FreeU_V2(BaseNode):
    """FreeU_V2 node from ComfyUI (category: model_patches/unet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the FreeU_V2 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_freelunch import FreeU_V2

        # Create node instance
        node = FreeU_V2()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class FreSca(BaseNode):
    """
    Applies frequency-dependent scaling to the guidance
    
    Category: _for_testing
    ComfyUI Node ID: FreSca
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the FreSca node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_fresca import FreSca

        # Create node instance
        node = FreSca()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class GITSScheduler(BaseNode):
    """GITSScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the GITSScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_gits import GITSScheduler

        # Create node instance
        node = GITSScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class QuadrupleCLIPLoader(BaseNode):
    """
    [Recipes]

hidream: long clip-l, long clip-g, t5xxl, llama_8b_3.1_instruct
    
    Category: advanced/loaders
    ComfyUI Node ID: QuadrupleCLIPLoader
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the QuadrupleCLIPLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hidream import QuadrupleCLIPLoader

        # Create node instance
        node = QuadrupleCLIPLoader()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodeHiDream(BaseNode):
    """CLIPTextEncodeHiDream node from ComfyUI (category: advanced/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodeHiDream node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hidream import CLIPTextEncodeHiDream

        # Create node instance
        node = CLIPTextEncodeHiDream()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodeHunyuanDiT(BaseNode):
    """CLIPTextEncodeHunyuanDiT node from ComfyUI (category: advanced/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodeHunyuanDiT node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import CLIPTextEncodeHunyuanDiT

        # Create node instance
        node = CLIPTextEncodeHunyuanDiT()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyHunyuanLatentVideo(BaseNode):
    """EmptyHunyuanLatentVideo node from ComfyUI (category: latent/video)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyHunyuanLatentVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo

        # Create node instance
        node = EmptyHunyuanLatentVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class HunyuanVideo15ImageToVideo(BaseNode):
    """HunyuanVideo15ImageToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the HunyuanVideo15ImageToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import HunyuanVideo15ImageToVideo

        # Create node instance
        node = HunyuanVideo15ImageToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class HunyuanVideo15SuperResolution(BaseNode):
    """HunyuanVideo15SuperResolution node from ComfyUI (category: uncategorized)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the HunyuanVideo15SuperResolution node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import HunyuanVideo15SuperResolution

        # Create node instance
        node = HunyuanVideo15SuperResolution()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentUpscaleModelLoader(BaseNode):
    """LatentUpscaleModelLoader node from ComfyUI (category: loaders)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentUpscaleModelLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader

        # Create node instance
        node = LatentUpscaleModelLoader()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class HunyuanVideo15LatentUpscaleWithModel(BaseNode):
    """HunyuanVideo15LatentUpscaleWithModel node from ComfyUI (category: latent)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the HunyuanVideo15LatentUpscaleWithModel node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import HunyuanVideo15LatentUpscaleWithModel

        # Create node instance
        node = HunyuanVideo15LatentUpscaleWithModel()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TextEncodeHunyuanVideo_ImageToVideo(BaseNode):
    """TextEncodeHunyuanVideo_ImageToVideo node from ComfyUI (category: advanced/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TextEncodeHunyuanVideo_ImageToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import TextEncodeHunyuanVideo_ImageToVideo

        # Create node instance
        node = TextEncodeHunyuanVideo_ImageToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class HunyuanImageToVideo(BaseNode):
    """HunyuanImageToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the HunyuanImageToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import HunyuanImageToVideo

        # Create node instance
        node = HunyuanImageToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyHunyuanImageLatent(BaseNode):
    """EmptyHunyuanImageLatent node from ComfyUI (category: latent)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyHunyuanImageLatent node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import EmptyHunyuanImageLatent

        # Create node instance
        node = EmptyHunyuanImageLatent()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class HunyuanRefinerLatent(BaseNode):
    """HunyuanRefinerLatent node from ComfyUI (category: uncategorized)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the HunyuanRefinerLatent node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan import HunyuanRefinerLatent

        # Create node instance
        node = HunyuanRefinerLatent()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyLatentHunyuan3Dv2(BaseNode):
    """EmptyLatentHunyuan3Dv2 node from ComfyUI (category: latent/3d)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyLatentHunyuan3Dv2 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan3d import EmptyLatentHunyuan3Dv2

        # Create node instance
        node = EmptyLatentHunyuan3Dv2()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Hunyuan3Dv2Conditioning(BaseNode):
    """Hunyuan3Dv2Conditioning node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Hunyuan3Dv2Conditioning node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan3d import Hunyuan3Dv2Conditioning

        # Create node instance
        node = Hunyuan3Dv2Conditioning()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Hunyuan3Dv2ConditioningMultiView(BaseNode):
    """Hunyuan3Dv2ConditioningMultiView node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Hunyuan3Dv2ConditioningMultiView node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan3d import Hunyuan3Dv2ConditioningMultiView

        # Create node instance
        node = Hunyuan3Dv2ConditioningMultiView()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class VAEDecodeHunyuan3D(BaseNode):
    """VAEDecodeHunyuan3D node from ComfyUI (category: latent/3d)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAEDecodeHunyuan3D node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan3d import VAEDecodeHunyuan3D

        # Create node instance
        node = VAEDecodeHunyuan3D()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class VoxelToMeshBasic(BaseNode):
    """VoxelToMeshBasic node from ComfyUI (category: 3d)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the VoxelToMeshBasic node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan3d import VoxelToMeshBasic

        # Create node instance
        node = VoxelToMeshBasic()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class VoxelToMesh(BaseNode):
    """VoxelToMesh node from ComfyUI (category: 3d)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the VoxelToMesh node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan3d import VoxelToMesh

        # Create node instance
        node = VoxelToMesh()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveGLB(BaseNode):
    """SaveGLB node from ComfyUI (category: 3d)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveGLB node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hunyuan3d import SaveGLB

        # Create node instance
        node = SaveGLB()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class HypernetworkLoader(BaseNode):
    """HypernetworkLoader node from ComfyUI (category: loaders)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the HypernetworkLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hypernetwork import HypernetworkLoader

        # Create node instance
        node = HypernetworkLoader()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class HyperTile(BaseNode):
    """HyperTile node from ComfyUI (category: model_patches/unet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the HyperTile node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_hypertile import HyperTile

        # Create node instance
        node = HyperTile()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageCrop(BaseNode):
    """ImageCrop node from ComfyUI (category: image/transform)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageCrop node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import ImageCrop

        # Create node instance
        node = ImageCrop()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class RepeatImageBatch(BaseNode):
    """RepeatImageBatch node from ComfyUI (category: image/batch)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the RepeatImageBatch node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import RepeatImageBatch

        # Create node instance
        node = RepeatImageBatch()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageFromBatch(BaseNode):
    """ImageFromBatch node from ComfyUI (category: image/batch)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageFromBatch node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import ImageFromBatch

        # Create node instance
        node = ImageFromBatch()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageAddNoise(BaseNode):
    """ImageAddNoise node from ComfyUI (category: image)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageAddNoise node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import ImageAddNoise

        # Create node instance
        node = ImageAddNoise()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveAnimatedWEBP(BaseNode):
    """SaveAnimatedWEBP node from ComfyUI (category: image/animation)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveAnimatedWEBP node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import SaveAnimatedWEBP

        # Create node instance
        node = SaveAnimatedWEBP()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveAnimatedPNG(BaseNode):
    """SaveAnimatedPNG node from ComfyUI (category: image/animation)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveAnimatedPNG node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import SaveAnimatedPNG

        # Create node instance
        node = SaveAnimatedPNG()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageStitch(BaseNode):
    """
    Stitches image2 to image1 in the specified direction.
If image2 is not provided, returns image1 unchanged.
Optional spacing can be added between images.
    
    Category: image/transform
    ComfyUI Node ID: ImageStitch
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageStitch node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import ImageStitch

        # Create node instance
        node = ImageStitch()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ResizeAndPadImage(BaseNode):
    """ResizeAndPadImage node from ComfyUI (category: image/transform)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ResizeAndPadImage node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import ResizeAndPadImage

        # Create node instance
        node = ResizeAndPadImage()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveSVGNode(BaseNode):
    """
    Save SVG files on disk.
    
    Category: image/save
    ComfyUI Node ID: SaveSVGNode
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveSVGNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import SaveSVGNode

        # Create node instance
        node = SaveSVGNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class GetImageSize(BaseNode):
    """
    Returns width and height of the image, and passes it through unchanged.
    
    Category: image
    ComfyUI Node ID: GetImageSize
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the GetImageSize node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import GetImageSize

        # Create node instance
        node = GetImageSize()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageRotate(BaseNode):
    """ImageRotate node from ComfyUI (category: image/transform)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageRotate node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import ImageRotate

        # Create node instance
        node = ImageRotate()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageFlip(BaseNode):
    """ImageFlip node from ComfyUI (category: image/transform)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageFlip node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import ImageFlip

        # Create node instance
        node = ImageFlip()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageScaleToMaxDimension(BaseNode):
    """ImageScaleToMaxDimension node from ComfyUI (category: image/upscaling)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageScaleToMaxDimension node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_images import ImageScaleToMaxDimension

        # Create node instance
        node = ImageScaleToMaxDimension()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class InstructPixToPixConditioning(BaseNode):
    """InstructPixToPixConditioning node from ComfyUI (category: conditioning/instructpix2pix)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the InstructPixToPixConditioning node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_ip2p import InstructPixToPixConditioning

        # Create node instance
        node = InstructPixToPixConditioning()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Kandinsky5ImageToVideo(BaseNode):
    """Kandinsky5ImageToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Kandinsky5ImageToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_kandinsky5 import Kandinsky5ImageToVideo

        # Create node instance
        node = Kandinsky5ImageToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class NormalizeVideoLatentStart(BaseNode):
    """
    Normalizes the initial frames of a video latent to match the mean and standard deviation of subsequent reference frames. Helps reduce differences between the starting frames and the rest of the video.
    
    Category: conditioning/video_models
    ComfyUI Node ID: NormalizeVideoLatentStart
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the NormalizeVideoLatentStart node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_kandinsky5 import NormalizeVideoLatentStart

        # Create node instance
        node = NormalizeVideoLatentStart()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodeKandinsky5(BaseNode):
    """CLIPTextEncodeKandinsky5 node from ComfyUI (category: advanced/conditioning/kandinsky5)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodeKandinsky5 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_kandinsky5 import CLIPTextEncodeKandinsky5

        # Create node instance
        node = CLIPTextEncodeKandinsky5()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentAdd(BaseNode):
    """LatentAdd node from ComfyUI (category: latent/advanced)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentAdd node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentAdd

        # Create node instance
        node = LatentAdd()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentSubtract(BaseNode):
    """LatentSubtract node from ComfyUI (category: latent/advanced)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentSubtract node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentSubtract

        # Create node instance
        node = LatentSubtract()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentMultiply(BaseNode):
    """LatentMultiply node from ComfyUI (category: latent/advanced)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentMultiply node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentMultiply

        # Create node instance
        node = LatentMultiply()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentInterpolate(BaseNode):
    """LatentInterpolate node from ComfyUI (category: latent/advanced)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentInterpolate node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentInterpolate

        # Create node instance
        node = LatentInterpolate()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentConcat(BaseNode):
    """LatentConcat node from ComfyUI (category: latent/advanced)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentConcat node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentConcat

        # Create node instance
        node = LatentConcat()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentCut(BaseNode):
    """LatentCut node from ComfyUI (category: latent/advanced)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentCut node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentCut

        # Create node instance
        node = LatentCut()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentCutToBatch(BaseNode):
    """LatentCutToBatch node from ComfyUI (category: latent/advanced)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentCutToBatch node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentCutToBatch

        # Create node instance
        node = LatentCutToBatch()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentBatch(BaseNode):
    """LatentBatch node from ComfyUI (category: latent/batch)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentBatch node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentBatch

        # Create node instance
        node = LatentBatch()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentBatchSeedBehavior(BaseNode):
    """LatentBatchSeedBehavior node from ComfyUI (category: latent/advanced)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentBatchSeedBehavior node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentBatchSeedBehavior

        # Create node instance
        node = LatentBatchSeedBehavior()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentApplyOperation(BaseNode):
    """LatentApplyOperation node from ComfyUI (category: latent/advanced/operations)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentApplyOperation node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentApplyOperation

        # Create node instance
        node = LatentApplyOperation()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentApplyOperationCFG(BaseNode):
    """LatentApplyOperationCFG node from ComfyUI (category: latent/advanced/operations)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentApplyOperationCFG node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentApplyOperationCFG

        # Create node instance
        node = LatentApplyOperationCFG()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentOperationTonemapReinhard(BaseNode):
    """LatentOperationTonemapReinhard node from ComfyUI (category: latent/advanced/operations)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentOperationTonemapReinhard node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentOperationTonemapReinhard

        # Create node instance
        node = LatentOperationTonemapReinhard()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentOperationSharpen(BaseNode):
    """LatentOperationSharpen node from ComfyUI (category: latent/advanced/operations)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentOperationSharpen node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import LatentOperationSharpen

        # Create node instance
        node = LatentOperationSharpen()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ReplaceVideoLatentFrames(BaseNode):
    """ReplaceVideoLatentFrames node from ComfyUI (category: latent/batch)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ReplaceVideoLatentFrames node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_latent import ReplaceVideoLatentFrames

        # Create node instance
        node = ReplaceVideoLatentFrames()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Load3D(BaseNode):
    """Load3D node from ComfyUI (category: 3d)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Load3D node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_load_3d import Load3D

        # Create node instance
        node = Load3D()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Preview3D(BaseNode):
    """Preview3D node from ComfyUI (category: 3d)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Preview3D node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_load_3d import Preview3D

        # Create node instance
        node = Preview3D()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SwitchNode(BaseNode):
    """ComfySwitchNode node from ComfyUI (category: logic)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ComfySwitchNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_logic import SwitchNode

        # Create node instance
        node = SwitchNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SoftSwitchNode(BaseNode):
    """ComfySoftSwitchNode node from ComfyUI (category: logic)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ComfySoftSwitchNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_logic import SoftSwitchNode

        # Create node instance
        node = SoftSwitchNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CustomComboNode(BaseNode):
    """CustomCombo node from ComfyUI (category: utils)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CustomCombo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_logic import CustomComboNode

        # Create node instance
        node = CustomComboNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class DCTestNode(BaseNode):
    """DCTestNode node from ComfyUI (category: logic)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the DCTestNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_logic import DCTestNode

        # Create node instance
        node = DCTestNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class AutogrowNamesTestNode(BaseNode):
    """AutogrowNamesTestNode node from ComfyUI (category: logic)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the AutogrowNamesTestNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_logic import AutogrowNamesTestNode

        # Create node instance
        node = AutogrowNamesTestNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class AutogrowPrefixTestNode(BaseNode):
    """AutogrowPrefixTestNode node from ComfyUI (category: logic)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the AutogrowPrefixTestNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_logic import AutogrowPrefixTestNode

        # Create node instance
        node = AutogrowPrefixTestNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ComboOutputTestNode(BaseNode):
    """ComboOptionTestNode node from ComfyUI (category: logic)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ComboOptionTestNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_logic import ComboOutputTestNode

        # Create node instance
        node = ComboOutputTestNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ConvertStringToComboNode(BaseNode):
    """ConvertStringToComboNode node from ComfyUI (category: logic)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConvertStringToComboNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_logic import ConvertStringToComboNode

        # Create node instance
        node = ConvertStringToComboNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class InvertBooleanNode(BaseNode):
    """InvertBooleanNode node from ComfyUI (category: logic)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the InvertBooleanNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_logic import InvertBooleanNode

        # Create node instance
        node = InvertBooleanNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LoraSave(BaseNode):
    """LoraSave node from ComfyUI (category: _for_testing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoraSave node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lora_extract import LoraSave

        # Create node instance
        node = LoraSave()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LotusConditioning(BaseNode):
    """LotusConditioning node from ComfyUI (category: conditioning/lotus)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LotusConditioning node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lotus import LotusConditioning

        # Create node instance
        node = LotusConditioning()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyLTXVLatentVideo(BaseNode):
    """EmptyLTXVLatentVideo node from ComfyUI (category: latent/video/ltxv)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyLTXVLatentVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lt import EmptyLTXVLatentVideo

        # Create node instance
        node = EmptyLTXVLatentVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LTXVImgToVideo(BaseNode):
    """LTXVImgToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LTXVImgToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lt import LTXVImgToVideo

        # Create node instance
        node = LTXVImgToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LTXVAddGuide(BaseNode):
    """LTXVAddGuide node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LTXVAddGuide node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lt import LTXVAddGuide

        # Create node instance
        node = LTXVAddGuide()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LTXVCropGuides(BaseNode):
    """LTXVCropGuides node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LTXVCropGuides node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lt import LTXVCropGuides

        # Create node instance
        node = LTXVCropGuides()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LTXVConditioning(BaseNode):
    """LTXVConditioning node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LTXVConditioning node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lt import LTXVConditioning

        # Create node instance
        node = LTXVConditioning()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ModelSamplingLTXV(BaseNode):
    """ModelSamplingLTXV node from ComfyUI (category: advanced/model)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelSamplingLTXV node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lt import ModelSamplingLTXV

        # Create node instance
        node = ModelSamplingLTXV()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LTXVScheduler(BaseNode):
    """LTXVScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LTXVScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lt import LTXVScheduler

        # Create node instance
        node = LTXVScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LTXVPreprocess(BaseNode):
    """LTXVPreprocess node from ComfyUI (category: image)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LTXVPreprocess node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lt import LTXVPreprocess

        # Create node instance
        node = LTXVPreprocess()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class RenormCFG(BaseNode):
    """RenormCFG node from ComfyUI (category: advanced/model)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the RenormCFG node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lumina2 import RenormCFG

        # Create node instance
        node = RenormCFG()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodeLumina2(BaseNode):
    """
    Encodes a system prompt and a user prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images.
    
    Category: conditioning
    ComfyUI Node ID: CLIPTextEncodeLumina2
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodeLumina2 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_lumina2 import CLIPTextEncodeLumina2

        # Create node instance
        node = CLIPTextEncodeLumina2()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Mahiro(BaseNode):
    """
    Modify the guidance to scale more on the 'direction' of the positive prompt rather than the difference between the negative prompt.
    
    Category: _for_testing
    ComfyUI Node ID: Mahiro
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Mahiro node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mahiro import Mahiro

        # Create node instance
        node = Mahiro()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentCompositeMasked(BaseNode):
    """LatentCompositeMasked node from ComfyUI (category: latent)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LatentCompositeMasked node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import LatentCompositeMasked

        # Create node instance
        node = LatentCompositeMasked()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageCompositeMasked(BaseNode):
    """ImageCompositeMasked node from ComfyUI (category: image)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageCompositeMasked node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import ImageCompositeMasked

        # Create node instance
        node = ImageCompositeMasked()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class MaskToImage(BaseNode):
    """MaskToImage node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the MaskToImage node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import MaskToImage

        # Create node instance
        node = MaskToImage()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageToMask(BaseNode):
    """ImageToMask node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageToMask node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import ImageToMask

        # Create node instance
        node = ImageToMask()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageColorToMask(BaseNode):
    """ImageColorToMask node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageColorToMask node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import ImageColorToMask

        # Create node instance
        node = ImageColorToMask()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SolidMask(BaseNode):
    """SolidMask node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SolidMask node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import SolidMask

        # Create node instance
        node = SolidMask()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class InvertMask(BaseNode):
    """InvertMask node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the InvertMask node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import InvertMask

        # Create node instance
        node = InvertMask()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CropMask(BaseNode):
    """CropMask node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CropMask node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import CropMask

        # Create node instance
        node = CropMask()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class MaskComposite(BaseNode):
    """MaskComposite node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the MaskComposite node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import MaskComposite

        # Create node instance
        node = MaskComposite()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class FeatherMask(BaseNode):
    """FeatherMask node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the FeatherMask node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import FeatherMask

        # Create node instance
        node = FeatherMask()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class GrowMask(BaseNode):
    """GrowMask node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the GrowMask node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import GrowMask

        # Create node instance
        node = GrowMask()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ThresholdMask(BaseNode):
    """ThresholdMask node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ThresholdMask node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import ThresholdMask

        # Create node instance
        node = ThresholdMask()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class MaskPreview(BaseNode):
    """
    Saves the input images to your ComfyUI output directory.
    
    Category: mask
    ComfyUI Node ID: MaskPreview
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the MaskPreview node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mask import MaskPreview

        # Create node instance
        node = MaskPreview()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyMochiLatentVideo(BaseNode):
    """EmptyMochiLatentVideo node from ComfyUI (category: latent/video)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyMochiLatentVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_mochi import EmptyMochiLatentVideo

        # Create node instance
        node = EmptyMochiLatentVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ModelSamplingDiscrete(BaseNode):
    """ModelSamplingDiscrete node from ComfyUI (category: advanced/model)"""

    model: Any = Field(default=None, description="model parameter")
    sampling: str = Field(default=None, description="sampling parameter")
    zsnr: bool = Field(default=False, description="zsnr parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelSamplingDiscrete node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_advanced import ModelSamplingDiscrete

        # Create node instance
        node = ModelSamplingDiscrete()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["sampling"] = self.sampling
        kwargs["zsnr"] = self.zsnr

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ModelSamplingContinuousEDM(BaseNode):
    """ModelSamplingContinuousEDM node from ComfyUI (category: advanced/model)"""

    model: Any = Field(default=None, description="model parameter")
    sampling: str = Field(default=None, description="sampling parameter")
    sigma_max: float = Field(default=120.0, description="sigma_max parameter", ge=0.0, le=1000.0)
    sigma_min: float = Field(default=0.002, description="sigma_min parameter", ge=0.0, le=1000.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelSamplingContinuousEDM node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_advanced import ModelSamplingContinuousEDM

        # Create node instance
        node = ModelSamplingContinuousEDM()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["sampling"] = self.sampling
        kwargs["sigma_max"] = self.sigma_max
        kwargs["sigma_min"] = self.sigma_min

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ModelSamplingContinuousV(BaseNode):
    """ModelSamplingContinuousV node from ComfyUI (category: advanced/model)"""

    model: Any = Field(default=None, description="model parameter")
    sampling: str = Field(default=None, description="sampling parameter")
    sigma_max: float = Field(default=500.0, description="sigma_max parameter", ge=0.0, le=1000.0)
    sigma_min: float = Field(default=0.03, description="sigma_min parameter", ge=0.0, le=1000.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelSamplingContinuousV node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_advanced import ModelSamplingContinuousV

        # Create node instance
        node = ModelSamplingContinuousV()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["sampling"] = self.sampling
        kwargs["sigma_max"] = self.sigma_max
        kwargs["sigma_min"] = self.sigma_min

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ModelSamplingStableCascade(BaseNode):
    """ModelSamplingStableCascade node from ComfyUI (category: advanced/model)"""

    model: Any = Field(default=None, description="model parameter")
    shift: float = Field(default=2.0, description="shift parameter", ge=0.0, le=100.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelSamplingStableCascade node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_advanced import ModelSamplingStableCascade

        # Create node instance
        node = ModelSamplingStableCascade()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["shift"] = self.shift

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ModelSamplingSD3(BaseNode):
    """ModelSamplingSD3 node from ComfyUI (category: advanced/model)"""

    model: Any = Field(default=None, description="model parameter")
    shift: float = Field(default=3.0, description="shift parameter", ge=0.0, le=100.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelSamplingSD3 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_advanced import ModelSamplingSD3

        # Create node instance
        node = ModelSamplingSD3()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["shift"] = self.shift

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ModelSamplingAuraFlow(BaseNode):
    """ModelSamplingAuraFlow node from ComfyUI (category: uncategorized)"""

    model: Any = Field(default=None, description="model parameter")
    shift: float = Field(default=1.73, description="shift parameter", ge=0.0, le=100.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelSamplingAuraFlow node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

        # Create node instance
        node = ModelSamplingAuraFlow()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["shift"] = self.shift

        # Call the node function
        result = node.patch_aura(**kwargs)

        # Return result
        return result


class ModelSamplingFlux(BaseNode):
    """ModelSamplingFlux node from ComfyUI (category: advanced/model)"""

    model: Any = Field(default=None, description="model parameter")
    max_shift: float = Field(default=1.15, description="max_shift parameter", ge=0.0, le=100.0)
    base_shift: float = Field(default=0.5, description="base_shift parameter", ge=0.0, le=100.0)
    width: int = Field(default=1024, description="width parameter", ge=16)
    height: int = Field(default=1024, description="height parameter", ge=16)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelSamplingFlux node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux

        # Create node instance
        node = ModelSamplingFlux()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["max_shift"] = self.max_shift
        kwargs["base_shift"] = self.base_shift
        kwargs["width"] = self.width
        kwargs["height"] = self.height

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class RescaleCFG(BaseNode):
    """RescaleCFG node from ComfyUI (category: advanced/model)"""

    model: Any = Field(default=None, description="model parameter")
    multiplier: float = Field(default=0.7, description="multiplier parameter", ge=0.0, le=1.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the RescaleCFG node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_advanced import RescaleCFG

        # Create node instance
        node = RescaleCFG()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["multiplier"] = self.multiplier

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ModelComputeDtype(BaseNode):
    """ModelComputeDtype node from ComfyUI (category: advanced/debug/model)"""

    model: Any = Field(default=None, description="model parameter")
    dtype: str = Field(default=None, description="dtype parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelComputeDtype node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_advanced import ModelComputeDtype

        # Create node instance
        node = ModelComputeDtype()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["dtype"] = self.dtype

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class PatchModelAddDownscale(BaseNode):
    """PatchModelAddDownscale node from ComfyUI (category: model_patches/unet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PatchModelAddDownscale node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_downscale import PatchModelAddDownscale

        # Create node instance
        node = PatchModelAddDownscale()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ModelMergeSimple(BaseNode):
    """ModelMergeSimple node from ComfyUI (category: advanced/model_merging)"""

    model1: Any = Field(default=None, description="model1 parameter")
    model2: Any = Field(default=None, description="model2 parameter")
    ratio: float = Field(default=1.0, description="ratio parameter", ge=0.0, le=1.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeSimple node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import ModelMergeSimple

        # Create node instance
        node = ModelMergeSimple()

        # Prepare inputs
        kwargs = {}
        kwargs["model1"] = self.model1
        kwargs["model2"] = self.model2
        kwargs["ratio"] = self.ratio

        # Call the node function
        result = node.merge(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ModelMergeBlocks(BaseNode):
    """ModelMergeBlocks node from ComfyUI (category: advanced/model_merging)"""

    model1: Any = Field(default=None, description="model1 parameter")
    model2: Any = Field(default=None, description="model2 parameter")
    input: float = Field(default=1.0, description="input parameter", ge=0.0, le=1.0)
    middle: float = Field(default=1.0, description="middle parameter", ge=0.0, le=1.0)
    out: float = Field(default=1.0, description="out parameter", ge=0.0, le=1.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeBlocks node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import ModelMergeBlocks

        # Create node instance
        node = ModelMergeBlocks()

        # Prepare inputs
        kwargs = {}
        kwargs["model1"] = self.model1
        kwargs["model2"] = self.model2
        kwargs["input"] = self.input
        kwargs["middle"] = self.middle
        kwargs["out"] = self.out

        # Call the node function
        result = node.merge(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ModelSubtract(BaseNode):
    """ModelMergeSubtract node from ComfyUI (category: advanced/model_merging)"""

    model1: Any = Field(default=None, description="model1 parameter")
    model2: Any = Field(default=None, description="model2 parameter")
    multiplier: float = Field(default=1.0, description="multiplier parameter", le=10.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeSubtract node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import ModelSubtract

        # Create node instance
        node = ModelSubtract()

        # Prepare inputs
        kwargs = {}
        kwargs["model1"] = self.model1
        kwargs["model2"] = self.model2
        kwargs["multiplier"] = self.multiplier

        # Call the node function
        result = node.merge(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ModelAdd(BaseNode):
    """ModelMergeAdd node from ComfyUI (category: advanced/model_merging)"""

    model1: Any = Field(default=None, description="model1 parameter")
    model2: Any = Field(default=None, description="model2 parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeAdd node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import ModelAdd

        # Create node instance
        node = ModelAdd()

        # Prepare inputs
        kwargs = {}
        kwargs["model1"] = self.model1
        kwargs["model2"] = self.model2

        # Call the node function
        result = node.merge(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class CheckpointSave(BaseNode):
    """CheckpointSave node from ComfyUI (category: advanced/model_merging)"""

    model: Any = Field(default=None, description="model parameter")
    clip: Any = Field(default=None, description="clip parameter")
    vae: Any = Field(default=None, description="vae parameter")
    filename_prefix: str = Field(default="checkpoints/ComfyUI", description="filename_prefix parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CheckpointSave node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import CheckpointSave

        # Create node instance
        node = CheckpointSave()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["clip"] = self.clip
        kwargs["vae"] = self.vae
        kwargs["filename_prefix"] = self.filename_prefix

        # Call the node function
        result = node.save(**kwargs)

        # Return result
        return result


class CLIPMergeSimple(BaseNode):
    """CLIPMergeSimple node from ComfyUI (category: advanced/model_merging)"""

    clip1: Any = Field(default=None, description="clip1 parameter")
    clip2: Any = Field(default=None, description="clip2 parameter")
    ratio: float = Field(default=1.0, description="ratio parameter", ge=0.0, le=1.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPMergeSimple node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import CLIPMergeSimple

        # Create node instance
        node = CLIPMergeSimple()

        # Prepare inputs
        kwargs = {}
        kwargs["clip1"] = self.clip1
        kwargs["clip2"] = self.clip2
        kwargs["ratio"] = self.ratio

        # Call the node function
        result = node.merge(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class CLIPSubtract(BaseNode):
    """CLIPMergeSubtract node from ComfyUI (category: advanced/model_merging)"""

    clip1: Any = Field(default=None, description="clip1 parameter")
    clip2: Any = Field(default=None, description="clip2 parameter")
    multiplier: float = Field(default=1.0, description="multiplier parameter", le=10.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPMergeSubtract node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import CLIPSubtract

        # Create node instance
        node = CLIPSubtract()

        # Prepare inputs
        kwargs = {}
        kwargs["clip1"] = self.clip1
        kwargs["clip2"] = self.clip2
        kwargs["multiplier"] = self.multiplier

        # Call the node function
        result = node.merge(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class CLIPAdd(BaseNode):
    """CLIPMergeAdd node from ComfyUI (category: advanced/model_merging)"""

    clip1: Any = Field(default=None, description="clip1 parameter")
    clip2: Any = Field(default=None, description="clip2 parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPMergeAdd node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import CLIPAdd

        # Create node instance
        node = CLIPAdd()

        # Prepare inputs
        kwargs = {}
        kwargs["clip1"] = self.clip1
        kwargs["clip2"] = self.clip2

        # Call the node function
        result = node.merge(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class CLIPSave(BaseNode):
    """CLIPSave node from ComfyUI (category: advanced/model_merging)"""

    clip: Any = Field(default=None, description="clip parameter")
    filename_prefix: str = Field(default="clip/ComfyUI", description="filename_prefix parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPSave node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import CLIPSave

        # Create node instance
        node = CLIPSave()

        # Prepare inputs
        kwargs = {}
        kwargs["clip"] = self.clip
        kwargs["filename_prefix"] = self.filename_prefix

        # Call the node function
        result = node.save(**kwargs)

        # Return result
        return result


class VAESave(BaseNode):
    """VAESave node from ComfyUI (category: advanced/model_merging)"""

    vae: Any = Field(default=None, description="vae parameter")
    filename_prefix: str = Field(default="vae/ComfyUI_vae", description="filename_prefix parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the VAESave node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import VAESave

        # Create node instance
        node = VAESave()

        # Prepare inputs
        kwargs = {}
        kwargs["vae"] = self.vae
        kwargs["filename_prefix"] = self.filename_prefix

        # Call the node function
        result = node.save(**kwargs)

        # Return result
        return result


class ModelSave(BaseNode):
    """ModelSave node from ComfyUI (category: advanced/model_merging)"""

    model: Any = Field(default=None, description="model parameter")
    filename_prefix: str = Field(default="diffusion_models/ComfyUI", description="filename_prefix parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelSave node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging import ModelSave

        # Create node instance
        node = ModelSave()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["filename_prefix"] = self.filename_prefix

        # Call the node function
        result = node.save(**kwargs)

        # Return result
        return result


class ModelMergeSD1(BaseNode):
    """ModelMergeSD1 node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeSD1 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeSD1

        # Create node instance
        node = ModelMergeSD1()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeSD1(BaseNode):
    """ModelMergeSD2 node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeSD2 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeSD1

        # Create node instance
        node = ModelMergeSD1()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeSDXL(BaseNode):
    """ModelMergeSDXL node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeSDXL node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeSDXL

        # Create node instance
        node = ModelMergeSDXL()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeSD3_2B(BaseNode):
    """ModelMergeSD3_2B node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeSD3_2B node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeSD3_2B

        # Create node instance
        node = ModelMergeSD3_2B()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeAuraflow(BaseNode):
    """ModelMergeAuraflow node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeAuraflow node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeAuraflow

        # Create node instance
        node = ModelMergeAuraflow()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeFlux1(BaseNode):
    """ModelMergeFlux1 node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeFlux1 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeFlux1

        # Create node instance
        node = ModelMergeFlux1()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeSD35_Large(BaseNode):
    """ModelMergeSD35_Large node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeSD35_Large node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeSD35_Large

        # Create node instance
        node = ModelMergeSD35_Large()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeMochiPreview(BaseNode):
    """ModelMergeMochiPreview node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeMochiPreview node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeMochiPreview

        # Create node instance
        node = ModelMergeMochiPreview()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeLTXV(BaseNode):
    """ModelMergeLTXV node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeLTXV node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeLTXV

        # Create node instance
        node = ModelMergeLTXV()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeCosmos7B(BaseNode):
    """ModelMergeCosmos7B node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeCosmos7B node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeCosmos7B

        # Create node instance
        node = ModelMergeCosmos7B()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeCosmos14B(BaseNode):
    """ModelMergeCosmos14B node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeCosmos14B node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeCosmos14B

        # Create node instance
        node = ModelMergeCosmos14B()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeWAN2_1(BaseNode):
    """
    1.3B model has 30 blocks, 14B model has 40 blocks. Image to video model has the extra img_emb.
    
    Category: advanced/model_merging/model_specific
    ComfyUI Node ID: ModelMergeWAN2_1
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeWAN2_1 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeWAN2_1

        # Create node instance
        node = ModelMergeWAN2_1()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeCosmosPredict2_2B(BaseNode):
    """ModelMergeCosmosPredict2_2B node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeCosmosPredict2_2B node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeCosmosPredict2_2B

        # Create node instance
        node = ModelMergeCosmosPredict2_2B()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeCosmosPredict2_14B(BaseNode):
    """ModelMergeCosmosPredict2_14B node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeCosmosPredict2_14B node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeCosmosPredict2_14B

        # Create node instance
        node = ModelMergeCosmosPredict2_14B()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelMergeQwenImage(BaseNode):
    """ModelMergeQwenImage node from ComfyUI (category: advanced/model_merging/model_specific)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelMergeQwenImage node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_merging_model_specific import ModelMergeQwenImage

        # Create node instance
        node = ModelMergeQwenImage()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ModelPatchLoader(BaseNode):
    """ModelPatchLoader node from ComfyUI (category: advanced/loaders)"""

    name: Any = Field(default=None, description="name parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ModelPatchLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_patch import ModelPatchLoader

        # Create node instance
        node = ModelPatchLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["name"] = self.name

        # Call the node function
        result = node.load_model_patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class QwenImageDiffsynthControlnet(BaseNode):
    """QwenImageDiffsynthControlnet node from ComfyUI (category: advanced/loaders/qwen)"""

    model: Any = Field(default=None, description="model parameter")
    model_patch: Any = Field(default=None, description="model_patch parameter")
    vae: Any = Field(default=None, description="vae parameter")
    image: Any = Field(default=None, description="image parameter")
    strength: float = Field(default=1.0, description="strength parameter", le=10.0)
    mask: Optional[Any] = Field(default=None, description="mask parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the QwenImageDiffsynthControlnet node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_patch import QwenImageDiffsynthControlnet

        # Create node instance
        node = QwenImageDiffsynthControlnet()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["model_patch"] = self.model_patch
        kwargs["vae"] = self.vae
        kwargs["image"] = self.image
        kwargs["strength"] = self.strength
        if self.mask is not None:
            kwargs["mask"] = self.mask

        # Call the node function
        result = node.diffsynth_controlnet(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ZImageFunControlnet(BaseNode):
    """ZImageFunControlnet node from ComfyUI (category: advanced/loaders/zimage)"""

    model: Any = Field(default=None, description="model parameter")
    model_patch: Any = Field(default=None, description="model_patch parameter")
    vae: Any = Field(default=None, description="vae parameter")
    strength: float = Field(default=1.0, description="strength parameter", le=10.0)
    image: Optional[Any] = Field(default=None, description="image parameter")
    inpaint_image: Optional[Any] = Field(default=None, description="inpaint_image parameter")
    mask: Optional[Any] = Field(default=None, description="mask parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ZImageFunControlnet node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_patch import ZImageFunControlnet

        # Create node instance
        node = ZImageFunControlnet()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["model_patch"] = self.model_patch
        kwargs["vae"] = self.vae
        kwargs["strength"] = self.strength
        if self.image is not None:
            kwargs["image"] = self.image
        if self.inpaint_image is not None:
            kwargs["inpaint_image"] = self.inpaint_image
        if self.mask is not None:
            kwargs["mask"] = self.mask

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class USOStyleReference(BaseNode):
    """USOStyleReference node from ComfyUI (category: advanced/model_patches/flux)"""

    model: Any = Field(default=None, description="model parameter")
    model_patch: Any = Field(default=None, description="model_patch parameter")
    clip_vision_output: Any = Field(default=None, description="clip_vision_output parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the USOStyleReference node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_model_patch import USOStyleReference

        # Create node instance
        node = USOStyleReference()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["model_patch"] = self.model_patch
        kwargs["clip_vision_output"] = self.clip_vision_output

        # Call the node function
        result = node.apply_patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class Morphology(BaseNode):
    """Morphology node from ComfyUI (category: image/postprocessing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Morphology node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_morphology import Morphology

        # Create node instance
        node = Morphology()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageRGBToYUV(BaseNode):
    """ImageRGBToYUV node from ComfyUI (category: image/batch)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageRGBToYUV node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_morphology import ImageRGBToYUV

        # Create node instance
        node = ImageRGBToYUV()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageYUVToRGB(BaseNode):
    """ImageYUVToRGB node from ComfyUI (category: image/batch)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageYUVToRGB node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_morphology import ImageYUVToRGB

        # Create node instance
        node = ImageYUVToRGB()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class wanBlockSwap(BaseNode):
    """
    NOP
    
    Category: uncategorized
    ComfyUI Node ID: wanBlockSwap
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the wanBlockSwap node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_nop import wanBlockSwap

        # Create node instance
        node = wanBlockSwap()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class OptimalStepsScheduler(BaseNode):
    """OptimalStepsScheduler node from ComfyUI (category: sampling/custom_sampling/schedulers)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the OptimalStepsScheduler node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_optimalsteps import OptimalStepsScheduler

        # Create node instance
        node = OptimalStepsScheduler()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class PerturbedAttentionGuidance(BaseNode):
    """PerturbedAttentionGuidance node from ComfyUI (category: model_patches/unet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PerturbedAttentionGuidance node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_pag import PerturbedAttentionGuidance

        # Create node instance
        node = PerturbedAttentionGuidance()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class PerpNeg(BaseNode):
    """PerpNeg node from ComfyUI (category: _for_testing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PerpNeg node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_perpneg import PerpNeg

        # Create node instance
        node = PerpNeg()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class PerpNegGuider(BaseNode):
    """PerpNegGuider node from ComfyUI (category: _for_testing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PerpNegGuider node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_perpneg import PerpNegGuider

        # Create node instance
        node = PerpNegGuider()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class PhotoMakerLoader(BaseNode):
    """PhotoMakerLoader node from ComfyUI (category: _for_testing/photomaker)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PhotoMakerLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_photomaker import PhotoMakerLoader

        # Create node instance
        node = PhotoMakerLoader()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class PhotoMakerEncode(BaseNode):
    """PhotoMakerEncode node from ComfyUI (category: _for_testing/photomaker)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PhotoMakerEncode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_photomaker import PhotoMakerEncode

        # Create node instance
        node = PhotoMakerEncode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodePixArtAlpha(BaseNode):
    """
    Encodes text and sets the resolution conditioning for PixArt Alpha. Does not apply to PixArt Sigma.
    
    Category: advanced/conditioning
    ComfyUI Node ID: CLIPTextEncodePixArtAlpha
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodePixArtAlpha node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_pixart import CLIPTextEncodePixArtAlpha

        # Create node instance
        node = CLIPTextEncodePixArtAlpha()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Blend(BaseNode):
    """ImageBlend node from ComfyUI (category: image/postprocessing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageBlend node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import Blend

        # Create node instance
        node = Blend()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Blur(BaseNode):
    """ImageBlur node from ComfyUI (category: image/postprocessing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageBlur node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import Blur

        # Create node instance
        node = Blur()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Quantize(BaseNode):
    """ImageQuantize node from ComfyUI (category: image/postprocessing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageQuantize node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import Quantize

        # Create node instance
        node = Quantize()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Sharpen(BaseNode):
    """ImageSharpen node from ComfyUI (category: image/postprocessing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageSharpen node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import Sharpen

        # Create node instance
        node = Sharpen()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageScaleToTotalPixels(BaseNode):
    """ImageScaleToTotalPixels node from ComfyUI (category: image/upscaling)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageScaleToTotalPixels node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels

        # Create node instance
        node = ImageScaleToTotalPixels()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ResizeImageMaskNode(BaseNode):
    """ResizeImageMaskNode node from ComfyUI (category: transform)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ResizeImageMaskNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import ResizeImageMaskNode

        # Create node instance
        node = ResizeImageMaskNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class BatchImagesNode(BaseNode):
    """BatchImagesNode node from ComfyUI (category: image)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the BatchImagesNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import BatchImagesNode

        # Create node instance
        node = BatchImagesNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class BatchMasksNode(BaseNode):
    """BatchMasksNode node from ComfyUI (category: mask)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the BatchMasksNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import BatchMasksNode

        # Create node instance
        node = BatchMasksNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class BatchLatentsNode(BaseNode):
    """BatchLatentsNode node from ComfyUI (category: latent)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the BatchLatentsNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import BatchLatentsNode

        # Create node instance
        node = BatchLatentsNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class BatchImagesMasksLatentsNode(BaseNode):
    """BatchImagesMasksLatentsNode node from ComfyUI (category: util)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the BatchImagesMasksLatentsNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_post_processing import BatchImagesMasksLatentsNode

        # Create node instance
        node = BatchImagesMasksLatentsNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class PreviewAny(BaseNode):
    """PreviewAny node from ComfyUI (category: utils)"""

    source: Any = Field(default=None, description="source parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the PreviewAny node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_preview_any import PreviewAny

        # Create node instance
        node = PreviewAny()

        # Prepare inputs
        kwargs = {}
        kwargs["source"] = self.source

        # Call the node function
        result = node.main(**kwargs)

        # Return result
        return result


class String(BaseNode):
    """PrimitiveString node from ComfyUI (category: utils/primitive)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PrimitiveString node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_primitive import String

        # Create node instance
        node = String()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StringMultiline(BaseNode):
    """PrimitiveStringMultiline node from ComfyUI (category: utils/primitive)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PrimitiveStringMultiline node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_primitive import StringMultiline

        # Create node instance
        node = StringMultiline()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Int(BaseNode):
    """PrimitiveInt node from ComfyUI (category: utils/primitive)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PrimitiveInt node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_primitive import Int

        # Create node instance
        node = Int()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Float(BaseNode):
    """PrimitiveFloat node from ComfyUI (category: utils/primitive)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PrimitiveFloat node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_primitive import Float

        # Create node instance
        node = Float()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Boolean(BaseNode):
    """PrimitiveBoolean node from ComfyUI (category: utils/primitive)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the PrimitiveBoolean node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_primitive import Boolean

        # Create node instance
        node = Boolean()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TextEncodeQwenImageEdit(BaseNode):
    """TextEncodeQwenImageEdit node from ComfyUI (category: advanced/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TextEncodeQwenImageEdit node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit

        # Create node instance
        node = TextEncodeQwenImageEdit()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TextEncodeQwenImageEditPlus(BaseNode):
    """TextEncodeQwenImageEditPlus node from ComfyUI (category: advanced/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TextEncodeQwenImageEditPlus node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus

        # Create node instance
        node = TextEncodeQwenImageEditPlus()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptyQwenImageLayeredLatentImage(BaseNode):
    """EmptyQwenImageLayeredLatentImage node from ComfyUI (category: latent/qwen)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptyQwenImageLayeredLatentImage node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_qwen import EmptyQwenImageLayeredLatentImage

        # Create node instance
        node = EmptyQwenImageLayeredLatentImage()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LatentRebatch(BaseNode):
    """RebatchLatents node from ComfyUI (category: latent/batch)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the RebatchLatents node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_rebatch import LatentRebatch

        # Create node instance
        node = LatentRebatch()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageRebatch(BaseNode):
    """RebatchImages node from ComfyUI (category: image/batch)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the RebatchImages node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_rebatch import ImageRebatch

        # Create node instance
        node = ImageRebatch()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ScaleROPE(BaseNode):
    """
    Scale and shift the ROPE of the model.
    
    Category: advanced/model_patches
    ComfyUI Node ID: ScaleROPE
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ScaleROPE node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_rope import ScaleROPE

        # Create node instance
        node = ScaleROPE()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SelfAttentionGuidance(BaseNode):
    """SelfAttentionGuidance node from ComfyUI (category: _for_testing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SelfAttentionGuidance node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_sag import SelfAttentionGuidance

        # Create node instance
        node = SelfAttentionGuidance()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TripleCLIPLoader(BaseNode):
    """
    [Recipes]

sd3: clip-l, clip-g, t5
    
    Category: advanced/loaders
    ComfyUI Node ID: TripleCLIPLoader
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TripleCLIPLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_sd3 import TripleCLIPLoader

        # Create node instance
        node = TripleCLIPLoader()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class EmptySD3LatentImage(BaseNode):
    """EmptySD3LatentImage node from ComfyUI (category: latent/sd3)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the EmptySD3LatentImage node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_sd3 import EmptySD3LatentImage

        # Create node instance
        node = EmptySD3LatentImage()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CLIPTextEncodeSD3(BaseNode):
    """CLIPTextEncodeSD3 node from ComfyUI (category: advanced/conditioning)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CLIPTextEncodeSD3 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_sd3 import CLIPTextEncodeSD3

        # Create node instance
        node = CLIPTextEncodeSD3()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ControlNetApplySD3(BaseNode):
    """ControlNetApplySD3 node from ComfyUI (category: conditioning/controlnet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ControlNetApplySD3 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_sd3 import ControlNetApplySD3

        # Create node instance
        node = ControlNetApplySD3()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SkipLayerGuidanceSD3(BaseNode):
    """
    Generic version of SkipLayerGuidance node that can be used on every DiT model.
    
    Category: advanced/guidance
    ComfyUI Node ID: SkipLayerGuidanceSD3
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SkipLayerGuidanceSD3 node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_sd3 import SkipLayerGuidanceSD3

        # Create node instance
        node = SkipLayerGuidanceSD3()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SD_4XUpscale_Conditioning(BaseNode):
    """SD_4XUpscale_Conditioning node from ComfyUI (category: conditioning/upscale_diffusion)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SD_4XUpscale_Conditioning node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_sdupscale import SD_4XUpscale_Conditioning

        # Create node instance
        node = SD_4XUpscale_Conditioning()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SkipLayerGuidanceDiT(BaseNode):
    """
    Generic version of SkipLayerGuidance node that can be used on every DiT model.
    
    Category: advanced/guidance
    ComfyUI Node ID: SkipLayerGuidanceDiT
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SkipLayerGuidanceDiT node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiT

        # Create node instance
        node = SkipLayerGuidanceDiT()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SkipLayerGuidanceDiTSimple(BaseNode):
    """
    Simple version of the SkipLayerGuidanceDiT node that only modifies the uncond pass.
    
    Category: advanced/guidance
    ComfyUI Node ID: SkipLayerGuidanceDiTSimple
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SkipLayerGuidanceDiTSimple node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

        # Create node instance
        node = SkipLayerGuidanceDiTSimple()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StableZero123_Conditioning(BaseNode):
    """StableZero123_Conditioning node from ComfyUI (category: conditioning/3d_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StableZero123_Conditioning node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_stable3d import StableZero123_Conditioning

        # Create node instance
        node = StableZero123_Conditioning()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StableZero123_Conditioning_Batched(BaseNode):
    """StableZero123_Conditioning_Batched node from ComfyUI (category: conditioning/3d_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StableZero123_Conditioning_Batched node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_stable3d import StableZero123_Conditioning_Batched

        # Create node instance
        node = StableZero123_Conditioning_Batched()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SV3D_Conditioning(BaseNode):
    """SV3D_Conditioning node from ComfyUI (category: conditioning/3d_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SV3D_Conditioning node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_stable3d import SV3D_Conditioning

        # Create node instance
        node = SV3D_Conditioning()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StableCascade_EmptyLatentImage(BaseNode):
    """StableCascade_EmptyLatentImage node from ComfyUI (category: latent/stable_cascade)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StableCascade_EmptyLatentImage node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_stable_cascade import StableCascade_EmptyLatentImage

        # Create node instance
        node = StableCascade_EmptyLatentImage()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StableCascade_StageC_VAEEncode(BaseNode):
    """StableCascade_StageC_VAEEncode node from ComfyUI (category: latent/stable_cascade)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StableCascade_StageC_VAEEncode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_stable_cascade import StableCascade_StageC_VAEEncode

        # Create node instance
        node = StableCascade_StageC_VAEEncode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StableCascade_StageB_Conditioning(BaseNode):
    """StableCascade_StageB_Conditioning node from ComfyUI (category: conditioning/stable_cascade)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StableCascade_StageB_Conditioning node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_stable_cascade import StableCascade_StageB_Conditioning

        # Create node instance
        node = StableCascade_StageB_Conditioning()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StableCascade_SuperResolutionControlnet(BaseNode):
    """StableCascade_SuperResolutionControlnet node from ComfyUI (category: _for_testing/stable_cascade)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StableCascade_SuperResolutionControlnet node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_stable_cascade import StableCascade_SuperResolutionControlnet

        # Create node instance
        node = StableCascade_SuperResolutionControlnet()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StringConcatenate(BaseNode):
    """StringConcatenate node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StringConcatenate node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import StringConcatenate

        # Create node instance
        node = StringConcatenate()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StringSubstring(BaseNode):
    """StringSubstring node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StringSubstring node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import StringSubstring

        # Create node instance
        node = StringSubstring()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StringLength(BaseNode):
    """StringLength node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StringLength node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import StringLength

        # Create node instance
        node = StringLength()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CaseConverter(BaseNode):
    """CaseConverter node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CaseConverter node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import CaseConverter

        # Create node instance
        node = CaseConverter()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StringTrim(BaseNode):
    """StringTrim node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StringTrim node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import StringTrim

        # Create node instance
        node = StringTrim()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StringReplace(BaseNode):
    """StringReplace node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StringReplace node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import StringReplace

        # Create node instance
        node = StringReplace()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StringContains(BaseNode):
    """StringContains node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StringContains node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import StringContains

        # Create node instance
        node = StringContains()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class StringCompare(BaseNode):
    """StringCompare node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the StringCompare node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import StringCompare

        # Create node instance
        node = StringCompare()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class RegexMatch(BaseNode):
    """RegexMatch node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the RegexMatch node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import RegexMatch

        # Create node instance
        node = RegexMatch()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class RegexExtract(BaseNode):
    """RegexExtract node from ComfyUI (category: utils/string)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the RegexExtract node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import RegexExtract

        # Create node instance
        node = RegexExtract()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class RegexReplace(BaseNode):
    """
    Find and replace text using regex patterns.
    
    Category: utils/string
    ComfyUI Node ID: RegexReplace
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the RegexReplace node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_string import RegexReplace

        # Create node instance
        node = RegexReplace()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TCFG(BaseNode):
    """
    TCFG – Tangential Damping CFG (2503.18137)

Refine the uncond (negative) to align with the cond (positive) for improving quality.
    
    Category: advanced/guidance
    ComfyUI Node ID: TCFG
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TCFG node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_tcfg import TCFG

        # Create node instance
        node = TCFG()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TomePatchModel(BaseNode):
    """TomePatchModel node from ComfyUI (category: model_patches/unet)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TomePatchModel node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_tomesd import TomePatchModel

        # Create node instance
        node = TomePatchModel()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TorchCompileModel(BaseNode):
    """TorchCompileModel node from ComfyUI (category: _for_testing)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TorchCompileModel node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_torch_compile import TorchCompileModel

        # Create node instance
        node = TorchCompileModel()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TrainLoraNode(BaseNode):
    """TrainLoraNode node from ComfyUI (category: training)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TrainLoraNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_train import TrainLoraNode

        # Create node instance
        node = TrainLoraNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LoraModelLoader(BaseNode):
    """LoraModelLoader node from ComfyUI (category: loaders)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoraModelLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_train import LoraModelLoader

        # Create node instance
        node = LoraModelLoader()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveLoRA(BaseNode):
    """SaveLoRA node from ComfyUI (category: loaders)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveLoRA node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_train import SaveLoRA

        # Create node instance
        node = SaveLoRA()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LossGraphNode(BaseNode):
    """LossGraphNode node from ComfyUI (category: training)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LossGraphNode node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_train import LossGraphNode

        # Create node instance
        node = LossGraphNode()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class UpscaleModelLoader(BaseNode):
    """UpscaleModelLoader node from ComfyUI (category: loaders)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the UpscaleModelLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_upscale_model import UpscaleModelLoader

        # Create node instance
        node = UpscaleModelLoader()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageUpscaleWithModel(BaseNode):
    """ImageUpscaleWithModel node from ComfyUI (category: image/upscaling)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageUpscaleWithModel node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

        # Create node instance
        node = ImageUpscaleWithModel()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveWEBM(BaseNode):
    """SaveWEBM node from ComfyUI (category: image/video)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveWEBM node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video import SaveWEBM

        # Create node instance
        node = SaveWEBM()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class SaveVideo(BaseNode):
    """
    Saves the input images to your ComfyUI output directory.
    
    Category: image/video
    ComfyUI Node ID: SaveVideo
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the SaveVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video import SaveVideo

        # Create node instance
        node = SaveVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class CreateVideo(BaseNode):
    """
    Create a video from images.
    
    Category: image/video
    ComfyUI Node ID: CreateVideo
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the CreateVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video import CreateVideo

        # Create node instance
        node = CreateVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class GetVideoComponents(BaseNode):
    """
    Extracts all components from a video: frames, audio, and framerate.
    
    Category: image/video
    ComfyUI Node ID: GetVideoComponents
    """


    async def process(self, context: ProcessingContext) -> Any:
        """Process the GetVideoComponents node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video import GetVideoComponents

        # Create node instance
        node = GetVideoComponents()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class LoadVideo(BaseNode):
    """LoadVideo node from ComfyUI (category: image/video)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the LoadVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video import LoadVideo

        # Create node instance
        node = LoadVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class ImageOnlyCheckpointLoader(BaseNode):
    """ImageOnlyCheckpointLoader node from ComfyUI (category: loaders/video_models)"""

    ckpt_name: Any = Field(default=None, description="ckpt_name parameter")

    async def process(self, context: ProcessingContext) -> tuple[Any, Any, Any]:
        """Process the ImageOnlyCheckpointLoader node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video_model import ImageOnlyCheckpointLoader

        # Create node instance
        node = ImageOnlyCheckpointLoader()

        # Prepare inputs
        kwargs = {}
        kwargs["ckpt_name"] = self.ckpt_name

        # Call the node function
        result = node.load_checkpoint(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class SVD_img2vid_Conditioning(BaseNode):
    """SVD_img2vid_Conditioning node from ComfyUI (category: conditioning/video_models)"""

    clip_vision: Any = Field(default=None, description="clip_vision parameter")
    init_image: Any = Field(default=None, description="init_image parameter")
    vae: Any = Field(default=None, description="vae parameter")
    width: int = Field(default=1024, description="width parameter", ge=16)
    height: int = Field(default=576, description="height parameter", ge=16)
    video_frames: int = Field(default=14, description="video_frames parameter", ge=1, le=4096)
    motion_bucket_id: int = Field(default=127, description="motion_bucket_id parameter", ge=1, le=1023)
    fps: int = Field(default=6, description="fps parameter", ge=1, le=1024)
    augmentation_level: float = Field(default=0.0, description="augmentation_level parameter", ge=0.0, le=10.0)

    async def process(self, context: ProcessingContext) -> tuple[Any, Any, Any]:
        """Process the SVD_img2vid_Conditioning node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video_model import SVD_img2vid_Conditioning

        # Create node instance
        node = SVD_img2vid_Conditioning()

        # Prepare inputs
        kwargs = {}
        kwargs["clip_vision"] = self.clip_vision
        kwargs["init_image"] = self.init_image
        kwargs["vae"] = self.vae
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["video_frames"] = self.video_frames
        kwargs["motion_bucket_id"] = self.motion_bucket_id
        kwargs["fps"] = self.fps
        kwargs["augmentation_level"] = self.augmentation_level

        # Call the node function
        result = node.encode(**kwargs)

        # Return result
        return result if isinstance(result, tuple) else (result,)


class VideoLinearCFGGuidance(BaseNode):
    """VideoLinearCFGGuidance node from ComfyUI (category: sampling/video_models)"""

    model: Any = Field(default=None, description="model parameter")
    min_cfg: float = Field(default=1.0, description="min_cfg parameter", ge=0.0, le=100.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the VideoLinearCFGGuidance node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video_model import VideoLinearCFGGuidance

        # Create node instance
        node = VideoLinearCFGGuidance()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["min_cfg"] = self.min_cfg

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class VideoTriangleCFGGuidance(BaseNode):
    """VideoTriangleCFGGuidance node from ComfyUI (category: sampling/video_models)"""

    model: Any = Field(default=None, description="model parameter")
    min_cfg: float = Field(default=1.0, description="min_cfg parameter", ge=0.0, le=100.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the VideoTriangleCFGGuidance node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video_model import VideoTriangleCFGGuidance

        # Create node instance
        node = VideoTriangleCFGGuidance()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["min_cfg"] = self.min_cfg

        # Call the node function
        result = node.patch(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class ImageOnlyCheckpointSave(BaseNode):
    """ImageOnlyCheckpointSave node from ComfyUI (category: advanced/model_merging)"""

    model: Any = Field(default=None, description="model parameter")
    clip_vision: Any = Field(default=None, description="clip_vision parameter")
    vae: Any = Field(default=None, description="vae parameter")
    filename_prefix: str = Field(default="checkpoints/ComfyUI", description="filename_prefix parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ImageOnlyCheckpointSave node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video_model import ImageOnlyCheckpointSave

        # Create node instance
        node = ImageOnlyCheckpointSave()

        # Prepare inputs
        kwargs = {}
        kwargs["model"] = self.model
        kwargs["clip_vision"] = self.clip_vision
        kwargs["vae"] = self.vae
        kwargs["filename_prefix"] = self.filename_prefix

        # Call the node function
        result = node.None(**kwargs)

        # Return result
        return result


class ConditioningSetAreaPercentageVideo(BaseNode):
    """ConditioningSetAreaPercentageVideo node from ComfyUI (category: conditioning)"""

    conditioning: Any = Field(default=None, description="conditioning parameter")
    width: float = Field(default=1.0, description="width parameter", ge=0, le=1.0)
    height: float = Field(default=1.0, description="height parameter", ge=0, le=1.0)
    temporal: float = Field(default=1.0, description="temporal parameter", ge=0, le=1.0)
    x: float = Field(default=0, description="x parameter", ge=0, le=1.0)
    y: float = Field(default=0, description="y parameter", ge=0, le=1.0)
    z: float = Field(default=0, description="z parameter", ge=0, le=1.0)
    strength: float = Field(default=1.0, description="strength parameter", ge=0.0, le=10.0)

    async def process(self, context: ProcessingContext) -> Any:
        """Process the ConditioningSetAreaPercentageVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_video_model import ConditioningSetAreaPercentageVideo

        # Create node instance
        node = ConditioningSetAreaPercentageVideo()

        # Prepare inputs
        kwargs = {}
        kwargs["conditioning"] = self.conditioning
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["temporal"] = self.temporal
        kwargs["x"] = self.x
        kwargs["y"] = self.y
        kwargs["z"] = self.z
        kwargs["strength"] = self.strength

        # Call the node function
        result = node.append(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result


class WanImageToVideo(BaseNode):
    """WanImageToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanImageToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanImageToVideo

        # Create node instance
        node = WanImageToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanFunControlToVideo(BaseNode):
    """WanFunControlToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanFunControlToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanFunControlToVideo

        # Create node instance
        node = WanFunControlToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Wan22FunControlToVideo(BaseNode):
    """Wan22FunControlToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Wan22FunControlToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import Wan22FunControlToVideo

        # Create node instance
        node = Wan22FunControlToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanFirstLastFrameToVideo(BaseNode):
    """WanFirstLastFrameToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanFirstLastFrameToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanFirstLastFrameToVideo

        # Create node instance
        node = WanFirstLastFrameToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanFunInpaintToVideo(BaseNode):
    """WanFunInpaintToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanFunInpaintToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanFunInpaintToVideo

        # Create node instance
        node = WanFunInpaintToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanVaceToVideo(BaseNode):
    """WanVaceToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanVaceToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanVaceToVideo

        # Create node instance
        node = WanVaceToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class TrimVideoLatent(BaseNode):
    """TrimVideoLatent node from ComfyUI (category: latent/video)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the TrimVideoLatent node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import TrimVideoLatent

        # Create node instance
        node = TrimVideoLatent()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanCameraImageToVideo(BaseNode):
    """WanCameraImageToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanCameraImageToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanCameraImageToVideo

        # Create node instance
        node = WanCameraImageToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanPhantomSubjectToVideo(BaseNode):
    """WanPhantomSubjectToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanPhantomSubjectToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanPhantomSubjectToVideo

        # Create node instance
        node = WanPhantomSubjectToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanTrackToVideo(BaseNode):
    """WanTrackToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanTrackToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanTrackToVideo

        # Create node instance
        node = WanTrackToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanSoundImageToVideo(BaseNode):
    """WanSoundImageToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanSoundImageToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanSoundImageToVideo

        # Create node instance
        node = WanSoundImageToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanSoundImageToVideoExtend(BaseNode):
    """WanSoundImageToVideoExtend node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanSoundImageToVideoExtend node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanSoundImageToVideoExtend

        # Create node instance
        node = WanSoundImageToVideoExtend()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanHuMoImageToVideo(BaseNode):
    """WanHuMoImageToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanHuMoImageToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanHuMoImageToVideo

        # Create node instance
        node = WanHuMoImageToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanAnimateToVideo(BaseNode):
    """WanAnimateToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanAnimateToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import WanAnimateToVideo

        # Create node instance
        node = WanAnimateToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class Wan22ImageToVideoLatent(BaseNode):
    """Wan22ImageToVideoLatent node from ComfyUI (category: conditioning/inpaint)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the Wan22ImageToVideoLatent node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wan import Wan22ImageToVideoLatent

        # Create node instance
        node = Wan22ImageToVideoLatent()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanMoveVisualizeTracks(BaseNode):
    """WanMoveVisualizeTracks node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanMoveVisualizeTracks node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wanmove import WanMoveVisualizeTracks

        # Create node instance
        node = WanMoveVisualizeTracks()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanMoveTracksFromCoords(BaseNode):
    """WanMoveTracksFromCoords node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanMoveTracksFromCoords node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wanmove import WanMoveTracksFromCoords

        # Create node instance
        node = WanMoveTracksFromCoords()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class GenerateTracks(BaseNode):
    """GenerateTracks node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the GenerateTracks node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wanmove import GenerateTracks

        # Create node instance
        node = GenerateTracks()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanMoveConcatTrack(BaseNode):
    """WanMoveConcatTrack node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanMoveConcatTrack node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wanmove import WanMoveConcatTrack

        # Create node instance
        node = WanMoveConcatTrack()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WanMoveTrackToVideo(BaseNode):
    """WanMoveTrackToVideo node from ComfyUI (category: conditioning/video_models)"""


    async def process(self, context: ProcessingContext) -> Any:
        """Process the WanMoveTrackToVideo node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_wanmove import WanMoveTrackToVideo

        # Create node instance
        node = WanMoveTrackToVideo()

        # Prepare inputs
        kwargs = {}

        # Call the node function
        result = node.process(**kwargs)

        # Return result
        return result


class WebcamCapture(BaseNode):
    """WebcamCapture node from ComfyUI (category: image)"""

    image: Any = Field(default=None, description="image parameter")
    width: int = Field(default=0, description="width parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    height: int = Field(default=0, description="height parameter", ge=0, le={'_ref': 'MAX_RESOLUTION'})
    capture_on_queue: bool = Field(default=True, description="capture_on_queue parameter")

    async def process(self, context: ProcessingContext) -> Any:
        """Process the WebcamCapture node."""
        # Import the ComfyUI node class
        from comfy_extras.nodes_webcam import WebcamCapture

        # Create node instance
        node = WebcamCapture()

        # Prepare inputs
        kwargs = {}
        kwargs["image"] = self.image
        kwargs["width"] = self.width
        kwargs["height"] = self.height
        kwargs["capture_on_queue"] = self.capture_on_queue

        # Call the node function
        result = node.load_capture(**kwargs)

        # Return result
        return result[0] if isinstance(result, tuple) else result

