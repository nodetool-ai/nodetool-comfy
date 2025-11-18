from __future__ import annotations

from typing import List

import PIL.Image
import comfy
import comfy.sd
import folder_paths
import numpy as np
import torch
from comfy_extras.nodes_flux import FluxGuidance, FluxKontextImageScale
from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from comfy_extras.nodes_edit_model import ReferenceLatent
from huggingface_hub import try_to_load_from_cache
from nodes import (
    CLIPTextEncode,
    ControlNetApply,
    ControlNetLoader,
    EmptyLatentImage,
    KSampler,
    LatentUpscale,
    VAEDecode,
    VAEDecodeTiled,
    VAEEncode,
    VAEEncodeForInpaint,
    ConditioningZeroOut,
)
from nodetool.metadata.types import (
    HFControlNet,
    HFImageToImage,
    HFTextToImage,
    HFUnet,
    ImageRef,
    LoRAConfig,
)
from nodetool.nodes.comfy.constants import (
    FLUX_DEV_MODELS,
    FLUX_SCHNELL_MODELS,
    HF_CONTROLNET_MODELS,
    HF_STABLE_DIFFUSION_MODELS,
    HF_STABLE_DIFFUSION_XL_MODELS,
    FLUX_CLIP_L,
    FLUX_CLIP_T5XXL,
    FLUX_VAE,
)
from nodetool.nodes.comfy.enums import Sampler, Scheduler
from nodetool.nodes.comfy.utils import comfy_progress, unload_comfy_model
from nodetool.nodes.comfy.text_to_image import (
    StableDiffusion as TextToImageSD,
    _load_flux_gguf_unet,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


class StableDiffusionImageToImage(TextToImageSD):
    """
    Transforms existing images based on text prompts using Stable Diffusion.
    image, image-to-image, generative AI, stable diffusion, SD1.5

    This node is strictly image-to-image: it requires an input image.
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(), description="The model to use."
    )
    input_image: ImageRef = Field(
        default=ImageRef(), description="Input image for img2img"
    )
    mask_image: ImageRef = Field(
        default=ImageRef(), description="Mask image for img2img (optional)"
    )
    grow_mask_by: int = Field(default=6, ge=0, le=100)
    denoise: float = Field(default=1.0, ge=0.0, le=1.0)
    loras: List[LoRAConfig] = Field(
        default=[],
        description="List of LoRA models to apply",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return HF_STABLE_DIFFUSION_MODELS

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "negative_prompt",
            "seed",
            "guidance_scale",
            "num_inference_steps",
            "width",
            "height",
            "scheduler",
            "sampler",
            "input_image",
            "mask_image",
            "grow_mask_by",
            "denoise",
            "loras",
        ]

    def required_inputs(self) -> list[str]:
        return ["input_image"]

    async def get_latent(
        self,
        vae: comfy.sd.VAE,
        context: ProcessingContext,
        width: int,
        height: int,
    ):
        if self.input_image.is_empty():
            raise ValueError("input_image must be provided for image-to-image.")

        input_pil = await context.image_to_pil(self.input_image)
        input_pil = input_pil.resize((width, height), PIL.Image.Resampling.LANCZOS)
        image = input_pil.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(image)[None,]

        if not self.mask_image.is_empty():
            mask_pil = await context.image_to_pil(self.mask_image)
            mask_pil = mask_pil.resize((width, height), PIL.Image.Resampling.LANCZOS)
            mask = mask_pil.convert("L")
            mask = np.array(mask).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask)[None,]
            return VAEEncodeForInpaint().encode(
                vae, input_tensor, mask_tensor, self.grow_mask_by
            )[0]

        return VAEEncode().encode(vae, input_tensor)[0]

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.model.is_empty():
            raise ValueError("Model repository ID must be selected.")

        assert self.model.path is not None, "Model path must be set."

        ckpt_path = try_to_load_from_cache(self.model.repo_id, self.model.path)

        unet, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        assert unet is not None, "UNet must be loaded."
        assert clip is not None, "CLIP must be loaded."
        assert vae is not None, "VAE must be loaded."

        with comfy_progress(context, self, unet):
            unet, clip = self.apply_loras(unet, clip)
            positive_conditioning, negative_conditioning = self.get_conditioning(clip)

            if (
                isinstance(self.model, HFTextToImage)
                and self.width >= 1024
                and self.height >= 1024
            ):
                num_lowres_steps = self.num_inference_steps // 4
                num_hires_steps = self.num_inference_steps - num_lowres_steps
                initial_width, initial_height = self.width // 2, self.height // 2
            else:
                num_hires_steps = 0
                num_lowres_steps = self.num_inference_steps
                initial_width, initial_height = self.width, self.height

            latent = await self.get_latent(vae, context, initial_width, initial_height)

            sampled_latent = self.sample(
                unet,
                latent,
                positive_conditioning,
                negative_conditioning,
                num_lowres_steps,
            )

            if num_hires_steps > 0:
                hires_latent = LatentUpscale().upscale(
                    samples=sampled_latent,
                    upscale_method="bilinear",
                    width=self.width,
                    height=self.height,
                    crop=False,
                )[0]

                sampled_latent = self.sample(
                    unet,
                    hires_latent,
                    positive_conditioning,
                    negative_conditioning,
                    num_hires_steps,
                )

            unload_comfy_model(unet)

            decoded_image = VAEDecodeTiled().decode(vae, sampled_latent, 512)[0]
            return await context.image_from_tensor(decoded_image)


class StableDiffusionXLImageToImage(StableDiffusionImageToImage):
    """
    Transforms existing images based on text prompts using Stable Diffusion XL.
    image, image-to-image, generative AI, SDXL
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(), description="The model to use."
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return HF_STABLE_DIFFUSION_XL_MODELS


class StableDiffusion3ImageToImage(StableDiffusionImageToImage):
    """
    Transforms existing images based on text prompts using Stable Diffusion 3.5.
    image, image-to-image, generative AI, SD3.5
    """

    guidance_scale: float = Field(default=4.0, ge=1.0, le=30.0)
    num_inference_steps: int = Field(default=20, ge=1, le=100)

    @classmethod
    def get_title(cls) -> str:
        return "Stable Diffusion 3.5 (Img2Img)"

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return [
            HFTextToImage(
                repo_id="Comfy-Org/stable-diffusion-3.5-fp8",
                path="sd3.5_large_fp8_scaled.safetensors",
            ),
        ]

    async def get_latent(
        self,
        vae: comfy.sd.VAE,
        context: ProcessingContext,
        width: int,
        height: int,
    ):
        # SD3 uses a different latent distribution; fall back to empty latent
        # if no image is provided, but still require an input image by default.
        if self.input_image.is_empty():
            return EmptySD3LatentImage().generate(width, height, 1)[0]
        return await super().get_latent(vae, context, width, height)


class FluxKontext(BaseNode):
    """
    Transforms existing images based on text prompts using the Flux Kontext model.
    image, image-to-image, generative AI, flux, kontext
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(),
        description="The Flux UNet checkpoint to use (e.g. flux1-dev or flux1-dev-kontext).",
    )
    input_image: ImageRef = Field(
        default=ImageRef(),
        description="Reference image for Flux Kontext img2img.",
    )
    prompt: str = Field(default="", description="The prompt to use.")
    negative_prompt: str = Field(
        default="", description="The negative prompt to use (optional)."
    )
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=1.0, ge=1.0, le=30.0)
    seed: int = Field(default=0, ge=0, le=1000000000)
    denoise: float = Field(default=1.0, ge=0.0, le=1.0)
    scheduler: Scheduler = Field(default=Scheduler.simple)
    sampler: Sampler = Field(default=Sampler.euler)

    @classmethod
    def get_title(cls) -> str:
        return "Flux Kontext (Img2Img)"

    @classmethod
    def get_recommended_models(cls) -> list[HFUnet | HFTextToImage]:
        # Mirror the Flux text-to-image node: recommend Flux UNets plus CLIP and VAE.
        return [
            HFImageToImage(
                repo_id="Comfy-Org/flux1-kontext-dev_ComfyUI",
                path="split_files/diffusion_modelsflux1-kontext-dev_ComfyUI.safetensors",
            )
        ] + [FLUX_VAE, FLUX_CLIP_L, FLUX_CLIP_T5XXL]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "input_image",
            "prompt",
            "negative_prompt",
            "steps",
            "guidance_scale",
            "seed",
            "denoise",
            "scheduler",
            "sampler",
        ]

    def required_inputs(self) -> list[str]:
        return ["input_image"]

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.model.is_empty():
            raise ValueError("Model must be selected.")

        if self.input_image.is_empty():
            raise ValueError("input_image must be provided for Flux Kontext img2img.")

        assert self.model.path is not None, "Model path must be set."

        ckpt_path = try_to_load_from_cache(self.model.repo_id, self.model.path)
        assert (
            ckpt_path is not None
        ), "Flux model checkpoint not found. Download from Recommended Models."

        assert FLUX_CLIP_L.path is not None, "CLIP model path must be set."
        clip_l_path = try_to_load_from_cache(FLUX_CLIP_L.repo_id, FLUX_CLIP_L.path)
        assert (
            clip_l_path is not None
        ), "CLIP model checkpoint not found. Download from Recommended Models."

        assert FLUX_CLIP_T5XXL.path is not None, "CLIP model path must be set."
        clip_t5xxl_path = try_to_load_from_cache(
            FLUX_CLIP_T5XXL.repo_id, FLUX_CLIP_T5XXL.path
        )
        assert (
            clip_t5xxl_path is not None
        ), "Second CLIP model checkpoint not found. Download from Recommended Models."

        assert FLUX_VAE.path is not None, "VAE model path must be set."
        vae_path = try_to_load_from_cache(FLUX_VAE.repo_id, FLUX_VAE.path)
        assert (
            vae_path is not None
        ), "VAE model checkpoint not found. Download from Recommended Models."

        ckpt_path = str(ckpt_path)

        # Load UNet, CLIP, and VAE following the Flux text-to-image setup.
        if ckpt_path.lower().endswith(".gguf"):
            model = _load_flux_gguf_unet(ckpt_path)
        else:
            model = comfy.sd.load_unet(ckpt_path)

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_l_path, clip_t5xxl_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.FLUX,
        )

        sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)

        with comfy_progress(context, self, model):
            # Convert input image to tensor (BHWC in [0,1]) and scale it with FluxKontextImageScale.
            input_pil = await context.image_to_pil(self.input_image)
            image = np.array(input_pil.convert("RGB")).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image)[None,]

            scaled_image = FluxKontextImageScale().scale(image_tensor)[0]

            # Encode the scaled image into a latent using the Flux VAE.
            latent = VAEEncode().encode(vae, scaled_image)[0]

            # Positive / negative conditioning from CLIP.
            positive = CLIPTextEncode().encode(clip, self.prompt)[0]

            if self.negative_prompt:
                negative = CLIPTextEncode().encode(clip, self.negative_prompt)[0]
            else:
                # Match the reference graph: derive negative conditioning by zeroing out the positive.
                negative = ConditioningZeroOut().zero_out(positive)[0]

            # Attach the image latent as a reference latent, then apply Flux guidance.
            positive = ReferenceLatent().execute(positive, latent)[0]
            positive = FluxGuidance().append(positive, self.guidance_scale)[0]

            # KSampler performs the Flux sampling using the image-derived latent.
            sampled_latent = KSampler().sample(
                model=model,
                seed=self.seed,
                steps=self.steps,
                cfg=self.guidance_scale,
                sampler_name=self.sampler.value,
                scheduler=self.scheduler.value,
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=self.denoise,
            )[0]

            unload_comfy_model(model)

            decoded_image = VAEDecodeTiled().decode(vae, sampled_latent, 512)[0]
            return await context.image_from_tensor(decoded_image)


class ControlNet(StableDiffusionImageToImage):
    """
    Generates images using Stable Diffusion with ControlNet for additional image control.
    image, controlnet, generative, stable diffusion, high-resolution, SD

    This node makes it explicit that it consumes an input image and a separate
    control image.
    """

    controlnet: HFControlNet = Field(
        default=HFControlNet(), description="The ControlNet model to use."
    )
    image: ImageRef = Field(
        default=ImageRef(), description="Canny edge detection image for ControlNet"
    )
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Strength of ControlNet (used for both low and high resolution)",
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFControlNet]:
        return HF_CONTROLNET_MODELS

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return super().get_basic_fields() + [
            "controlnet",
            "image",
            "strength",
        ]

    def required_inputs(self) -> list[str]:
        return ["input_image", "image"]

    async def apply_controlnet(
        self,
        context: ProcessingContext,
        width: int,
        height: int,
        conditioning: list,
    ):
        if self.controlnet.is_empty():
            raise ValueError("ControlNet repository ID must be selected.")

        assert self.controlnet.path is not None, "ControlNet path must be set."

        controlnet_path = try_to_load_from_cache(
            self.controlnet.repo_id, self.controlnet.path
        )

        if controlnet_path is None:
            raise ValueError(
                "ControlNet checkpoint not found. Download from Recommended Models."
            )

        controlnet = ControlNetLoader().load_controlnet(controlnet_path)[0]
        if not self.image.is_empty():
            pil = await context.image_to_pil(self.image)
            pil = pil.resize((width, height), PIL.Image.Resampling.LANCZOS)
            image = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
            tensor = torch.from_numpy(image)[None,]
            conditioning = ControlNetApply().apply_controlnet(
                conditioning,
                controlnet,
                tensor,
                self.strength,
            )[0]

        return conditioning

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self.model.is_empty():
            raise ValueError("Model repository ID must be selected.")

        assert self.model.path is not None, "Model path must be set."

        ckpt_path = try_to_load_from_cache(self.model.repo_id, self.model.path)

        unet, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        assert unet is not None, "UNet must be loaded."
        assert clip is not None, "CLIP must be loaded."
        assert vae is not None, "VAE must be loaded."

        with comfy_progress(context, self, unet):
            positive_conditioning, negative_conditioning = self.get_conditioning(clip)

            if self.width >= 1024 and self.height >= 1024:
                num_hires_steps = self.num_inference_steps // 2
                num_lowres_steps = self.num_inference_steps - num_hires_steps
                initial_width, initial_height = self.width // 2, self.height // 2
            else:
                num_hires_steps = 0
                num_lowres_steps = self.num_inference_steps
                initial_width, initial_height = self.width, self.height

            latent = await self.get_latent(vae, context, initial_width, initial_height)

            positive_conditioning_with_controlnet = await self.apply_controlnet(
                context, initial_width, initial_height, positive_conditioning
            )

            sampled_latent = self.sample(
                unet,
                latent,
                positive_conditioning_with_controlnet,
                negative_conditioning,
                num_lowres_steps,
            )

            if num_hires_steps > 0:
                hires_latent = LatentUpscale().upscale(
                    samples=sampled_latent,
                    upscale_method="bilinear",
                    width=self.width,
                    height=self.height,
                    crop=False,
                )[0]

                hires_positive_conditioning = await self.apply_controlnet(
                    context, self.width, self.height, positive_conditioning
                )

                sampled_latent = self.sample(
                    unet,
                    hires_latent,
                    hires_positive_conditioning,
                    negative_conditioning,
                    num_hires_steps,
                )

            decoded_image = VAEDecode().decode(vae, sampled_latent)[0]
            return await context.image_from_tensor(decoded_image)


__all__ = [
    "StableDiffusionImageToImage",
    "StableDiffusion3ImageToImage",
    "StableDiffusionXLImageToImage",
    "FluxKontext",
    "ControlNet",
]
