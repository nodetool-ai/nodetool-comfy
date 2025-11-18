from __future__ import annotations

import asyncio
from typing import List, Tuple

import comfy
import comfy.sd
import folder_paths
from comfy.model_patcher import ModelPatcher
from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from custom_nodes.ComfyUI_GGUF.loader import gguf_sd_loader
from custom_nodes.ComfyUI_GGUF.ops import GGMLOps
from custom_nodes.ComfyUI_GGUF.nodes import GGUFModelPatcher
from huggingface_hub import try_to_load_from_cache
from nodes import (
    CLIPTextEncode,
    EmptyLatentImage,
    KSampler,
    LatentUpscale,
    LoraLoader,
    VAEDecodeTiled,
)
from nodetool.ml.core.model_manager import ModelManager
from nodetool.metadata.types import HFTextToImage, HFUnet, ImageRef, LoRAConfig
from nodetool.nodes.comfy.constants import (
    FLUX_CLIP_L,
    FLUX_CLIP_T5XXL,
    FLUX_DEV_MODELS,
    FLUX_SCHNELL_MODELS,
    FLUX_VAE,
    HF_STABLE_DIFFUSION_MODELS,
    HF_STABLE_DIFFUSION_XL_MODELS,
)
from nodetool.nodes.comfy.enums import Sampler, Scheduler
from nodetool.nodes.comfy.utils import comfy_progress
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


def _load_flux_gguf_unet(ckpt_path: str) -> ModelPatcher:
    ops = GGMLOps()
    sd = gguf_sd_loader(ckpt_path)
    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": ops}
    )
    if model is None:
        raise RuntimeError(
            f"Could not detect model type for GGUF checkpoint: {ckpt_path}"
        )

    model = GGUFModelPatcher.clone(model)
    model.patch_on_device = False
    return model


class StableDiffusion(BaseNode):
    """
    Generates images from text prompts using Stable Diffusion.
    image, text-to-image, generative AI, stable diffusion, SD1.5

    This node is strictly text-to-image: it never consumes an input image.
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(), description="The model to use."
    )
    prompt: str = Field(default="", description="The prompt to use.")
    negative_prompt: str = Field(default="", description="The negative prompt to use.")
    seed: int = Field(default=0, ge=0, le=1000000)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=30.0)
    num_inference_steps: int = Field(default=30, ge=1, le=100)
    width: int = Field(default=768, ge=64, le=2048, multiple_of=64)
    height: int = Field(default=768, ge=64, le=2048, multiple_of=64)
    scheduler: Scheduler = Field(default=Scheduler.exponential)
    sampler: Sampler = Field(default=Sampler.euler_ancestral)
    loras: List[LoRAConfig] = Field(
        default=[],
        description="List of LoRA models to apply",
    )
    _model: ModelPatcher | None = None
    _clip: comfy.sd.CLIP | None = None
    _vae: comfy.sd.VAE | None = None

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
            "loras",
        ]

    def apply_loras(
        self,
        unet: ModelPatcher,
        clip: comfy.sd.CLIP,
    ) -> Tuple[ModelPatcher, comfy.sd.CLIP]:
        for lora_config in self.loras:
            unet, clip = LoraLoader().load_lora(
                unet,
                clip,
                lora_config.lora.name,
                lora_config.strength,
                lora_config.strength,
            )  # type: ignore[assignment]
        return unet, clip

    def get_empty_latent(self, width: int, height: int):
        return EmptyLatentImage().generate(width, height, 1)[0]

    def get_conditioning(self, clip: comfy.sd.CLIP) -> Tuple[list, list]:
        positive_conditioning = CLIPTextEncode().encode(clip, self.prompt)[0]
        negative_conditioning = CLIPTextEncode().encode(clip, self.negative_prompt)[0]
        return positive_conditioning, negative_conditioning

    def sample(self, model, latent, positive, negative, num_steps):
        return KSampler().sample(
            model=model,
            seed=self.seed,
            steps=num_steps,
            cfg=self.guidance_scale,
            sampler_name=self.sampler.value,
            scheduler=self.scheduler.value,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0,
        )[0]

    async def preload_model(self, context: ProcessingContext):
        if self.model.is_empty():
            raise ValueError("Model repository ID must be selected.")

        assert self.model.path is not None, "Model path must be set."

        self._model = ModelManager.get_model(self.model.repo_id, "unet", self.model.path)
        self._clip = ModelManager.get_model(self.model.repo_id, "clip", self.model.path)
        self._vae = ModelManager.get_model(self.model.repo_id, "vae", self.model.path)

        if self._model and self._clip and self._vae:
            return

        cache_path = try_to_load_from_cache(self.model.repo_id, self.model.path)
        if cache_path is None:
            raise ValueError(
                f"Model checkpoint not found for {self.model.repo_id}/{self.model.path}"
            )

        self._model, self._clip, self._vae, _ = comfy.sd.load_checkpoint_guess_config(
            cache_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        assert self._model is not None, "UNet must be loaded."
        assert self._clip is not None, "CLIP must be loaded."
        assert self._vae is not None, "VAE must be loaded."

        ModelManager.set_model(
            self.id,
            self.model.repo_id,
            "unet",
            self._model,
            self.model.path,
        )
        ModelManager.set_model(
            self.id,
            self.model.repo_id,
            "clip",
            self._clip,
            self.model.path,
        )
        ModelManager.set_model(
            self.id,
            self.model.repo_id,
            "vae",
            self._vae,
            self.model.path,
        )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._model is None or self._clip is None or self._vae is None:
            raise RuntimeError("Model components must be loaded before processing.")

        with comfy_progress(context, self, self._model):
            unet = ModelPatcher.clone(self._model)
            clip = self._clip.clone()
            vae = self._vae

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

            latent = self.get_empty_latent(initial_width, initial_height)

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

            decoded_image = VAEDecodeTiled().decode(vae, sampled_latent, 512)[0]
            return await context.image_from_tensor(decoded_image)


class StableDiffusionXL(StableDiffusion):
    """
    Generates images from text prompts using Stable Diffusion XL.
    image, text-to-image, generative AI, SDXL

    Text-only SDXL variant; no input image.
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(), description="The model to use."
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return HF_STABLE_DIFFUSION_XL_MODELS


class StableDiffusion3(StableDiffusion):
    """
    Generates images from text prompts using Stable Diffusion 3.5.
    image, text-to-image, generative AI, SD3.5
    """

    guidance_scale: float = Field(default=4.0, ge=1.0, le=30.0)
    num_inference_steps: int = Field(default=20, ge=1, le=100)

    @classmethod
    def get_title(cls) -> str:
        return "Stable Diffusion 3.5"

    @classmethod
    def get_recommended_models(cls) -> list[HFTextToImage]:
        return [
            HFTextToImage(
                repo_id="Comfy-Org/stable-diffusion-3.5-fp8",
                path="sd3.5_large_fp8_scaled.safetensors",
            ),
        ]

    def get_empty_latent(self, width: int, height: int):
        return EmptySD3LatentImage().generate(width, height, 1)[0]



class Flux(BaseNode):
    """
    Generates images from text prompts using the Flux model.
    image, text-to-image, generative AI, flux

    ComfyUI-native Flux implementation in the comfy.text_to_image namespace.
    """

    model: HFTextToImage = Field(
        default=HFTextToImage(), description="The model to use."
    )
    prompt: str = Field(default="", description="The prompt to use.")
    negative_prompt: str = Field(default="", description="The negative prompt to use.")
    width: int = Field(default=1024, ge=64, le=2048, multiple_of=64)
    height: int = Field(default=1024, ge=64, le=2048, multiple_of=64)
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=30.0)
    seed: int = Field(default=0, ge=0, le=1000000000)
    denoise: float = Field(default=1.0, ge=0.0, le=1.0)
    scheduler: Scheduler = Field(default=Scheduler.simple)
    sampler: Sampler = Field(default=Sampler.euler)
    _model: ModelPatcher | None = None
    _clip: comfy.sd.CLIP | None = None
    _vae: comfy.sd.VAE | None = None

    @classmethod
    def get_title(cls) -> str:
        return "Flux"

    @classmethod
    def get_recommended_models(cls) -> list[HFUnet | HFTextToImage]:
        return (
            [FLUX_VAE, FLUX_CLIP_L, FLUX_CLIP_T5XXL]
            + FLUX_DEV_MODELS
            + FLUX_SCHNELL_MODELS
        )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "steps",
        ]

    async def preload_model(self, context: ProcessingContext):
        if self.model.is_empty():
            raise ValueError("Model must be selected.")

        assert self.model.path is not None, "Model path must be set."

        self._model = ModelManager.get_model(
            self.model.repo_id, "unet", self.model.path
        )
        self._clip = ModelManager.get_model(self.model.repo_id, "clip", self.model.path)
        self._vae = ModelManager.get_model(self.model.repo_id, "vae", self.model.path)

        if self._model and self._clip and self._vae:
            return

        cache_path = try_to_load_from_cache(self.model.repo_id, self.model.path)

        if cache_path is not None:
            if self.model.path.lower().endswith(".gguf"):
                # GGUF Flux models are UNet-only; load the UNet through ComfyUI-GGUF
                # and let the CLIP/VAE loading logic fall back to the recommended
                # Flux components below.
                self._model = _load_flux_gguf_unet(cache_path)
                self._clip = None
                self._vae = None
            else:
                # First, try to load UNet, CLIP, and VAE directly from the checkpoint.
                self._model, self._clip, self._vae, _ = (
                    comfy.sd.load_checkpoint_guess_config(
                        cache_path,
                        output_vae=True,
                        output_clip=True,
                        embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    )
                )

        # If CLIP or VAE are not present in the checkpoint, fall back to the
        # default Flux CLIP and VAE weights.
        if self._clip is None or self._vae is None:
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

            self._clip = comfy.sd.load_clip(
                ckpt_paths=[clip_l_path, clip_t5xxl_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.FLUX,
            )

            sd = comfy.utils.load_torch_file(vae_path)
            self._vae = comfy.sd.VAE(sd=sd)

        if self._model:
            ModelManager.set_model(
                self.id,
                self.model.repo_id,
                "unet",
                self._model,
                self.model.path,
            )
        if self._clip:
            ModelManager.set_model(
                self.id,
                self.model.repo_id,
                "clip",
                self._clip,
                self.model.path,
            )
        if self._vae:
            ModelManager.set_model(
                self.id,
                self.model.repo_id,
                "vae",
                self._vae,
                self.model.path,
            )

    async def process(self, context: ProcessingContext) -> ImageRef:
        assert self._model is not None, "Model must be loaded."
        assert self._clip is not None, "CLIP must be loaded."
        assert self._vae is not None, "VAE must be loaded."

        with comfy_progress(context, self, self._model):
            latent = EmptySD3LatentImage().generate(self.width, self.height, 1)[0]

            positive = CLIPTextEncode().encode(self._clip, self.prompt)[0]
            negative = CLIPTextEncode().encode(self._clip, self.negative_prompt)[0]

            positive = FluxGuidance().append(positive, self.guidance_scale)[0]

            if "schnell" in self.model.repo_id:
                steps = 4
            else:
                steps = self.steps

            sampled_latent = KSampler().sample(
                model=self._model,
                seed=self.seed,
                steps=steps,
                cfg=self.guidance_scale,
                sampler_name=self.sampler.value,
                scheduler=self.scheduler.value,
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=self.denoise,
            )[0]

            decoded_image = VAEDecodeTiled().decode(self._vae, sampled_latent, 512)[0]
            return await context.image_from_tensor(decoded_image)


__all__ = [
    "StableDiffusion",
    "StableDiffusion3",
    "StableDiffusionXL",
    "Flux",
    "FluxFP8",
]


if __name__ == "__main__":

    async def main():
        node = Flux(
            model=HFTextToImage(
                repo_id="Comfy-Org/flux1-dev",
                path="flux1-dev-fp8.safetensors",
            ),
            prompt="A beautiful sunset over a calm ocean",
            negative_prompt="A dark and stormy ocean",
            width=1024,
            height=1024,
        )
        context = ProcessingContext()
        return await node.process(context)

    asyncio.run(main())
