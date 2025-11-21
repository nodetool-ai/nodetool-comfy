from __future__ import annotations

import asyncio
from typing import List, Tuple

import comfy
import comfy.sd
import comfy.utils
import folder_paths
import numpy as np
import torch
from comfy.model_patcher import ModelPatcher
from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit
from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from custom_nodes.gguf.loader import gguf_sd_loader
from custom_nodes.gguf.ops import GGMLOps
from custom_nodes.gguf.nodes import GGUFModelPatcher
from huggingface_hub import try_to_load_from_cache
from nodes import (
    CLIPTextEncode,
    EmptyLatentImage,
    KSampler,
    LatentUpscale,
    LoraLoader,
    VAEEncode,
    VAEDecodeTiled,
)
from nodetool.ml.core.model_manager import ModelManager
from nodetool.metadata.types import (
    HFCLIP,
    HFQwenImage,
    HFQwenImageEdit,
    HFTextToImage,
    HFUnet,
    HFVAE,
    ImageRef,
    LoRAConfig,
)
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

        self._model = ModelManager.get_model(
            self.model.repo_id, "unet", self.model.path
        )
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

    unet_model: HFTextToImage = Field(
        default=HFTextToImage(), description="The UNet/diffusion model to use."
    )
    clip_model: HFCLIP = Field(
        default=HFCLIP(
            repo_id=FLUX_CLIP_L.repo_id,
            path=FLUX_CLIP_L.path,
        ),
        description="The primary Flux CLIP checkpoint (clip-l).",
    )
    clip_model_secondary: HFCLIP = Field(
        default=HFCLIP(
            repo_id=FLUX_CLIP_T5XXL.repo_id,
            path=FLUX_CLIP_T5XXL.path,
        ),
        description="The secondary Flux CLIP checkpoint (t5xxl).",
    )
    vae_model: HFVAE = Field(
        default=HFVAE(repo_id=FLUX_VAE.repo_id, path=FLUX_VAE.path),
        description="The Flux VAE checkpoint.",
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
            "unet_model",
            "clip_model",
            "clip_model_secondary",
            "vae_model",
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "steps",
        ]

    async def preload_model(self, context: ProcessingContext):
        if self.unet_model.is_empty():
            raise ValueError("UNet model must be selected.")
        if self.clip_model.is_empty() or self.clip_model_secondary.is_empty():
            raise ValueError("Both Flux CLIP models must be selected.")
        if self.vae_model.is_empty():
            raise ValueError("VAE model must be selected.")

        assert self.unet_model.path is not None, "UNet path must be set."
        assert self.clip_model.path is not None, "CLIP primary path must be set."
        assert self.clip_model_secondary.path is not None, "CLIP secondary path must be set."
        assert self.vae_model.path is not None, "VAE path must be set."

        self._model = ModelManager.get_model(
            self.unet_model.repo_id, "unet", self.unet_model.path
        )
        self._clip = ModelManager.get_model(
            self.clip_model.repo_id, "clip", self.clip_model.path
        )
        self._vae = ModelManager.get_model(
            self.vae_model.repo_id, "vae", self.vae_model.path
        )

        if self._model and self._clip and self._vae:
            return

        cache_path = try_to_load_from_cache(self.unet_model.repo_id, self.unet_model.path)

        if cache_path is not None:
            if self.unet_model.path.lower().endswith(".gguf"):
                self._model = _load_flux_gguf_unet(cache_path)
                self._clip = None
                self._vae = None
            else:
                self._model, self._clip, self._vae, _ = (
                    comfy.sd.load_checkpoint_guess_config(
                        cache_path,
                        output_vae=True,
                        output_clip=True,
                        embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    )
                )

        def _resolve_clip(path: str | None, repo_id: str):
            if path is None:
                return None
            cp = try_to_load_from_cache(repo_id, path)
            if cp:
                return cp
            lp = folder_paths.get_full_path("text_encoders", path)
            return lp

        if self._clip is None:
            clip_l_path = _resolve_clip(self.clip_model.path, self.clip_model.repo_id)
            clip_t5_path = _resolve_clip(
                self.clip_model_secondary.path, self.clip_model_secondary.repo_id
            )
            if clip_l_path is None or clip_t5_path is None:
                raise ValueError("CLIP model checkpoint(s) not found. Download from Recommended Models.")
            self._clip = comfy.sd.load_clip(
                ckpt_paths=[clip_l_path, clip_t5_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.FLUX,
            )

        if self._vae is None:
            vae_path = try_to_load_from_cache(self.vae_model.repo_id, self.vae_model.path)
            if vae_path is None:
                vae_path = folder_paths.get_full_path("vae", self.vae_model.path or "")
            if vae_path is None:
                raise ValueError("VAE model checkpoint not found. Download from Recommended Models.")
            sd = comfy.utils.load_torch_file(vae_path)
            self._vae = comfy.sd.VAE(sd=sd)

        if self._model:
            ModelManager.set_model(
                self.id,
                self.unet_model.repo_id,
                "unet",
                self._model,
                self.unet_model.path,
            )
        if self._clip:
            ModelManager.set_model(
                self.id,
                self.clip_model.repo_id,
                "clip",
                self._clip,
                self.clip_model.path,
            )
        if self._vae:
            ModelManager.set_model(
                self.id,
                self.vae_model.repo_id,
                "vae",
                self._vae,
                self.vae_model.path,
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

            repo_id = self.unet_model.repo_id or ""
            steps = 4 if "schnell" in repo_id else self.steps

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


class QwenImage(BaseNode):
    """
    Generates images from text prompts using Qwen-Image.
    image, text-to-image, generative AI, qwen
    """

    unet_model: HFQwenImage = Field(
        default=HFQwenImage(
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            path="non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors",
        ),
        description="The Qwen-Image UNet/diffusion checkpoint.",
    )
    clip_model: HFCLIP = Field(
        default=HFCLIP(
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            path="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        ),
        description="The Qwen-Image CLIP/text encoder checkpoint.",
    )
    vae_model: HFVAE = Field(
        default=HFVAE(
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            path="split_files/vae/qwen_image_vae.safetensors",
        ),
        description="The Qwen-Image VAE checkpoint.",
    )
    prompt: str = Field(default="", description="The prompt to use.")
    negative_prompt: str = Field(default="", description="The negative prompt to use.")
    true_cfg_scale: float = Field(default=1.0, ge=0.0, le=10.0)
    steps: int = Field(default=30, ge=1, le=100)
    width: int = Field(default=1024, ge=64, le=2048, multiple_of=64)
    height: int = Field(default=1024, ge=64, le=2048, multiple_of=64)
    scheduler: Scheduler = Field(default=Scheduler.simple)
    sampler: Sampler = Field(default=Sampler.euler)
    seed: int = Field(default=0, ge=0, le=1000000000)
    denoise: float = Field(default=1.0, ge=0.0, le=1.0)
    loras: List[LoRAConfig] = Field(
        default=[], description="List of LoRA models to apply."
    )

    _model: ModelPatcher | None = None
    _clip: comfy.sd.CLIP | None = None
    _vae: comfy.sd.VAE | None = None

    @classmethod
    def get_title(cls) -> str:
        return "Qwen-Image"

    @classmethod
    def get_recommended_models(cls) -> list[HFQwenImage | HFCLIP | HFVAE]:
        return [
            HFQwenImage(
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                path="non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors",
            ),
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q4_K_M.gguf",
            ),
            HFQwenImage(
                repo_id="city96/Qwen-Image-gguf",
                path="qwen-image-Q8_0.gguf",
            ),
            HFCLIP(
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                path="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            ),
            HFVAE(
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                path="split_files/vae/qwen_image_vae.safetensors",
            ),
        ]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "unet_model",
            "clip_model",
            "vae_model",
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "steps",
            "true_cfg_scale",
            "seed",
            "sampler",
            "scheduler",
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

    def get_empty_latent(self) -> dict:
        return EmptySD3LatentImage().generate(self.width, self.height, 1)[0]

    def get_conditioning(self, clip: comfy.sd.CLIP) -> Tuple[list, list]:
        positive_conditioning = CLIPTextEncode().encode(clip, self.prompt)[0]
        negative_conditioning = CLIPTextEncode().encode(clip, self.negative_prompt)[0]
        # Qwen models support a guidance embedding similar to Flux.
        positive_conditioning = FluxGuidance().append(
            positive_conditioning, self.true_cfg_scale
        )[0]
        return positive_conditioning, negative_conditioning

    def sample(self, model, latent, positive, negative, num_steps):
        return KSampler().sample(
            model=model,
            seed=self.seed,
            steps=num_steps,
            cfg=self.true_cfg_scale,
            sampler_name=self.sampler.value,
            scheduler=self.scheduler.value,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=self.denoise,
        )[0]

    def _cached_or_local(self, repo_id: str, paths: tuple[str, ...], folder: str):
        for candidate in paths:
            cp = try_to_load_from_cache(repo_id, candidate)
            if cp:
                return cp
        for candidate in paths:
            lp = folder_paths.get_full_path(folder, candidate)
            if lp:
                return lp
        return None

    def _load_clip(self) -> comfy.sd.CLIP:
        assert self.clip_model.path is not None, "CLIP path must be set."
        clip_path = self._cached_or_local(
            self.clip_model.repo_id,
            (
                self.clip_model.path,
                "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            ),
            "text_encoders",
        )
        if clip_path is None:
            raise ValueError("CLIP checkpoint not found.")
        return comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        )

    def _load_vae(self) -> comfy.sd.VAE:
        assert self.vae_model.path is not None, "VAE path must be set."
        vae_path = self._cached_or_local(
            self.vae_model.repo_id,
            (
                self.vae_model.path,
                "split_files/vae/qwen_image_vae.safetensors",
                "vae/qwen_image_vae.safetensors",
                "qwen_image_vae.safetensors",
            ),
            "vae",
        )
        if vae_path is None:
            raise ValueError("VAE checkpoint not found.")
        sd = comfy.utils.load_torch_file(vae_path)
        return comfy.sd.VAE(sd=sd)

    async def preload_model(self, context: ProcessingContext):
        if self.unet_model.is_empty():
            raise ValueError("UNet model must be selected.")
        if self.clip_model.is_empty():
            raise ValueError("CLIP model must be selected.")
        if self.vae_model.is_empty():
            raise ValueError("VAE model must be selected.")

        assert self.unet_model.path is not None, "UNet path must be set."
        assert self.clip_model.path is not None, "CLIP path must be set."
        assert self.vae_model.path is not None, "VAE path must be set."

        self._model = ModelManager.get_model(
            self.unet_model.repo_id, "unet", self.unet_model.path
        )
        self._clip = ModelManager.get_model(
            self.clip_model.repo_id, "clip", self.clip_model.path
        )
        self._vae = ModelManager.get_model(
            self.vae_model.repo_id, "vae", self.vae_model.path
        )

        if self._model and self._clip and self._vae:
            return

        cache_path = try_to_load_from_cache(self.unet_model.repo_id, self.unet_model.path)
        if cache_path is None:
            raise ValueError(
                f"Model checkpoint not found for {self.unet_model.repo_id}/{self.unet_model.path}"
            )

        if self.unet_model.path.lower().endswith(".gguf"):
            self._model = _load_flux_gguf_unet(cache_path)
        else:
            self._model, self._clip, self._vae, _ = comfy.sd.load_checkpoint_guess_config(
                cache_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )

        if self._clip is None:
            self._clip = self._load_clip()

        if self._vae is None:
            self._vae = self._load_vae()

        assert self._model is not None, "UNet must be loaded."
        assert self._clip is not None, "CLIP must be loaded."
        assert self._vae is not None, "VAE must be loaded."

        ModelManager.set_model(
            self.id,
            self.unet_model.repo_id,
            "unet",
            self._model,
            self.unet_model.path,
        )
        ModelManager.set_model(
            self.id,
            self.clip_model.repo_id,
            "clip",
            self._clip,
            self.clip_model.path,
        )
        ModelManager.set_model(
            self.id,
            self.vae_model.repo_id,
            "vae",
            self._vae,
            self.vae_model.path,
        )

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._model is None or self._clip is None or self._vae is None:
            raise RuntimeError("Model components must be loaded before processing.")

        with comfy_progress(context, self, self._model):
            unet = ModelPatcher.clone(self._model)
            clip = self._clip.clone()
            unet, clip = self.apply_loras(unet, clip)

            latent = self.get_empty_latent()

            positive_conditioning, negative_conditioning = self.get_conditioning(clip)

            sampled_latent = self.sample(
                unet,
                latent,
                positive_conditioning,
                negative_conditioning,
                self.steps,
            )

            decoded_image = VAEDecodeTiled().decode(self._vae, sampled_latent, 512)[0]
            return await context.image_from_tensor(decoded_image)


class QwenImageEdit(BaseNode):
    """
    Performs image editing using Qwen-Image-Edit with reference image conditioning.
    image, image-editing, generative AI, qwen
    """

    unet_model: HFQwenImageEdit = Field(
        default=HFQwenImageEdit(
            repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI",
            path="split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
        ),
        description="The Qwen-Image-Edit UNet checkpoint.",
    )
    clip_model: HFCLIP = Field(
        default=HFCLIP(
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            path="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        ),
        description="The Qwen-Image-Edit CLIP/text encoder checkpoint.",
    )
    vae_model: HFVAE = Field(
        default=HFVAE(
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            path="split_files/vae/qwen_image_vae.safetensors",
        ),
        description="The Qwen-Image-Edit VAE checkpoint.",
    )
    input_image: ImageRef = Field(
        default=ImageRef(), description="The reference image to edit."
    )
    prompt: str = Field(default="", description="Editing prompt.")
    negative_prompt: str = Field(default="", description="Negative prompt.")
    true_cfg_scale: float = Field(default=1.0, ge=0.0, le=10.0)
    steps: int = Field(default=20, ge=1, le=100)
    sampler: Sampler = Field(default=Sampler.euler)
    scheduler: Scheduler = Field(default=Scheduler.simple)
    seed: int = Field(default=0, ge=0, le=1000000000)
    denoise: float = Field(default=1.0, ge=0.0, le=1.0)
    loras: List[LoRAConfig] = Field(
        default=[], description="List of LoRA models to apply."
    )

    _model: ModelPatcher | None = None
    _clip: comfy.sd.CLIP | None = None
    _vae: comfy.sd.VAE | None = None

    @classmethod
    def get_title(cls) -> str:
        return "Qwen-Image-Edit"

    @classmethod
    def get_recommended_models(cls) -> list[HFQwenImageEdit | HFCLIP | HFVAE]:
        return [
            HFQwenImageEdit(
                repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI",
                path="split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
            ),
            HFCLIP(
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                path="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            ),
            HFVAE(
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                path="split_files/vae/qwen_image_vae.safetensors",
            ),
        ]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "unet_model",
            "clip_model",
            "vae_model",
            "input_image",
            "prompt",
            "negative_prompt",
            "steps",
            "true_cfg_scale",
            "sampler",
            "scheduler",
            "seed",
        ]

    def required_inputs(self) -> list[str]:
        return ["input_image"]

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

    def _cached_or_local(self, repo_id: str, paths: tuple[str, ...], folder: str):
        for candidate in paths:
            cp = try_to_load_from_cache(repo_id, candidate)
            if cp:
                return cp
        for candidate in paths:
            lp = folder_paths.get_full_path(folder, candidate)
            if lp:
                return lp
        return None

    def _load_clip(self) -> comfy.sd.CLIP:
        assert self.clip_model.path is not None, "CLIP path must be set."
        clip_path = self._cached_or_local(
            self.clip_model.repo_id,
            (
                self.clip_model.path,
                "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            ),
            "text_encoders",
        )
        if clip_path is None:
            raise ValueError("CLIP checkpoint not found.")
        return comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        )

    def _load_vae(self) -> comfy.sd.VAE:
        assert self.vae_model.path is not None, "VAE path must be set."
        vae_path = self._cached_or_local(
            self.vae_model.repo_id,
            (
                self.vae_model.path,
                "split_files/vae/qwen_image_vae.safetensors",
                "vae/qwen_image_vae.safetensors",
                "qwen_image_vae.safetensors",
            ),
            "vae",
        )
        if vae_path is None:
            raise ValueError("VAE checkpoint not found.")
        sd = comfy.utils.load_torch_file(vae_path)
        return comfy.sd.VAE(sd=sd)

    async def preload_model(self, context: ProcessingContext):
        if self.unet_model.is_empty():
            raise ValueError("UNet model must be selected.")
        if self.clip_model.is_empty():
            raise ValueError("CLIP model must be selected.")
        if self.vae_model.is_empty():
            raise ValueError("VAE model must be selected.")

        assert self.unet_model.path is not None, "UNet path must be set."
        assert self.clip_model.path is not None, "CLIP path must be set."
        assert self.vae_model.path is not None, "VAE path must be set."

        self._model = ModelManager.get_model(
            self.unet_model.repo_id, "unet", self.unet_model.path
        )
        self._clip = ModelManager.get_model(
            self.clip_model.repo_id, "clip", self.clip_model.path
        )
        self._vae = ModelManager.get_model(
            self.vae_model.repo_id, "vae", self.vae_model.path
        )

        if self._model and self._clip and self._vae:
            return

        cache_path = try_to_load_from_cache(self.unet_model.repo_id, self.unet_model.path)
        if cache_path is None:
            raise ValueError(
                f"Model checkpoint not found for {self.unet_model.repo_id}/{self.unet_model.path}"
            )

        if self.unet_model.path.lower().endswith(".gguf"):
            self._model = _load_flux_gguf_unet(cache_path)
        else:
            self._model, self._clip, self._vae, _ = comfy.sd.load_checkpoint_guess_config(
                cache_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )

        if self._clip is None:
            self._clip = self._load_clip()

        if self._vae is None:
            self._vae = self._load_vae()

        assert self._model is not None, "UNet must be loaded."
        assert self._clip is not None, "CLIP must be loaded."
        assert self._vae is not None, "VAE must be loaded."

        ModelManager.set_model(
            self.id,
            self.unet_model.repo_id,
            "unet",
            self._model,
            self.unet_model.path,
        )
        ModelManager.set_model(
            self.id,
            self.clip_model.repo_id,
            "clip",
            self._clip,
            self.clip_model.path,
        )
        ModelManager.set_model(
            self.id,
            self.vae_model.repo_id,
            "vae",
            self._vae,
            self.vae_model.path,
        )

    async def _image_to_tensor(self, context: ProcessingContext) -> torch.Tensor:
        if self.input_image.is_empty():
            raise ValueError("input_image must be provided.")
        pil_image = await context.image_to_pil(self.input_image)
        image = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(image)[None, ...]

    def get_conditioning(
        self, clip: comfy.sd.CLIP, vae: comfy.sd.VAE, image_tensor: torch.Tensor
    ) -> Tuple[list, list]:
        pos = TextEncodeQwenImageEdit().execute(
            clip=clip, prompt=self.prompt, vae=vae, image=image_tensor
        )[0]
        neg = TextEncodeQwenImageEdit().execute(
            clip=clip, prompt=self.negative_prompt, vae=vae, image=image_tensor
        )[0]
        pos = FluxGuidance().append(pos, self.true_cfg_scale)[0]
        return pos, neg

    def encode_latent(self, vae: comfy.sd.VAE, image_tensor: torch.Tensor) -> dict:
        return VAEEncode().encode(vae, image_tensor)[0]

    def sample(self, model, latent, positive, negative, num_steps):
        return KSampler().sample(
            model=model,
            seed=self.seed,
            steps=num_steps,
            cfg=self.true_cfg_scale,
            sampler_name=self.sampler.value,
            scheduler=self.scheduler.value,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=self.denoise,
        )[0]

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._model is None or self._clip is None or self._vae is None:
            raise RuntimeError("Model components must be loaded before processing.")

        with comfy_progress(context, self, self._model):
            unet = ModelPatcher.clone(self._model)
            clip = self._clip.clone()
            vae = self._vae
            unet, clip = self.apply_loras(unet, clip)

            image_tensor = await self._image_to_tensor(context)
            latent = self.encode_latent(vae, image_tensor)
            positive_conditioning, negative_conditioning = self.get_conditioning(
                clip, vae, image_tensor
            )

            sampled_latent = self.sample(
                unet,
                latent,
                positive_conditioning,
                negative_conditioning,
                self.steps,
            )

            decoded_image = VAEDecodeTiled().decode(vae, sampled_latent, 512)[0]
            return await context.image_from_tensor(decoded_image)


__all__ = [
    "StableDiffusion",
    "StableDiffusion3",
    "StableDiffusionXL",
    "Flux",
    "QwenImage",
    "QwenImageEdit",
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
