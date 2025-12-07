from __future__ import annotations

import asyncio
from contextlib import nullcontext
from typing import List, Tuple

import comfy
import comfy.sd
import comfy.utils
import comfy.model_management
import folder_paths
import numpy as np
import torch
from comfy.model_patcher import ModelPatcher
from comfy_extras.nodes_flux import FluxGuidance
from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit
from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from nodetool.integrations.huggingface.huggingface_models import HF_FAST_CACHE
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
    HFT5,
    HFFluxCheckpoint,
    HFFluxFP8Checkpoint,
    HFQwenImageCheckpoint,
    HFQwenImageEditCheckpoint,
    HFQwenVL,
    HFStableDiffusion3Checkpoint,
    HFStableDiffusionCheckpoint,
    HFStableDiffusionXLCheckpoint,
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


class StableDiffusion(BaseNode):
    """
    Generates images from text prompts using Stable Diffusion.
    image, text-to-image, generative AI, stable diffusion, SD1.5

    This node is strictly text-to-image: it never consumes an input image.
    """

    model: HFStableDiffusionCheckpoint = Field(
        default=HFStableDiffusionCheckpoint(), description="The model to use."
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
    def get_recommended_models(cls) -> list[HFStableDiffusionCheckpoint]:
        return HF_STABLE_DIFFUSION_MODELS  # type: ignore[return-value]

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

        cache_path = await HF_FAST_CACHE.resolve(self.model.repo_id, self.model.path)
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

        with torch.inference_mode():
            with comfy_progress(context, self, self._model):
                unet = ModelPatcher.clone(self._model)
                clip = self._clip.clone()
                vae = self._vae

                unet, clip = self.apply_loras(unet, clip)
                positive_conditioning, negative_conditioning = self.get_conditioning(
                    clip
                )

                if (
                    isinstance(self.model, HFStableDiffusionCheckpoint)
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

    model: HFStableDiffusionXLCheckpoint = Field(
        default=HFStableDiffusionXLCheckpoint(), description="The model to use."
    )

    @classmethod
    def get_recommended_models(cls) -> list[HFStableDiffusionXLCheckpoint]:
        return HF_STABLE_DIFFUSION_XL_MODELS  # type: ignore[return-value]


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
    def get_recommended_models(cls) -> list[HFStableDiffusion3Checkpoint]:
        return [
            HFStableDiffusion3Checkpoint(
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

    model: HFFluxCheckpoint = Field(
        default=HFFluxCheckpoint(
            repo_id="Comfy-Org/flux1-dev",
            path="flux1-dev-fp8.safetensors",
        ),
        description="The UNet/diffusion model to use.",
    )
    text_encoder: HFCLIP = Field(
        default=HFCLIP(
            repo_id=FLUX_CLIP_L.repo_id,
            path=FLUX_CLIP_L.path,
        ),
        description="The primary Flux CLIP checkpoint (clip-l).",
    )
    t5_model: HFT5 = Field(
        default=HFT5(
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
    def get_recommended_models(cls) -> list[HFFluxCheckpoint | HFCLIP | HFVAE]:
        return (
            [FLUX_VAE, FLUX_CLIP_L, FLUX_CLIP_T5XXL]
            + FLUX_DEV_MODELS
            + FLUX_SCHNELL_MODELS
        )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "text_encoder",
            "t5_model",
            "vae_model",
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "steps",
        ]

    async def preload_model(self, context: ProcessingContext):
        if self.model.is_empty():
            raise ValueError("UNet model must be selected.")
        if self.text_encoder.is_empty() or self.t5_model.is_empty():
            raise ValueError("Both Flux CLIP models must be selected.")
        if self.vae_model.is_empty():
            raise ValueError("VAE model must be selected.")

        assert self.model.path is not None, "UNet path must be set."
        assert self.text_encoder.path is not None, "CLIP primary path must be set."
        assert self.t5_model.path is not None, "T5 path must be set."
        assert self.vae_model.path is not None, "VAE path must be set."

        self._model = ModelManager.get_model(
            self.model.repo_id, "unet", self.model.path
        )
        self._clip = ModelManager.get_model(
            self.text_encoder.repo_id, "clip", self.text_encoder.path
        )
        self._vae = ModelManager.get_model(
            self.vae_model.repo_id, "vae", self.vae_model.path
        )

        # Older cached VAE instances may just be placeholders without weights.
        if getattr(self._vae, "first_stage_model", None) is None:
            self._vae = None

        if self._model and self._clip and self._vae:
            return

        cache_path = await HF_FAST_CACHE.resolve(self.model.repo_id, self.model.path)

        if cache_path is not None:
            self._model, self._clip, self._vae, _ = (
                comfy.sd.load_checkpoint_guess_config(
                    cache_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                )
            )

        async def _resolve_clip(path: str | None, repo_id: str):
            if path is None:
                return None
            cp = await HF_FAST_CACHE.resolve(repo_id, path)
            if cp:
                return cp
            lp = folder_paths.get_full_path("text_encoders", path)
            return lp

        if self._clip is None:
            clip_l_path = await _resolve_clip(self.text_encoder.path, self.text_encoder.repo_id)
            clip_t5_path = await _resolve_clip(
                self.t5_model.path, self.t5_model.repo_id
            )
            if clip_l_path is None or clip_t5_path is None:
                raise ValueError("CLIP model checkpoint(s) not found. Download from Recommended Models.")
            self._clip = comfy.sd.load_clip(
                ckpt_paths=[clip_l_path, clip_t5_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.FLUX,
            )

        if self._vae is None:
            vae_path = await HF_FAST_CACHE.resolve(self.vae_model.repo_id, self.vae_model.path)
            if vae_path is None:
                vae_path = folder_paths.get_full_path("vae", self.vae_model.path or "")
            if vae_path is None:
                raise ValueError("VAE model checkpoint not found. Download from Recommended Models.")
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
                self.text_encoder.repo_id,
                "clip",
                self._clip,
                self.text_encoder.path,
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

        with torch.inference_mode():
            with comfy_progress(context, self, self._model):
                latent = EmptySD3LatentImage().generate(self.width, self.height, 1)[0]

                positive = CLIPTextEncode().encode(self._clip, self.prompt)[0]
                negative = CLIPTextEncode().encode(self._clip, self.negative_prompt)[0]

                positive = FluxGuidance().append(positive, self.guidance_scale)[0]

                repo_id = self.model.repo_id or ""
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

                decoded_image = VAEDecodeTiled().decode(
                    self._vae, sampled_latent, 512
                )[0]
                return await context.image_from_tensor(decoded_image)


class FluxFP8(BaseNode):
    """
    Single-checkpoint Flux fp8 pipeline (clip/vae bundled).
    image, text-to-image, generative AI, flux
    """

    model: HFFluxFP8Checkpoint = Field(
        default=HFFluxFP8Checkpoint(
            repo_id="Comfy-Org/flux1-dev",
            path="flux1-dev-fp8.safetensors",
        ),
        description="Flux fp8 checkpoint (clip/vae bundled).",
    )   
    prompt: str = Field(default="", description="The prompt to use.")
    negative_prompt: str = Field(default="", description="The negative prompt to use.")
    width: int = Field(default=1024, ge=64, le=2048, multiple_of=64)
    height: int = Field(default=1024, ge=64, le=2048, multiple_of=64)
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=3.5, ge=0.0, le=30.0)
    sampler_cfg: float = Field(default=1.0, ge=0.0, le=30.0)
    seed: int = Field(default=0, ge=0, le=1000000000)
    denoise: float = Field(default=1.0, ge=0.0, le=1.0)
    scheduler: Scheduler = Field(default=Scheduler.simple)
    sampler: Sampler = Field(default=Sampler.euler)
    loras: List[LoRAConfig] = Field(default=[], description="List of LoRA models to apply.")

    _model: ModelPatcher | None = None
    _clip: comfy.sd.CLIP | None = None
    _vae: comfy.sd.VAE | None = None

    @classmethod
    def get_title(cls) -> str:
        return "Flux FP8"

    @classmethod
    def get_recommended_models(cls) -> list[HFFluxFP8Checkpoint]:
        return [
            HFFluxFP8Checkpoint(repo_id="Comfy-Org/flux1-dev", path="flux1-dev-fp8.safetensors"),
            HFFluxFP8Checkpoint(repo_id="Comfy-Org/flux1-schnell", path="flux1-schnell-fp8.safetensors"),
        ]

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "model",
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "steps",
            "guidance_scale",
            "sampler_cfg",
            "seed",
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

    async def preload_model(self, context: ProcessingContext):
        if self.model.is_empty():
            raise ValueError("Model must be selected.")
        assert self.model.path is not None, "Model path must be set."

        self._model = ModelManager.get_model(self.model.repo_id, "unet", self.model.path)
        self._clip = ModelManager.get_model(self.model.repo_id, "clip", self.model.path)
        self._vae = ModelManager.get_model(self.model.repo_id, "vae", self.model.path)

        if self._model and self._clip and self._vae:
            return

        cache_path = await HF_FAST_CACHE.resolve(self.model.repo_id, self.model.path)
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

        if self._model is None or self._clip is None or self._vae is None:
            raise RuntimeError("Failed to load Flux fp8 checkpoint (UNet/CLIP/VAE).")

        ModelManager.set_model(self.id, self.model.repo_id, "unet", self._model, self.model.path)
        ModelManager.set_model(self.id, self.model.repo_id, "clip", self._clip, self.model.path)
        ModelManager.set_model(self.id, self.model.repo_id, "vae", self._vae, self.model.path)

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._model is None or self._clip is None or self._vae is None:
            raise RuntimeError("Model components must be loaded before processing.")

        with torch.inference_mode():
            with comfy_progress(context, self, self._model):
                unet = ModelPatcher.clone(self._model)
                clip = self._clip.clone()
                vae = self._vae
                unet, clip = self.apply_loras(unet, clip)

                latent = EmptySD3LatentImage().generate(self.width, self.height, 1)[0]
                positive = CLIPTextEncode().encode(clip, self.prompt)[0]
                negative = CLIPTextEncode().encode(clip, self.negative_prompt)[0]
                positive = FluxGuidance().append(positive, self.guidance_scale)[0]

                sampled_latent = KSampler().sample(
                    model=unet,
                    seed=self.seed,
                    steps=self.steps,
                    cfg=self.sampler_cfg,
                    sampler_name=self.sampler.value,
                    scheduler=self.scheduler.value,
                    positive=positive,
                    negative=negative,
                    latent_image=latent,
                    denoise=self.denoise,
                )[0]

                decoded_image = VAEDecodeTiled().decode(vae, sampled_latent, 512)[0]
                return await context.image_from_tensor(decoded_image)


class QwenImage(BaseNode):
    """
    Generates images from text prompts using Qwen-Image.
    image, text-to-image, generative AI, qwen
    """

    model: HFQwenImageCheckpoint = Field(
        default=HFQwenImageCheckpoint(
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            path="split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
            # path="non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors",
        ),
        description="The Qwen-Image UNet/diffusion checkpoint.",
    )
    text_encoder: HFQwenVL = Field(
        default=HFQwenVL(
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
    def get_recommended_models(cls) -> list[HFQwenImageCheckpoint | HFCLIP | HFVAE]:
        return [
            HFQwenImageCheckpoint(
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                path="split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
                # path="non_official/diffusion_models/qwen_image_distill_full_fp8_e4m3fn.safetensors",
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
            "model",
            "text_encoder",
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

    async def _cached_or_local(self, repo_id: str, paths: tuple[str, ...], folder: str):
        for candidate in paths:
            cp = await HF_FAST_CACHE.resolve(repo_id, candidate)
            if cp:
                return cp
        for candidate in paths:
            lp = folder_paths.get_full_path(folder, candidate)
            if lp:
                return lp
        return None

    async def _load_clip(self) -> comfy.sd.CLIP:
        assert self.text_encoder.path is not None, "CLIP path must be set."
        clip_path = await self._cached_or_local(
            self.text_encoder.repo_id,
            (
                self.text_encoder.path,
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

    async def _load_vae(self) -> comfy.sd.VAE:
        assert self.vae_model.path is not None, "VAE path must be set."
        vae_path = await self._cached_or_local(
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
        if self.model.is_empty():
            raise ValueError("UNet model must be selected.")
        if self.text_encoder.is_empty():
            raise ValueError("CLIP model must be selected.")
        if self.vae_model.is_empty():
            raise ValueError("VAE model must be selected.")

        assert self.model.path is not None, "UNet path must be set."
        assert self.text_encoder.path is not None, "CLIP path must be set."
        assert self.vae_model.path is not None, "VAE path must be set."

        self._model = ModelManager.get_model(
            self.model.repo_id, "unet", self.model.path
        )
        self._clip = ModelManager.get_model(
            self.text_encoder.repo_id, "clip", self.text_encoder.path
        )
        self._vae = ModelManager.get_model(
            self.vae_model.repo_id, "vae", self.vae_model.path
        )

        if getattr(self._vae, "first_stage_model", None) is None:
            self._vae = None

        if self._model and self._clip and self._vae:
            return

        cache_path = await HF_FAST_CACHE.resolve(self.model.repo_id, self.model.path)
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

        if self._clip is None:
            self._clip = await self._load_clip()

        # Qwen checkpoints ship the VAE in a separate file, so the auto loader
        # returns a placeholder VAE with no weights. Swap in the real VAE if
        # the loaded instance is missing its first_stage_model.
        if self._vae is None or getattr(self._vae, "first_stage_model", None) is None:
            self._vae = await self._load_vae()

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
            self.text_encoder.repo_id,
            "clip",
            self._clip,
            self.text_encoder.path,
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

        with torch.inference_mode():
            with comfy_progress(context, self, self._model):
                base_unet = self._model
                base_clip = self._clip
                unet = ModelPatcher.clone(base_unet) if self.loras else base_unet
                clip = base_clip.clone() if self.loras else base_clip
                if self.loras:
                    unet, clip = self.apply_loras(unet, clip)

                device = getattr(unet, "load_device", None) or getattr(
                    unet, "device", None
                )
                dtype = getattr(unet, "dtype", None) or torch.float16
                autocast_ctx = (
                    torch.autocast(device_type=device.type, dtype=dtype)
                    if device is not None and hasattr(device, "type")
                    else nullcontext()
                )

                with autocast_ctx:
                    latent = self.get_empty_latent()

                    positive_conditioning, negative_conditioning = (
                        self.get_conditioning(clip)
                    )

                    # Free CLIP weights once conditioning is built to reduce VRAM.
                    del clip
                    soft_empty = getattr(
                        comfy.model_management, "soft_empty_cache", None
                    )
                    if callable(soft_empty):
                        soft_empty()

                    sampled_latent = self.sample(
                        unet,
                        latent,
                        positive_conditioning,
                        negative_conditioning,
                        self.steps,
                    )

                    decoded_image = VAEDecodeTiled().decode(
                        self._vae, sampled_latent, 512
                    )[0]
                    return await context.image_from_tensor(decoded_image)

__all__ = [
    "StableDiffusion",
    "StableDiffusion3",
    "StableDiffusionXL",
    "Flux",
    "FluxFP8",
    "QwenImage",
    "QwenImageEdit",
]


if __name__ == "__main__":

    async def main():
        node = Flux(
            model=HFFluxCheckpoint(
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
