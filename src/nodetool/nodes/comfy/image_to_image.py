from __future__ import annotations

from ast import Tuple
from contextlib import nullcontext
from typing import List

import PIL.Image
import comfy
import comfy.sd
import comfy.model_management
import folder_paths
import numpy as np
import torch
from comfy.model_patcher import ModelPatcher
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
    LoraLoader,
    VAEDecode,
    VAEDecodeTiled,
    VAEEncode,
    VAEEncodeForInpaint,
    ConditioningZeroOut,
)
from nodetool.ml.core.model_manager import ModelManager
from nodetool.metadata.types import (
    HFCLIP,
    HFT5,
    HFVAE,
    HFControlNet,
    HFFluxKontextCheckpoint,
    HFImageToImage,
    HFQwenImageEditCheckpoint,
    HFQwenVL,
    HFTextToImage,
    HFUnet,
    ImageRef,
    LoRAConfig,
)
from nodetool.nodes.comfy.constants import (
    FLUX_DEV_MODELS,
    FLUX_SCHNELL_MODELS,
    HF_CONTROLNET_MODELS,
    HF_CONTROLNET_XL_MODELS,
    HF_STABLE_DIFFUSION_MODELS,
    HF_STABLE_DIFFUSION_XL_MODELS,
    FLUX_CLIP_L,
    FLUX_CLIP_T5XXL,
    FLUX_VAE,
)
from nodetool.nodes.comfy.enums import Sampler, Scheduler
from nodetool.nodes.comfy.utils import comfy_progress
from nodetool.nodes.comfy.text_to_image import (
    StableDiffusion as TextToImageSD,
)
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)


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

                latent = await self.get_latent(
                    vae, context, initial_width, initial_height
                )

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

    model: HFFluxKontextCheckpoint = Field(
        default=HFFluxKontextCheckpoint(
            repo_id="Comfy-Org/flux1-kontext-dev_ComfyUI",
            path="split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
        ),
        description="The Flux UNet checkpoint to use (e.g. flux1-dev or flux1-dev-kontext).",
    )
    text_encoder: HFCLIP = Field(
        default=FLUX_CLIP_L,
        description="The Flux CLIP checkpoint to use (e.g. flux1-dev or flux1-dev-kontext).",
    )
    t5_model: HFT5 = Field(
        default=FLUX_CLIP_T5XXL,
        description="The Flux T5 checkpoint to use (e.g. flux1-dev or flux1-dev-kontext).",
    )
    vae: HFVAE = Field(
        default=FLUX_VAE,
        description="The Flux VAE checkpoint to use (e.g. flux1-dev or flux1-dev-kontext).",
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
    guidance_scale: float = Field(
        default=2.5, ge=0.0, le=30.0, description="Flux guidance strength."
    )
    sampler_cfg: float = Field(
        default=1.0,
        ge=0.0,
        le=30.0,
        description="Classifier-free guidance scale passed to the sampler.",
    )
    seed: int = Field(default=0, ge=0, le=1000000000)
    denoise: float = Field(default=1.0, ge=0.0, le=1.0)
    scheduler: Scheduler = Field(default=Scheduler.simple)
    sampler: Sampler = Field(default=Sampler.euler)
    _model: ModelPatcher | None = None
    _clip: comfy.sd.CLIP | None = None
    _vae: comfy.sd.VAE | None = None

    @classmethod
    def get_title(cls) -> str:
        return "Flux Kontext (Img2Img)"

    @classmethod
    def get_recommended_models(cls) -> list[HFFluxKontextCheckpoint | HFVAE | HFCLIP | HFT5]:
        # Mirror the Flux text-to-image node: recommend Flux UNets plus CLIP and VAE.
        return [
            HFFluxKontextCheckpoint(
                repo_id="Comfy-Org/flux1-kontext-dev_ComfyUI",
                path="split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
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
            "sampler_cfg",
            "seed",
            "denoise",
            "scheduler",
            "sampler",
        ]

    def required_inputs(self) -> list[str]:
        return ["input_image"]

    async def preload_model(self, context: ProcessingContext):
        import comfy.utils
        import comfy.sd

        if self.model.is_empty():
            raise ValueError("Model must be selected.")

        assert self.model.path is not None, "Model path must be set."

        self._model = ModelManager.get_model(
            self.model.repo_id, "unet", self.model.path
        )
        self._clip = ModelManager.get_model(
            FLUX_CLIP_L.repo_id, "clip", FLUX_CLIP_L.path
        )
        self._vae = ModelManager.get_model(FLUX_VAE.repo_id, "vae", FLUX_VAE.path)

        if getattr(self._vae, "first_stage_model", None) is None:
            self._vae = None

        if self._model and self._clip and self._vae:
            return

        print(
            f"Trying to load model from cache: {self.model.repo_id}/{self.model.path}"
        )
        cache_path = try_to_load_from_cache(self.model.repo_id, self.model.path)

        logger.info(f"Cache path: {cache_path}")

        if cache_path is not None:
            self._model, self._clip, self._vae, _ = (
                comfy.sd.load_checkpoint_guess_config(
                    cache_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                )
            )

        if self._clip is None or self._vae is None:
            import comfy.utils
            import comfy.sd

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

        assert (
            self._model is not None
        ), f"Model {self.model.repo_id}/{self.model.path} must be loaded."
        assert (
            self._clip is not None
        ), f"CLIP {FLUX_CLIP_L.repo_id}/{FLUX_CLIP_L.path} and {FLUX_CLIP_T5XXL.repo_id}/{FLUX_CLIP_T5XXL.path} must be loaded."
        assert (
            self._vae is not None
        ), f"VAE {FLUX_VAE.repo_id}/{FLUX_VAE.path} must be loaded."

        ModelManager.set_model(
            self.id,
            self.model.repo_id,
            "unet",
            self._model,
            self.model.path,
        )
        ModelManager.set_model(
            self.id,
            FLUX_CLIP_L.repo_id,
            "clip",
            self._clip,
            FLUX_CLIP_L.path,
        )
        ModelManager.set_model(
            self.id,
            FLUX_VAE.repo_id,
            "vae",
            self._vae,
            FLUX_VAE.path,
        )

    async def process(self, context: ProcessingContext) -> ImageRef:
        assert self._model is not None, "Model must be loaded."
        assert self._clip is not None, "CLIP must be loaded."
        assert self._vae is not None, "VAE must be loaded."

        if self.input_image.is_empty():
            raise ValueError("input_image must be provided for Flux Kontext img2img.")

        with torch.inference_mode():
            with comfy_progress(context, self, self._model):
                # Convert input image to tensor (BHWC in [0,1]) and scale it with FluxKontextImageScale.
                input_pil = await context.image_to_pil(self.input_image)
                image = np.array(input_pil.convert("RGB")).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image)[None,]

                scaled_image = FluxKontextImageScale().scale(image_tensor)[0]

                # Encode the scaled image into a latent using the Flux VAE.
                latent = VAEEncode().encode(self._vae, scaled_image)[0]

                # Positive / negative conditioning from CLIP.
                positive = CLIPTextEncode().encode(self._clip, self.prompt)[0]

                if self.negative_prompt:
                    negative = CLIPTextEncode().encode(
                        self._clip, self.negative_prompt
                    )[0]
                else:
                    negative = ConditioningZeroOut().zero_out(positive)[0]

                # Attach the image latent as a reference latent, then apply Flux guidance.
                positive = ReferenceLatent().execute(positive, latent)[0]
                positive = FluxGuidance().append(positive, self.guidance_scale)[0]

                # KSampler performs the Flux sampling using the image-derived latent.
                sampled_latent = KSampler().sample(
                    model=self._model,
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

                decoded_image = VAEDecodeTiled().decode(
                    self._vae, sampled_latent, 512
                )[0]
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
    _controlnet_model: object | None = None

    @classmethod
    def get_recommended_models(cls) -> list[HFControlNet]:
        return HF_CONTROLNET_MODELS + HF_CONTROLNET_XL_MODELS

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return super().get_basic_fields() + [
            "controlnet",
            "image",
            "strength",
        ]

    def required_inputs(self) -> list[str]:
        return ["input_image", "image"]

    async def preload_model(self, context: ProcessingContext):
        # Load the base model (UNet, CLIP, VAE) from parent class
        await super().preload_model(context)

        # Load and cache the ControlNet model
        if self.controlnet.is_empty():
            return

        assert self.controlnet.path is not None, "ControlNet path must be set."

        self._controlnet_model = ModelManager.get_model(
            self.controlnet.repo_id, "controlnet", self.controlnet.path
        )

        if self._controlnet_model:
            return

        controlnet_path = try_to_load_from_cache(
            self.controlnet.repo_id, self.controlnet.path
        )

        if controlnet_path is None:
            raise ValueError(
                "ControlNet checkpoint not found. Download from Recommended Models."
            )

        self._controlnet_model = ControlNetLoader().load_controlnet(controlnet_path)[0]

        ModelManager.set_model(
            self.id,
            self.controlnet.repo_id,
            "controlnet",
            self._controlnet_model,
            self.controlnet.path,
        )

    async def apply_controlnet(
        self,
        context: ProcessingContext,
        width: int,
        height: int,
        conditioning: list,
    ):
        if self.controlnet.is_empty():
            return conditioning

        if self._controlnet_model is None:
            raise RuntimeError("ControlNet model must be loaded before processing.")

        if not self.image.is_empty():
            pil = await context.image_to_pil(self.image)
            pil = pil.resize((width, height), PIL.Image.Resampling.LANCZOS)
            image = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
            tensor = torch.from_numpy(image)[None,]
            conditioning = ControlNetApply().apply_controlnet(
                conditioning,
                self._controlnet_model,
                tensor,
                self.strength,
            )[0]

        return conditioning

    async def process(self, context: ProcessingContext) -> ImageRef:
        if self._model is None or self._clip is None or self._vae is None:
            raise RuntimeError("Model components must be loaded before processing.")

        with torch.inference_mode():
            with comfy_progress(context, self, self._model):
                unet = ModelPatcher.clone(self._model)
                clip = self._clip.clone()
                vae = self._vae

                positive_conditioning, negative_conditioning = self.get_conditioning(
                    clip
                )

                if self.width >= 1024 and self.height >= 1024:
                    num_hires_steps = self.num_inference_steps // 2
                    num_lowres_steps = self.num_inference_steps - num_hires_steps
                    initial_width, initial_height = self.width // 2, self.height // 2
                else:
                    num_hires_steps = 0
                    num_lowres_steps = self.num_inference_steps
                    initial_width, initial_height = self.width, self.height

                latent = await self.get_latent(
                    vae, context, initial_width, initial_height
                )

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



class QwenImageEdit(BaseNode):
    """
    Performs image editing using Qwen-Image-Edit with reference image conditioning.
    image, image-editing, generative AI, qwen
    """

    model: HFQwenImageEditCheckpoint = Field(
        default=HFQwenImageEditCheckpoint(
            repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI",
            path="split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
        ),
        description="The Qwen-Image-Edit UNet checkpoint.",
    )
    text_encoder: HFQwenVL = Field(
        default=HFQwenVL(
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
    def get_recommended_models(cls) -> list[HFQwenImageEditCheckpoint | HFQwenVL | HFVAE]:
        return [
            HFQwenImageEditCheckpoint(
                repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI",
                path="split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors",
            ),
            HFQwenVL(
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
    ) -> tuple[ModelPatcher, comfy.sd.CLIP]:
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
        assert self.text_encoder.path is not None, "CLIP path must be set."
        clip_path = self._cached_or_local(
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

    def _load_vae(self) -> comfy.sd.VAE:
        assert self.vae_model.path is not None, "VAE path must be set."
        import comfy.utils
        import comfy.sd

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

        if self._clip is None:
            self._clip = self._load_clip()

        # Qwen checkpoints do not bundle a VAE; load the external checkpoint
        # if the auto-detected VAE is missing its weights.
        if self._vae is None or getattr(self._vae, "first_stage_model", None) is None:
            self._vae = self._load_vae()

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

    async def _image_to_tensor(self, context: ProcessingContext) -> torch.Tensor:
        if self.input_image.is_empty():
            raise ValueError("input_image must be provided.")
        pil_image = await context.image_to_pil(self.input_image)
        image = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(image)[None, ...]

    def get_conditioning(
        self, clip: comfy.sd.CLIP, vae: comfy.sd.VAE, image_tensor: torch.Tensor
    ) -> tuple[list, list]:
        from comfy_extras.nodes_qwen import TextEncodeQwenImageEdit
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

        with torch.inference_mode():
            with comfy_progress(context, self, self._model):
                base_unet = self._model
                base_clip = self._clip
                vae = self._vae

                # Only clone/patch when LoRAs are present to avoid extra allocations.
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
                    image_tensor = await self._image_to_tensor(context)
                    if device is not None:
                        image_tensor = image_tensor.to(device=device, dtype=dtype)

                    latent = self.encode_latent(vae, image_tensor)
                    positive_conditioning, negative_conditioning = (
                        self.get_conditioning(clip, vae, image_tensor)
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

                    decoded_image = VAEDecodeTiled().decode(vae, sampled_latent, 512)[0]
                    return await context.image_from_tensor(decoded_image)


__all__ = [
    "StableDiffusionImageToImage",
    "StableDiffusion3ImageToImage",
    "StableDiffusionXLImageToImage",
    "FluxKontext",
    "ControlNet",
]
