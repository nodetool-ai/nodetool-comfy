from __future__ import annotations

from typing import Optional

from nodetool.metadata.types import LORAFile, LoRAConfig
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


class LoRASelector(BaseNode):
    """
    Selects up to 5 LoRA models to apply to a Stable Diffusion model.
    lora, model customization, fine-tuning

    Use cases:
    - Combining multiple LoRA models for unique image styles
    - Fine-tuning Stable Diffusion models with specific attributes
    - Experimenting with different LoRA combinations
    """

    lora1: Optional[LORAFile] = Field(
        default=LORAFile(), description="First LoRA model"
    )
    strength1: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0, description="Strength for first LoRA"
    )

    lora2: Optional[LORAFile] = Field(
        default=LORAFile(), description="Second LoRA model"
    )
    strength2: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0, description="Strength for second LoRA"
    )

    lora3: Optional[LORAFile] = Field(
        default=LORAFile(), description="Third LoRA model"
    )
    strength3: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0, description="Strength for third LoRA"
    )

    lora4: Optional[LORAFile] = Field(
        default=LORAFile(), description="Fourth LoRA model"
    )
    strength4: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0, description="Strength for fourth LoRA"
    )

    lora5: Optional[LORAFile] = Field(
        default=LORAFile(), description="Fifth LoRA model"
    )
    strength5: Optional[float] = Field(
        default=1.0, ge=0.0, le=2.0, description="Strength for fifth LoRA"
    )

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [
            "lora1",
            "strength1",
            "lora2",
            "strength2",
            "lora3",
            "strength3",
            "lora4",
            "strength4",
            "lora5",
            "strength5",
        ]

    async def process(self, context: ProcessingContext) -> list[LoRAConfig]:
        loras: list[LoRAConfig] = []
        for i in range(1, 6):
            lora = getattr(self, f"lora{i}")
            strength = getattr(self, f"strength{i}")
            if lora.is_set():
                loras.append(LoRAConfig(lora=lora, strength=strength))
        return loras


__all__ = ["LoRASelector"]

