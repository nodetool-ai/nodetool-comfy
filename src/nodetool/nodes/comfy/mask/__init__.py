from enum import Enum
from pydantic import Field
from nodetool.metadata.types import ImageRef, Mask
from nodetool.nodes.comfy.comfy_node import ComfyNode
import comfy_extras.nodes_mask


class ChannelEnum(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    ALPHA = "alpha"


class OperationEnum(str, Enum):
    MULTIPLY = "multiply"
    ADD = "add"
    SUBTRACT = "subtract"
    AND = "and"
    OR = "or"
    XOR = "xor"


class MaskToImage(ComfyNode):
    """
    The Convert Mask to Image node can be used to convert a mask to a grey scale image.
    """

    _comfy_class = comfy_extras.nodes_mask.MaskToImage

    mask: Mask = Field(default=Mask(), description="The mask to convert.")

    @classmethod
    def get_title(cls):
        return "Convert Mask to Image"

    @classmethod
    def return_type(cls):
        return {"image": ImageRef}


class ImageToMask(ComfyNode):
    """
    The Convert Image yo Mask node can be used to convert a specific channel of an image into a mask.
    """

    _comfy_class = comfy_extras.nodes_mask.ImageToMask

    image: ImageRef = Field(
        default=ImageRef(), description="The image to extract the mask."
    )
    channel: ChannelEnum = Field(
        default=ChannelEnum.RED, description="The channel to use for the mask."
    )

    @classmethod
    def get_title(cls):
        return "Convert Image to Mask"

    @classmethod
    def return_type(cls):
        return {"mask": Mask}


class ImageColorToMask(ComfyNode):
    """
    The Image Color to Mask node can be used to extract a mask from an image based on a specific color.
    """

    _comfy_class = comfy_extras.nodes_mask.ImageColorToMask

    image: ImageRef = Field(
        default=ImageRef(), description="The image to extract the color mask."
    )
    color: int = Field(default=0, description="The color to use for the mask.")

    @classmethod
    def return_type(cls):
        return {"mask": Mask}


class SolidMask(ComfyNode):
    """
    The Solid Mask node can be used to create a solid masking containing a single value.
    """

    _comfy_class = comfy_extras.nodes_mask.SolidMask

    value: float = Field(default=1.0, description="The value for the solid mask.")
    width: int = Field(default=512, description="Width of the solid mask.")
    height: int = Field(default=512, description="Height of the solid mask.")

    @classmethod
    def return_type(cls):
        return {"mask": Mask}


class InvertMask(ComfyNode):
    """
    The Invert Mask node can be used to invert a mask.
    """

    _comfy_class = comfy_extras.nodes_mask.InvertMask

    mask: Mask = Field(default=Mask(), description="The mask to invert.")

    @classmethod
    def return_type(cls):
        return {"mask": Mask}


class CropMask(ComfyNode):
    """
    The Crop Mask node can be used to crop a mask to a new shape.
    """

    _comfy_class = comfy_extras.nodes_mask.CropMask

    mask: Mask = Field(default=Mask(), description="The mask to crop.")
    x: int = Field(default=0, description="The x position for cropping.")
    y: int = Field(default=0, description="The y position for cropping.")
    width: int = Field(default=512, description="Width of the crop.")
    height: int = Field(default=512, description="Height of the crop.")

    @classmethod
    def return_type(cls):
        return {"mask": Mask}


class MaskComposite(ComfyNode):
    """
    The Mask Composite node can be used to paste one mask into another.
    """

    _comfy_class = comfy_extras.nodes_mask.MaskComposite

    destination: Mask = Field(default=Mask(), description="The destination mask.")
    source: Mask = Field(default=Mask(), description="The source mask.")
    x: int = Field(default=0, description="The x position.")
    y: int = Field(default=0, description="The y position.")
    operation: OperationEnum = Field(
        default=OperationEnum.MULTIPLY, description="The operation to use."
    )

    @classmethod
    def return_type(cls):
        return {"mask": Mask}


class FeatherMask(ComfyNode):
    """
    The Feather Mask node can be used to feather a mask.
    """

    _comfy_class = comfy_extras.nodes_mask.FeatherMask

    mask: Mask = Field(default=Mask(), description="The mask to feather.")
    left: int = Field(default=0, description="Feather amount on the left.")
    top: int = Field(default=0, description="Feather amount on the top.")
    right: int = Field(default=0, description="Feather amount on the right.")
    bottom: int = Field(default=0, description="Feather amount on the bottom.")

    @classmethod
    def return_type(cls):
        return {"mask": Mask}


class GrowMask(ComfyNode):
    """
    The Grow Mask node can be used to grow a mask.
    """

    _comfy_class = comfy_extras.nodes_mask.GrowMask

    mask: Mask = Field(default=Mask(), description="The mask to grow.")
    expand: int = Field(default=0, description="The amount to expand the mask.")
    tapered_corners: bool = Field(
        default=True, description="Whether to taper the corners."
    )

    @classmethod
    def return_type(cls):
        return {"mask": Mask}
