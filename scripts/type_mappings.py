"""
Type mappings between ComfyUI types and Nodetool types.

This module defines how ComfyUI node types are converted to Nodetool types
and vice versa for the proxy node generator.
"""

from typing import Dict, Optional, Any


# ComfyUI type to Nodetool type mapping
COMFY_TO_NODETOOL_TYPE_MAP: Dict[str, str] = {
    # Image and visual types
    "IMAGE": "ImageRef",
    "MASK": "ImageRef",  # Masks are represented as images
    
    # Latent types
    "LATENT": "dict",  # Latent is a dict with 'samples' key containing tensor
    
    # Model types - these are internal ComfyUI objects
    "MODEL": "Any",  # ModelPatcher object
    "CLIP": "Any",  # CLIP model object
    "VAE": "Any",  # VAE model object
    "CONDITIONING": "Any",  # Conditioning tuple/list
    "CONTROL_NET": "Any",  # ControlNet model
    "GLIGEN": "Any",  # GLIGEN model
    "STYLE_MODEL": "Any",  # Style model
    "UPSCALE_MODEL": "Any",  # Upscale model
    "SAMPLER": "Any",  # Sampler object
    "SIGMAS": "Any",  # Sigmas tensor
    "NOISE": "Any",  # Noise object
    "GUIDER": "Any",  # Guider object
    
    # Audio types
    "AUDIO": "AudioRef",
    "VHS_AUDIO": "AudioRef",
    
    # Primitive types
    "INT": "int",
    "FLOAT": "float",
    "STRING": "str",
    "BOOLEAN": "bool",
    
    # Special types
    "COMBO": "str",  # Combo boxes are string enums
    "*": "Any",  # Wildcard type
}

# Import statements needed for various types
TYPE_IMPORTS: Dict[str, str] = {
    "ImageRef": "from nodetool.metadata.types import ImageRef",
    "AudioRef": "from nodetool.metadata.types import AudioRef",
    "Any": "from typing import Any",
    "Optional": "from typing import Optional",
    "List": "from typing import List",
    "Dict": "from typing import Dict",
    "Tuple": "from typing import Tuple",
}


def get_nodetool_type(comfy_type: str) -> str:
    """
    Convert a ComfyUI type string to a Nodetool type string.
    
    Args:
        comfy_type: ComfyUI type string (e.g., "IMAGE", "MODEL", "INT")
        
    Returns:
        Nodetool type string (e.g., "ImageRef", "Any", "int")
    """
    # Handle IO. prefixed types from v1 nodes
    if comfy_type.startswith("IO."):
        comfy_type = comfy_type[3:]  # Remove "IO." prefix
    
    # Handle comfy.* prefixed types (references to enums/lists)
    if comfy_type.startswith("comfy."):
        return "str"  # These are typically combo box selections
    
    # Look up in mapping
    return COMFY_TO_NODETOOL_TYPE_MAP.get(comfy_type, "Any")


def get_pydantic_field_type(
    comfy_type: str,
    config: Optional[Dict[str, Any]] = None,
    is_optional: bool = False
) -> str:
    """
    Get the Pydantic field type declaration for a ComfyUI input.
    
    Args:
        comfy_type: ComfyUI type string
        config: Optional configuration dict with constraints
        is_optional: Whether the field is optional
        
    Returns:
        Pydantic field type string (e.g., "int", "Optional[str]", "ImageRef")
    """
    base_type = get_nodetool_type(comfy_type)
    
    if is_optional:
        return f"Optional[{base_type}]"
    
    return base_type


def get_field_constraints(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract Pydantic field constraints from ComfyUI config.
    
    Args:
        config: ComfyUI input config dictionary
        
    Returns:
        Dictionary of Pydantic Field constraints
    """
    if not config:
        return {}
    
    constraints = {}
    
    # Numeric constraints
    if "min" in config:
        min_val = config["min"]
        # Skip if it's a reference
        if not isinstance(min_val, dict):
            constraints["ge"] = min_val
    if "max" in config:
        max_val = config["max"]
        # Skip if it's a reference
        if not isinstance(max_val, dict):
            constraints["le"] = max_val
    if "step" in config:
        # step is not a direct Pydantic constraint, but we can note it
        pass
    
    # Default value
    if "default" in config:
        constraints["default"] = config["default"]
    
    # Description/tooltip
    if "tooltip" in config:
        constraints["description"] = config["tooltip"]
    elif "description" in config:
        constraints["description"] = config["description"]
    
    # String constraints
    if "multiline" in config and config["multiline"]:
        # Note: Pydantic doesn't have multiline constraint, but we can document it
        pass
    
    return constraints


def get_required_imports(types_used: set) -> list:
    """
    Get the required import statements for a set of types.
    
    Args:
        types_used: Set of type names used in the generated code
        
    Returns:
        List of import statement strings
    """
    imports = set()
    
    for type_name in types_used:
        # Extract base type from Optional[], List[], etc.
        base_type = type_name
        for prefix in ["Optional[", "List[", "Dict[", "Tuple["]:
            if type_name.startswith(prefix):
                base_type = type_name[len(prefix):-1]
                imports.add(TYPE_IMPORTS.get(prefix[:-1], ""))
                break
        
        if base_type in TYPE_IMPORTS:
            imports.add(TYPE_IMPORTS[base_type])
    
    # Remove empty strings
    imports.discard("")
    
    return sorted(list(imports))


def needs_type_conversion(comfy_type: str) -> bool:
    """
    Check if a ComfyUI type needs conversion when calling the underlying node.
    
    Args:
        comfy_type: ComfyUI type string
        
    Returns:
        True if type conversion is needed
    """
    # Internal ComfyUI types don't need conversion (they're passed through)
    internal_types = {
        "MODEL", "CLIP", "VAE", "CONDITIONING", "CONTROL_NET",
        "SAMPLER", "SIGMAS", "NOISE", "GUIDER", "LATENT"
    }
    
    # Image types need conversion from ImageRef to tensor
    image_types = {"IMAGE", "MASK"}
    
    # Audio types need conversion
    audio_types = {"AUDIO", "VHS_AUDIO"}
    
    return comfy_type in (image_types | audio_types)


def get_type_conversion_code(
    param_name: str,
    comfy_type: str,
    direction: str = "to_comfy"
) -> Optional[str]:
    """
    Generate code for type conversion.
    
    Args:
        param_name: Parameter name
        comfy_type: ComfyUI type string
        direction: "to_comfy" or "from_comfy"
        
    Returns:
        Python code string for conversion, or None if no conversion needed
    """
    if not needs_type_conversion(comfy_type):
        return None
    
    if direction == "to_comfy":
        if comfy_type == "IMAGE":
            return f"await context.tensor_from_image({param_name})"
        elif comfy_type == "MASK":
            return f"await context.tensor_from_image({param_name})"
        elif comfy_type in ["AUDIO", "VHS_AUDIO"]:
            return f"await context.tensor_from_audio({param_name})"
    
    elif direction == "from_comfy":
        if comfy_type == "IMAGE":
            return f"await context.image_from_tensor({param_name})"
        elif comfy_type == "MASK":
            return f"await context.image_from_tensor({param_name})"
        elif comfy_type in ["AUDIO", "VHS_AUDIO"]:
            return f"await context.audio_from_tensor({param_name})"
    
    return None
