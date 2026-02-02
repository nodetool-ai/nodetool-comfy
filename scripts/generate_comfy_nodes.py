#!/usr/bin/env python3
"""
ComfyUI Node Code Generator for Nodetool

This script generates nodetool-compatible Python code from the parsed ComfyUI
nodes metadata (comfy_nodes_metadata.json).

It creates:
- Generated node classes that inherit from BaseNode
- Type mappings from ComfyUI types to nodetool types
- Generic process methods that call ComfyUI nodes
"""

import json
import keyword
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# Type mappings from ComfyUI types to Python/Nodetool types
TYPE_MAPPINGS = {
    # Basic types
    "INT": "int",
    "FLOAT": "float",
    "STRING": "str",
    "BOOLEAN": "bool",
    
    # ComfyUI wrapper types - use our wrapper classes
    "MODEL": "Model",  # Model patcher
    "CLIP": "Clip",  # CLIP model
    "VAE": "Vae",  # VAE model
    "CONDITIONING": "Conditioning",  # Conditioning tensor
    "LATENT": "Latent",  # Latent dict with samples
    "IMAGE": "ImageRef",  # Image - use ImageRef from nodetool
    "MASK": "Mask",  # Mask tensor
    "CONTROL_NET": "ControlNet",  # ControlNet
    "STYLE_MODEL": "StyleModel",  # Style model
    "GLIGEN": "Gligen",  # GLIGEN model
    "UPSCALE_MODEL": "UpscaleModel",  # Upscale model
    "SAMPLER": "Sampler",  # Sampler
    "SIGMAS": "Sigmas",  # Sigmas
    "NOISE": "Noise",  # Noise
    "GUIDER": "Guider",  # Guider
    "AUDIO": "Audio",  # Audio tensor
}

# Types that need wrapping/unwrapping
WRAPPER_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "MASK",
    "CONTROL_NET", "STYLE_MODEL", "GLIGEN", "UPSCALE_MODEL",
    "SAMPLER", "SIGMAS", "NOISE", "GUIDER", "AUDIO"
}


def get_python_type(comfy_type: str, config: Optional[Dict] = None) -> str:
    """
    Convert ComfyUI type to Python type annotation.
    
    Args:
        comfy_type: ComfyUI type string (e.g., "INT", "FLOAT", "MODEL", "IO.STRING")
        config: Optional config dict with default, min, max, etc.
        
    Returns:
        Python type annotation string
    """
    # Handle combo/enum types (these should be List[str] for now)
    if comfy_type == "COMBO":
        return "str"
    
    # Handle IO.* types (strip IO. prefix)
    if comfy_type.startswith("IO."):
        comfy_type = comfy_type[3:]  # Remove "IO." prefix
    
    # Handle references like comfy.samplers.KSampler.SAMPLERS
    if comfy_type.startswith("comfy."):
        return "str"
    
    # Handle references like $variable_name
    if comfy_type.startswith("$"):
        return "Any"
    
    # Look up in type mappings
    return TYPE_MAPPINGS.get(comfy_type, "Any")


def sanitize_field_name(name: str) -> str:
    """
    Sanitize field name to be a valid Python identifier.
    
    Args:
        name: Original field name
        
    Returns:
        Sanitized field name
    """
    # Replace spaces and special characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = f"field_{name}"
    
    # Avoid Python keywords using the keyword module
    if keyword.iskeyword(name):
        name = f"{name}_"
    
    return name.lower()


def get_default_value(comfy_type: str, config: Optional[Dict]) -> Any:
    """
    Get default value for a field based on type and config.
    
    Args:
        comfy_type: ComfyUI type string
        config: Optional config dict
        
    Returns:
        Default value
    """
    if config and "default" in config:
        default = config["default"]
        if comfy_type == "STRING":
            # Use repr() for proper string escaping
            return repr(default)
        return default
    
    # Default values for basic types
    if comfy_type == "INT":
        return 0
    elif comfy_type == "FLOAT":
        return 0.0
    elif comfy_type == "STRING":
        return repr("")
    elif comfy_type == "BOOLEAN":
        return "False"
    
    # For complex types, use None
    return "None"


def generate_field_definition(field_name: str, field_info: Dict, python_type: str) -> str:
    """
    Generate Pydantic Field definition for a node input.
    
    Args:
        field_name: Name of the field
        field_info: Field info from metadata
        python_type: Python type annotation
        
    Returns:
        Field definition string
    """
    field_type = field_info.get("type", "")
    config = field_info.get("config", {})
    
    # Get default value
    default = get_default_value(field_type, config)
    
    # Get description (tooltip)
    description = config.get("tooltip", "")
    if not description:
        description = f"{field_name} parameter"
    
    # Build Field kwargs
    field_kwargs = [f'default={default}']
    field_kwargs.append(f'description="{description}"')
    
    # Add constraints for numeric types
    if field_type in ["INT", "FLOAT"]:
        if "min" in config:
            field_kwargs.append(f'ge={config["min"]}')
        if "max" in config:
            field_kwargs.append(f'le={config["max"]}')
    
    field_def = f"    {field_name}: {python_type} = Field({', '.join(field_kwargs)})"
    return field_def


def generate_process_method(node_info: Dict) -> List[str]:
    """
    Generate the process method implementation for a node.
    
    Args:
        node_info: Node metadata dict
        
    Returns:
        List of code lines for the process method
    """
    lines = []
    node_id = node_info.get("node_id") or "Unknown"
    class_name = node_info.get("class_name") or node_id
    function_name = node_info.get("function") or "process"
    node_style = node_info.get("node_style", "v1")
    return_types = node_info.get("return_types", [])
    
    # If function_name is None or empty, skip implementation
    if not function_name or function_name == "None":
        lines.append("    async def process(self, context: ProcessingContext) -> Any:")
        lines.append(f'        """Process the {node_id} node."""')
        lines.append("        # TODO: Function name not specified in metadata")
        lines.append("        raise NotImplementedError(")
        lines.append(f'            "Function name not available for {class_name}"')
        lines.append("        )")
        return lines
    
    # Determine return type annotation with proper wrapper types
    if return_types:
        return_type_strs = []
        for rt in return_types:
            # Normalize type (strip IO. prefix if present)
            normalized_rt = rt[3:] if rt.startswith("IO.") else rt
            if normalized_rt in TYPE_MAPPINGS:
                return_type_strs.append(TYPE_MAPPINGS[normalized_rt])
            else:
                return_type_strs.append("Any")
        
        if len(return_types) == 1:
            return_annotation = f" -> {return_type_strs[0]}"
        else:
            return_annotation = f" -> tuple[{', '.join(return_type_strs)}]"
    else:
        return_annotation = " -> Any"
    
    lines.append(f"    async def process(self, context: ProcessingContext){return_annotation}:")
    lines.append(f'        """Process the {node_id} node."""')
    lines.append("        # Import the ComfyUI node class")
    
    # Determine import path based on source
    source = node_info.get("source", "nodes.py")
    if source == "nodes.py":
        lines.append(f"        from nodes import {class_name}")
    elif source.startswith("comfy_extras/"):
        module_name = source.replace("comfy_extras/", "").replace(".py", "")
        lines.append(f"        from comfy_extras.{module_name} import {class_name}")
    else:
        lines.append(f"        # TODO: Import from {source}")
        lines.append(f"        raise NotImplementedError('Import path for {source} not implemented')")
        return lines
    
    lines.append("")
    lines.append("        # Create node instance")
    lines.append(f"        node = {class_name}()")
    lines.append("")
    lines.append("        # Prepare inputs (unwrap wrapper types)")
    lines.append("        kwargs = {}")
    
    # Add input parameters with unwrapping
    if node_style == "v1":
        input_types = node_info.get("input_types", {})
        required = input_types.get("required", {})
        optional = input_types.get("optional", {})
        
        for field_name, field_info in required.items():
            if not isinstance(field_info, dict):
                continue
            sanitized_name = sanitize_field_name(field_name)
            field_type = field_info.get("type", "")
            
            # Normalize type (strip IO. prefix if present)
            normalized_type = field_type[3:] if field_type.startswith("IO.") else field_type
            
            # Unwrap ComfyUI wrapper types, convert ImageRef to tensor
            if normalized_type in WRAPPER_TYPES:
                lines.append(f'        kwargs["{field_name}"] = self.{sanitized_name}.value if self.{sanitized_name} else None')
            elif normalized_type == "IMAGE":
                lines.append(f'        kwargs["{field_name}"] = await context.image_to_tensor(self.{sanitized_name}) if self.{sanitized_name} else None')
            else:
                lines.append(f'        kwargs["{field_name}"] = self.{sanitized_name}')
        
        for field_name, field_info in optional.items():
            if not isinstance(field_info, dict):
                continue
            sanitized_name = sanitize_field_name(field_name)
            field_type = field_info.get("type", "")
            
            # Normalize type (strip IO. prefix if present)
            normalized_type = field_type[3:] if field_type.startswith("IO.") else field_type
            
            # Unwrap ComfyUI wrapper types, convert ImageRef to tensor
            if normalized_type in WRAPPER_TYPES:
                lines.append(f'        if self.{sanitized_name} is not None:')
                lines.append(f'            kwargs["{field_name}"] = self.{sanitized_name}.value if self.{sanitized_name} else None')
            elif normalized_type == "IMAGE":
                lines.append(f'        if self.{sanitized_name} is not None:')
                lines.append(f'            kwargs["{field_name}"] = await context.image_to_tensor(self.{sanitized_name})')
            else:
                lines.append(f'        if self.{sanitized_name} is not None:')
                lines.append(f'            kwargs["{field_name}"] = self.{sanitized_name}')
    
    lines.append("")
    lines.append("        # Call the node function")
    lines.append(f"        result = node.{function_name}(**kwargs)")
    lines.append("")
    lines.append("        # Wrap results in appropriate types")
    
    if return_types:
        if len(return_types) == 1:
            rt = return_types[0]
            # Normalize type (strip IO. prefix if present)
            normalized_rt = rt[3:] if rt.startswith("IO.") else rt
            
            if normalized_rt in WRAPPER_TYPES:
                wrapper_class = TYPE_MAPPINGS[normalized_rt]
                lines.append("        raw_result = result[0] if isinstance(result, tuple) else result")
                lines.append(f"        return {wrapper_class}(raw_result)")
            elif normalized_rt == "IMAGE":
                lines.append("        raw_result = result[0] if isinstance(result, tuple) else result")
                lines.append("        return await context.image_from_tensor(raw_result)")
            else:
                lines.append("        return result[0] if isinstance(result, tuple) else result")
        else:
            lines.append("        raw_results = result if isinstance(result, tuple) else (result,)")
            lines.append("        wrapped = []")
            lines.append("        for i, raw_val in enumerate(raw_results):")
            
            # Build wrapping logic for each return type
            for idx, rt in enumerate(return_types):
                # Normalize type (strip IO. prefix if present)
                normalized_rt = rt[3:] if rt.startswith("IO.") else rt
                
                if normalized_rt in WRAPPER_TYPES:
                    wrapper_class = TYPE_MAPPINGS[normalized_rt]
                    lines.append(f"            if i == {idx}:")
                    lines.append(f"                wrapped.append({wrapper_class}(raw_val))")
                elif normalized_rt == "IMAGE":
                    lines.append(f"            if i == {idx}:")
                    lines.append(f"                wrapped.append(await context.image_from_tensor(raw_val))")
                else:
                    lines.append(f"            if i == {idx}:")
                    lines.append(f"                wrapped.append(raw_val)")
            
            lines.append("        return tuple(wrapped)")
    else:
        lines.append("        return result")
    
    return lines


def generate_node_class(node_info: Dict) -> str:
    """
    Generate a complete node class from node metadata.
    
    Args:
        node_info: Node metadata dict
        
    Returns:
        Python class definition string
    """
    node_id = node_info.get("node_id") or "Unknown"
    class_name = node_info.get("class_name") or node_id.replace(" ", "")
    category = node_info.get("category") or "uncategorized"
    description = node_info.get("description") or ""
    node_style = node_info.get("node_style", "v1")
    
    # Sanitize class name
    class_name = re.sub(r'[^a-zA-Z0-9_]', '', class_name)
    if class_name and class_name[0].isdigit():
        class_name = f"Node{class_name}"
    
    # Start building the class
    lines = []
    lines.append(f"class {class_name}(BaseNode):")
    
    # Docstring
    if description:
        lines.append(f'    """')
        lines.append(f'    {description}')
        lines.append(f'    ')
        lines.append(f'    Category: {category}')
        lines.append(f'    ComfyUI Node ID: {node_id}')
        lines.append(f'    """')
    else:
        lines.append(f'    """{node_id} node from ComfyUI (category: {category})"""')
    
    lines.append("")
    
    # Generate fields based on node style
    if node_style == "v1":
        input_types = node_info.get("input_types", {})
        required = input_types.get("required", {})
        optional = input_types.get("optional", {})
        
        # Generate required fields
        for field_name, field_info in required.items():
            # Skip if field_info is not a dict (could be list or other type)
            if not isinstance(field_info, dict):
                continue
            
            sanitized_name = sanitize_field_name(field_name)
            python_type = get_python_type(field_info.get("type", ""), field_info.get("config"))
            field_def = generate_field_definition(sanitized_name, field_info, python_type)
            lines.append(field_def)
        
        # Generate optional fields
        for field_name, field_info in optional.items():
            # Skip if field_info is not a dict (could be list or other type)
            if not isinstance(field_info, dict):
                continue
            
            sanitized_name = sanitize_field_name(field_name)
            python_type = get_python_type(field_info.get("type", ""), field_info.get("config"))
            # Make optional fields Optional
            python_type = f"Optional[{python_type}]"
            field_def = generate_field_definition(sanitized_name, field_info, python_type)
            lines.append(field_def)
    
    lines.append("")
    
    # Generate process method
    process_lines = generate_process_method(node_info)
    lines.extend(process_lines)
    
    return "\n".join(lines)


def generate_nodes_file(nodes: List[Dict], output_path: Path, category_filter: Optional[str] = None):
    """
    Generate a Python file with multiple node classes.
    
    Args:
        nodes: List of node metadata dicts
        output_path: Path to output Python file
        category_filter: Optional category to filter nodes
    """
    # Filter nodes if needed
    if category_filter:
        nodes = [n for n in nodes if n.get("category") == category_filter]
    
    # Start building the file
    lines = []
    
    # File header
    lines.append('"""')
    lines.append("Auto-generated ComfyUI nodes for Nodetool")
    lines.append("")
    lines.append("This file was generated from comfy_nodes_metadata.json")
    lines.append("DO NOT EDIT MANUALLY - regenerate using scripts/generate_comfy_nodes.py")
    lines.append('"""')
    lines.append("")
    
    # Imports
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from typing import Any, Optional")
    lines.append("")
    lines.append("from nodetool.workflows.base_node import BaseNode")
    lines.append("from nodetool.workflows.processing_context import ProcessingContext")
    lines.append("from nodetool.metadata.types import ImageRef")
    lines.append("from nodetool.nodes.comfy.types import (")
    lines.append("    Model, Clip, Vae, Conditioning, Latent, Mask,")
    lines.append("    ControlNet, StyleModel, Gligen, UpscaleModel,")
    lines.append("    Sampler, Sigmas, Noise, Guider, Audio")
    lines.append(")")
    lines.append("from pydantic import Field")
    lines.append("")
    lines.append("")
    
    # Generate each node class
    for node in nodes:
        if node.get("deprecated", False):
            continue  # Skip deprecated nodes
        
        class_def = generate_node_class(node)
        lines.append(class_def)
        lines.append("")
        lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Generated {len(nodes)} nodes to {output_path}")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    # Load metadata
    metadata_path = repo_root / "comfy_nodes_metadata.json"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found")
        print("Run: python scripts/parse_comfy_nodes.py")
        return 1
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    nodes = metadata.get("comfy_nodes", [])
    print(f"Loaded {len(nodes)} nodes from metadata")
    
    # Get categories
    categories = set()
    for node in nodes:
        cat = node.get("category") or "uncategorized"
        categories.add(cat)
    
    print(f"\nFound categories: {sorted(categories)}")
    
    # Generate nodes by category
    output_dir = repo_root / "src" / "nodetool" / "nodes" / "comfy" / "generated"
    
    # Generate all nodes in one file for now
    all_nodes_path = output_dir / "all_nodes.py"
    generate_nodes_file(nodes, all_nodes_path)
    
    # Optionally generate by category (uncomment to enable)
    # for category in categories:
    #     safe_category = re.sub(r'[^a-zA-Z0-9_]', '_', category)
    #     output_path = output_dir / f"{safe_category}_nodes.py"
    #     generate_nodes_file(nodes, output_path, category_filter=category)
    
    print("\nNode generation complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
