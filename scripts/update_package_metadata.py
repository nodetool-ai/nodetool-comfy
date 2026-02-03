#!/usr/bin/env python3
"""
Update Package Metadata for Generated ComfyUI Nodes

PREFERRED METHOD: Use `nodetool pack scan` command from nodetool-core:
    nodetool pack scan --package nodetool-comfy

This script is a fallback for when nodetool-core is not available.
It generates package metadata JSON from the generated ComfyUI nodes
in the format expected by nodetool for DSL code generation.
"""

import json
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional


def extract_node_metadata_from_class(class_node: ast.ClassDef, module_name: str) -> Optional[Dict[str, Any]]:
    """Extract metadata from a generated node class."""
    
    # Get docstring
    docstring = ast.get_docstring(class_node)
    if not docstring:
        return None
    
    # Parse docstring to get description and category
    lines = docstring.strip().split('\n')
    description = lines[0] if lines else ""
    
    category = None
    node_id = None
    for line in lines:
        if line.strip().startswith("Category:"):
            category = line.split("Category:")[1].strip()
        elif line.strip().startswith("ComfyUI Node ID:"):
            node_id = line.split("ComfyUI Node ID:")[1].strip()
    
    if not node_id:
        return None
    
    # Extract properties from class
    properties = []
    for item in class_node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            field_name = item.target.id
            
            # Skip private fields
            if field_name.startswith('_'):
                continue
            
            # Get type annotation
            type_annotation = ast.unparse(item.annotation) if item.annotation else "Any"
            
            # Try to extract Field() call for defaults and constraints
            field_info = {
                "name": field_name,
                "type": {"type": map_type_to_metadata(type_annotation)},
                "title": field_name.replace('_', ' ').title()
            }
            
            if isinstance(item.value, ast.Call):
                # Extract Field parameters
                for keyword in item.value.keywords:
                    if keyword.arg == "default":
                        field_info["default"] = ast.literal_eval(keyword.value) if isinstance(keyword.value, (ast.Constant, ast.List, ast.Dict, ast.Tuple)) else None
                    elif keyword.arg == "description":
                        if isinstance(keyword.value, ast.Constant):
                            field_info["description"] = keyword.value.value
                    elif keyword.arg == "ge":
                        field_info["min"] = ast.literal_eval(keyword.value)
                    elif keyword.arg == "le":
                        field_info["max"] = ast.literal_eval(keyword.value)
            
            properties.append(field_info)
    
    # Extract return type from process method
    return_types = []
    for item in class_node.body:
        if isinstance(item, ast.AsyncFunctionDef) and item.name == "process":
            if item.returns:
                return_type_str = ast.unparse(item.returns)
                return_types = [return_type_str]
            break
    
    return {
        "title": class_node.name,
        "description": description,
        "namespace": f"comfy.{category.replace('/', '.')}" if category else "comfy.generated",
        "node_type": f"comfy.generated.{class_node.name}",
        "properties": properties,
        "outputs": return_types
    }


def map_type_to_metadata(type_str: str) -> str:
    """Map Python type annotations to metadata type strings."""
    type_map = {
        "int": "int",
        "float": "float",
        "str": "str",
        "bool": "bool",
        "Model": "comfy.model",
        "Clip": "comfy.clip",
        "Vae": "comfy.vae",
        "Conditioning": "comfy.conditioning",
        "Latent": "comfy.latent",
        "Mask": "comfy.mask",
        "ImageRef": "image",
        "ControlNet": "comfy.controlnet",
        "StyleModel": "comfy.style_model",
        "Gligen": "comfy.gligen",
        "UpscaleModel": "comfy.upscale_model",
        "Sampler": "comfy.sampler",
        "Sigmas": "comfy.sigmas",
        "Noise": "comfy.noise",
        "Guider": "comfy.guider",
        "Audio": "comfy.audio",
    }
    
    # Strip Optional wrapper
    if type_str.startswith("Optional[") and type_str.endswith("]"):
        type_str = type_str[9:-1]
    
    return type_map.get(type_str, "any")


def parse_generated_nodes(file_path: Path) -> List[Dict[str, Any]]:
    """Parse the generated nodes file and extract metadata."""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error parsing {file_path}: {e}")
        return []
    
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it inherits from BaseNode
            is_base_node = any(
                isinstance(base, ast.Name) and base.id == "BaseNode"
                for base in node.bases
            )
            
            if is_base_node:
                metadata = extract_node_metadata_from_class(node, "comfy.generated")
                if metadata:
                    nodes.append(metadata)
    
    return nodes


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    generated_nodes_file = repo_root / "src" / "nodetool" / "nodes" / "comfy" / "generated" / "all_nodes.py"
    output_file = repo_root / "src" / "nodetool" / "package_metadata" / "nodetool-comfy.json"
    
    if not generated_nodes_file.exists():
        print(f"Error: Generated nodes file not found: {generated_nodes_file}")
        return 1
    
    print(f"Parsing generated nodes from {generated_nodes_file}...")
    nodes = parse_generated_nodes(generated_nodes_file)
    print(f"Found {len(nodes)} nodes")
    
    # Load existing metadata if it exists
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = {
            "name": "nodetool-comfy",
            "description": "ComfyUI nodes for Nodetool",
            "version": "0.6.2-rc.19",
            "authors": ["Matthias Georgi <matti.georgi@gmail.com>"],
            "repo_id": "",
            "nodes": []
        }
    
    # Keep existing manually created nodes, add generated ones
    manual_nodes = [n for n in existing_metadata.get("nodes", []) if not n.get("node_type", "").startswith("comfy.generated.")]
    
    # Combine manual and generated nodes
    all_nodes = manual_nodes + nodes
    
    existing_metadata["nodes"] = all_nodes
    
    # Write updated metadata
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, indent=2)
    
    print(f"\nMetadata written to: {output_file}")
    print(f"Total nodes: {len(all_nodes)} ({len(manual_nodes)} manual + {len(nodes)} generated)")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
