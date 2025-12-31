#!/usr/bin/env python3
"""
ComfyUI Node Parser

This script automatically parses ComfyUI nodes from the ComfyUI submodule
and extracts their metadata for use in nodetool-comfy.

It reads node definitions from:
- ComfyUI/nodes.py (base nodes)
- ComfyUI/comfy_extras/*.py (extra nodes)

And supports two node definition styles:
1. V1/Classic style: NODE_CLASS_MAPPINGS dict with INPUT_TYPES method
2. V3 style: io.ComfyNode subclasses with define_schema method

Extracts:
- Node class name / node_id
- Inputs (INPUT_TYPES or schema inputs)
- Outputs (RETURN_TYPES or schema outputs)  
- Category
- Description
- Function name
"""

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_comfyui_path() -> Path:
    """Get the path to the ComfyUI submodule."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    comfyui_path = repo_root / "ComfyUI"
    
    if not comfyui_path.exists():
        raise RuntimeError(
            f"ComfyUI submodule not found at {comfyui_path}. "
            "Please run: git submodule update --init"
        )
    
    return comfyui_path


def extract_node_class_mappings(file_path: Path) -> Dict[str, str]:
    """
    Extract NODE_CLASS_MAPPINGS dictionary from a Python file using AST.
    
    Returns a dict mapping node name to class name.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"Syntax error parsing {file_path}: {e}")
        return {}
    
    mappings = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS":
                    if isinstance(node.value, ast.Dict):
                        for key, value in zip(node.value.keys, node.value.values):
                            if isinstance(key, ast.Constant) and isinstance(value, ast.Name):
                                mappings[key.value] = value.id
    
    return mappings


def extract_class_info(file_path: Path, class_name: str) -> Optional[Dict[str, Any]]:
    """
    Extract information about a node class from a Python file using AST.
    
    Returns a dict with:
    - input_types: The INPUT_TYPES definition
    - return_types: The RETURN_TYPES tuple
    - category: The CATEGORY string
    - description: The DESCRIPTION string
    - function: The FUNCTION string
    - deprecated: Whether DEPRECATED is True
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            info = {
                "class_name": class_name,
                "input_types": None,
                "return_types": None,
                "return_names": None,
                "category": None,
                "description": None,
                "function": None,
                "deprecated": False,
                "output_node": False,
            }
            
            for item in node.body:
                # Class attributes
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            attr_name = target.id
                            
                            if attr_name == "RETURN_TYPES":
                                info["return_types"] = _extract_tuple_strings(item.value)
                            elif attr_name == "RETURN_NAMES":
                                info["return_names"] = _extract_tuple_strings(item.value)
                            elif attr_name == "CATEGORY":
                                if isinstance(item.value, ast.Constant):
                                    info["category"] = item.value.value
                            elif attr_name == "DESCRIPTION":
                                if isinstance(item.value, ast.Constant):
                                    info["description"] = item.value.value
                            elif attr_name == "FUNCTION":
                                if isinstance(item.value, ast.Constant):
                                    info["function"] = item.value.value
                            elif attr_name == "DEPRECATED":
                                if isinstance(item.value, ast.Constant):
                                    info["deprecated"] = item.value.value
                            elif attr_name == "OUTPUT_NODE":
                                if isinstance(item.value, ast.Constant):
                                    info["output_node"] = item.value.value
                
                # INPUT_TYPES method
                if isinstance(item, ast.FunctionDef) and item.name == "INPUT_TYPES":
                    info["input_types"] = _extract_input_types(item, source)
            
            return info
    
    return None


def _extract_tuple_strings(node: ast.AST) -> List[str]:
    """Extract strings from a tuple AST node."""
    result = []
    
    if isinstance(node, ast.Tuple):
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                result.append(elt.value)
            elif isinstance(elt, ast.Attribute):
                # Handle IO.STRING style references
                result.append(f"{_get_attribute_string(elt)}")
    
    return result


def _get_attribute_string(node: ast.Attribute) -> str:
    """Convert an attribute node to a string like 'IO.STRING'."""
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _extract_input_types(func_def: ast.FunctionDef, source: str) -> Optional[Dict[str, Any]]:
    """
    Extract INPUT_TYPES dictionary from the function definition.
    
    This is a simplified extraction that captures the structure.
    """
    # Get source code segment for the function
    for node in ast.walk(func_def):
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Dict):
                return _extract_dict_structure(node.value)
    
    return None


def _extract_dict_structure(node: ast.Dict) -> Dict[str, Any]:
    """
    Extract a dictionary structure from an AST Dict node.
    
    Returns a simplified representation of the dict.
    """
    result = {}
    
    for key, value in zip(node.keys, node.values):
        if key is None:
            continue
            
        key_str = None
        if isinstance(key, ast.Constant):
            key_str = key.value
        elif isinstance(key, ast.Name):
            key_str = key.id
        
        if key_str is None:
            continue
        
        if isinstance(value, ast.Dict):
            result[key_str] = _extract_dict_structure(value)
        elif isinstance(value, ast.Tuple):
            result[key_str] = _extract_input_spec(value)
        elif isinstance(value, ast.Constant):
            result[key_str] = value.value
        elif isinstance(value, ast.Name):
            result[key_str] = {"_ref": value.id}
        elif isinstance(value, ast.List):
            result[key_str] = _extract_list_values(value)
    
    return result


def _extract_input_spec(node: ast.Tuple) -> Dict[str, Any]:
    """
    Extract an input specification from a tuple like:
    ("INT", {"default": 0, "min": 0, "max": 100})
    """
    spec = {}
    
    if len(node.elts) >= 1:
        first = node.elts[0]
        if isinstance(first, ast.Constant):
            spec["type"] = first.value
        elif isinstance(first, ast.Attribute):
            spec["type"] = _get_attribute_string(first)
        elif isinstance(first, ast.Name):
            spec["type"] = f"${first.id}"  # Reference to variable
        elif isinstance(first, ast.List):
            spec["type"] = "COMBO"
            spec["options"] = _extract_list_values(first)
        elif isinstance(first, ast.Tuple):
            spec["type"] = "COMBO"
            spec["options"] = _extract_list_values(first)
    
    if len(node.elts) >= 2:
        second = node.elts[1]
        if isinstance(second, ast.Dict):
            spec["config"] = _extract_dict_structure(second)
    
    return spec


def _extract_list_values(node: ast.AST) -> List[Any]:
    """Extract values from a list or tuple AST node."""
    result = []
    
    if isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            if isinstance(elt, ast.Constant):
                result.append(elt.value)
            elif isinstance(elt, ast.Name):
                result.append(f"${elt.id}")
    
    return result


def extract_v3_nodes(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract V3-style nodes (io.ComfyNode subclasses with define_schema).
    
    Returns a list of node info dictionaries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    nodes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it's a ComfyNode subclass
            is_comfy_node = False
            for base in node.bases:
                if isinstance(base, ast.Attribute):
                    if base.attr == "ComfyNode":
                        is_comfy_node = True
                        break
            
            if not is_comfy_node:
                continue
            
            # Look for define_schema method
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "define_schema":
                    schema_info = _extract_v3_schema(item)
                    if schema_info:
                        schema_info["class_name"] = node.name
                        schema_info["node_style"] = "v3"
                        nodes.append(schema_info)
                    break
    
    return nodes


def _extract_v3_schema(func_def: ast.FunctionDef) -> Optional[Dict[str, Any]]:
    """
    Extract schema information from a define_schema method.
    """
    for stmt in ast.walk(func_def):
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            # Check if it's io.Schema(...)
            if isinstance(call.func, ast.Attribute) and call.func.attr == "Schema":
                return _extract_schema_call(call)
    return None


def _extract_schema_call(call: ast.Call) -> Dict[str, Any]:
    """
    Extract information from an io.Schema(...) call.
    """
    info = {
        "node_name": None,
        "display_name": None,
        "category": None,
        "description": None,
        "inputs": [],
        "outputs": [],
    }
    
    for keyword in call.keywords:
        if keyword.arg == "node_id":
            if isinstance(keyword.value, ast.Constant):
                info["node_name"] = keyword.value.value
        elif keyword.arg == "display_name":
            if isinstance(keyword.value, ast.Constant):
                info["display_name"] = keyword.value.value
        elif keyword.arg == "category":
            if isinstance(keyword.value, ast.Constant):
                info["category"] = keyword.value.value
        elif keyword.arg == "description":
            if isinstance(keyword.value, ast.Constant):
                info["description"] = keyword.value.value
        elif keyword.arg == "inputs":
            if isinstance(keyword.value, ast.List):
                info["inputs"] = _extract_v3_io_list(keyword.value)
        elif keyword.arg == "outputs":
            if isinstance(keyword.value, ast.List):
                info["outputs"] = _extract_v3_io_list(keyword.value)
    
    return info


def _extract_v3_io_list(node: ast.List) -> List[Dict[str, Any]]:
    """
    Extract input/output definitions from a list of io.Type.Input/Output calls.
    """
    result = []
    
    for elt in node.elts:
        if isinstance(elt, ast.Call):
            io_info = _extract_v3_io_call(elt)
            if io_info:
                result.append(io_info)
    
    return result


def _extract_v3_io_call(call: ast.Call) -> Optional[Dict[str, Any]]:
    """
    Extract information from an io.Type.Input(...) or io.Type.Output() call.
    """
    info = {}
    
    # Get the type and direction (Input/Output)
    if isinstance(call.func, ast.Attribute):
        direction = call.func.attr  # "Input" or "Output"
        info["direction"] = direction.lower()
        
        # Get the type (e.g., "Int", "Float", "Clip", etc.)
        if isinstance(call.func.value, ast.Attribute):
            io_type = call.func.value.attr
            info["type"] = io_type
    
    # Get the name (first positional arg for inputs)
    if call.args:
        first_arg = call.args[0]
        if isinstance(first_arg, ast.Constant):
            info["name"] = first_arg.value
    
    # Get keyword arguments like default, min, max, etc.
    for keyword in call.keywords:
        if keyword.arg and isinstance(keyword.value, ast.Constant):
            info[keyword.arg] = keyword.value.value
    
    return info if info else None


def parse_nodes_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse all nodes from a Python file.
    
    Supports both V1 (NODE_CLASS_MAPPINGS) and V3 (io.ComfyNode) styles.
    
    Returns a list of node info dictionaries.
    """
    nodes = []
    
    # Try V1 style first (NODE_CLASS_MAPPINGS)
    mappings = extract_node_class_mappings(file_path)
    for node_name, class_name in mappings.items():
        info = extract_class_info(file_path, class_name)
        if info:
            info["node_name"] = node_name
            info["source_file"] = str(file_path.name)
            info["node_style"] = "v1"
            nodes.append(info)
    
    # Try V3 style (io.ComfyNode subclasses)
    v3_nodes = extract_v3_nodes(file_path)
    for info in v3_nodes:
        info["source_file"] = str(file_path.name)
        nodes.append(info)
    
    return nodes


def get_all_comfy_nodes(comfyui_path: Path) -> List[Dict[str, Any]]:
    """
    Get all ComfyUI nodes from the submodule.
    
    Parses:
    - nodes.py (base nodes)
    - comfy_extras/*.py (extra nodes)
    """
    all_nodes = []
    
    # Parse base nodes
    nodes_py = comfyui_path / "nodes.py"
    if nodes_py.exists():
        print(f"Parsing {nodes_py}...")
        nodes = parse_nodes_file(nodes_py)
        for node in nodes:
            node["source"] = "nodes.py"
        all_nodes.extend(nodes)
        print(f"  Found {len(nodes)} nodes")
    
    # Parse extra nodes
    extras_dir = comfyui_path / "comfy_extras"
    if extras_dir.exists():
        for py_file in sorted(extras_dir.glob("nodes_*.py")):
            print(f"Parsing {py_file}...")
            nodes = parse_nodes_file(py_file)
            for node in nodes:
                node["source"] = f"comfy_extras/{py_file.name}"
            all_nodes.extend(nodes)
            print(f"  Found {len(nodes)} nodes")
    
    return all_nodes


def generate_metadata(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate nodetool-comfy compatible metadata from parsed nodes.
    """
    metadata = {
        "name": "nodetool-comfy",
        "description": "ComfyUI nodes for Nodetool (auto-generated from ComfyUI source)",
        "version": "auto-generated",
        "comfy_nodes": []
    }
    
    for node in nodes:
        node_style = node.get("node_style", "v1")
        
        if node_style == "v3":
            # V3 style node
            node_meta = {
                "node_id": node["node_name"],
                "class_name": node.get("class_name"),
                "display_name": node.get("display_name"),
                "category": node.get("category", "uncategorized"),
                "description": node.get("description"),
                "source": node.get("source"),
                "node_style": "v3",
                "inputs": node.get("inputs", []),
                "outputs": node.get("outputs", []),
            }
        else:
            # V1 style node
            node_meta = {
                "node_id": node["node_name"],
                "class_name": node["class_name"],
                "category": node.get("category", "uncategorized"),
                "description": node.get("description"),
                "function": node.get("function"),
                "deprecated": node.get("deprecated", False),
                "output_node": node.get("output_node", False),
                "source": node.get("source"),
                "node_style": "v1",
                "input_types": node.get("input_types"),
                "return_types": node.get("return_types"),
                "return_names": node.get("return_names"),
            }
        
        metadata["comfy_nodes"].append(node_meta)
    
    return metadata


def main():
    """Main entry point."""
    comfyui_path = get_comfyui_path()
    print(f"ComfyUI path: {comfyui_path}")
    
    nodes = get_all_comfy_nodes(comfyui_path)
    print(f"\nTotal nodes found: {len(nodes)}")
    
    metadata = generate_metadata(nodes)
    
    # Output metadata
    output_path = comfyui_path.parent / "comfy_nodes_metadata.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata written to: {output_path}")
    
    # Print summary by category
    categories = {}
    for node in metadata["comfy_nodes"]:
        cat = node.get("category") or "uncategorized"
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    print("\nNodes by category:")
    for cat in sorted(categories.keys(), key=lambda x: x or ""):
        print(f"  {cat}: {categories[cat]}")
    
    # Print summary by node style
    styles = {}
    for node in nodes:
        style = node.get("node_style", "unknown")
        if style not in styles:
            styles[style] = 0
        styles[style] += 1
    
    print("\nNodes by style:")
    for style, count in styles.items():
        print(f"  {style}: {count}")


if __name__ == "__main__":
    main()
