#!/usr/bin/env python3
"""
ComfyUI Proxy Node Generator

This script generates Nodetool proxy nodes from ComfyUI metadata.
Each proxy node wraps a ComfyUI node, handling type conversions and
providing a clean Nodetool interface.

Usage:
    python scripts/generate_proxy_nodes.py [--output-dir OUTPUT_DIR] [--categories CATEGORIES]

The generator:
1. Reads comfy_nodes_metadata.json
2. Generates BaseNode subclasses for each ComfyUI node
3. Handles type conversions between Nodetool and ComfyUI types
4. Organizes generated nodes by category
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from type_mappings import (
    get_nodetool_type,
    get_pydantic_field_type,
    get_field_constraints,
    get_required_imports,
    get_type_conversion_code,
)


class ProxyNodeGenerator:
    """Generates Nodetool proxy nodes from ComfyUI metadata."""
    
    def __init__(self, metadata_path: Path, output_dir: Path):
        """
        Initialize the generator.
        
        Args:
            metadata_path: Path to comfy_nodes_metadata.json
            output_dir: Output directory for generated nodes
        """
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load ComfyUI metadata from JSON file."""
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _sanitize_name(self, name: str) -> str:
        """Convert a name to a valid Python identifier."""
        # Replace special characters with underscores
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a number
        if name and name[0].isdigit():
            name = f"_{name}"
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name
    
    def _get_category_path(self, category: str) -> Path:
        """
        Get the file path for a category.
        
        Args:
            category: Node category (e.g., "sampling", "advanced/model")
            
        Returns:
            Path to the Python file for this category
        """
        # Sanitize category for filesystem
        cat_parts = category.split('/')
        cat_parts = [self._sanitize_name(part) for part in cat_parts]
        
        # Create directory structure
        cat_dir = self.output_dir
        for part in cat_parts[:-1]:
            cat_dir = cat_dir / part
        
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        # Use last part as filename
        filename = f"{cat_parts[-1]}.py" if len(cat_parts) > 0 else "uncategorized.py"
        return cat_dir / filename
    
    def _generate_class_name(self, node_id: str, class_name: str) -> str:
        """
        Generate a class name for a proxy node.
        
        Args:
            node_id: ComfyUI node ID
            class_name: Original ComfyUI class name
            
        Returns:
            Python class name
        """
        # Use the class name if available, otherwise use node_id
        name = class_name or node_id
        name = self._sanitize_name(name)
        
        # Ensure CamelCase
        if '_' in name:
            parts = name.split('_')
            name = ''.join(p.capitalize() for p in parts)
        
        # Avoid conflicts with ComfyUI imports
        if not name.startswith("Comfy"):
            name = f"Comfy{name}"
        
        return name
    
    def _generate_field_definition(
        self,
        param_name: str,
        param_info: Dict[str, Any],
        is_required: bool
    ) -> tuple[str, str, Set[str]]:
        """
        Generate a Pydantic field definition.
        
        Args:
            param_name: Parameter name
            param_info: Parameter info from metadata (dict or string)
            is_required: Whether the parameter is required
            
        Returns:
            Tuple of (field_type, field_definition, types_used)
        """
        types_used = set()
        
        # Handle case where param_info is just a string (type name)
        if isinstance(param_info, str):
            param_type = param_info
            config = {}
        else:
            # Get type info
            param_type = param_info.get("type", "Any")
            config = param_info.get("config", {})
        
        # Handle COMBO type (list of options)
        if param_type == "COMBO" or (isinstance(param_info, dict) and param_info.get("options")):
            options = param_info.get("options", []) if isinstance(param_info, dict) else []
            if options and isinstance(options, list):
                # Create an enum-like Field with choices
                field_type = "str"
                types_used.add("str")
            else:
                field_type = "str"
                types_used.add("str")
        else:
            field_type = get_pydantic_field_type(param_type, config, not is_required)
            types_used.add(field_type)
        
        # Get field constraints
        constraints = get_field_constraints(config)
        
        # Build Field() call
        field_parts = []
        
        # Add default value
        if "default" in constraints:
            default_val = constraints["default"]
            if isinstance(default_val, str):
                field_parts.append(f'default="{default_val}"')
            elif isinstance(default_val, bool):
                field_parts.append(f'default={str(default_val)}')
            else:
                field_parts.append(f'default={default_val}')
        elif not is_required:
            field_parts.append('default=None')
        else:
            field_parts.append('...')  # Required field
        
        # Add constraints
        for key, value in constraints.items():
            if key == "default":
                continue  # Already handled
            elif key == "description":
                field_parts.append(f'description="{value}"')
            elif key in ["ge", "le", "gt", "lt"]:
                field_parts.append(f'{key}={value}')
        
        field_def = f"Field({', '.join(field_parts)})"
        
        return field_type, field_def, types_used
    
    def _generate_node_class(self, node_info: Dict[str, Any]) -> Optional[str]:
        """
        Generate a proxy node class.
        
        Args:
            node_info: Node metadata
            
        Returns:
            Generated Python class code or None if generation fails
        """
        node_id = node_info.get("node_id")
        class_name = node_info.get("class_name")
        category = node_info.get("category") or "uncategorized"
        description = node_info.get("description", "")
        node_style = node_info.get("node_style", "v1")
        
        if not node_id or not class_name:
            return None
        
        # Skip deprecated nodes
        if node_info.get("deprecated", False):
            return None
        
        # Skip testing nodes
        if "_for_testing" in category:
            return None
        
        # Generate class name
        proxy_class_name = self._generate_class_name(node_id, class_name)
        
        # Collect all types used
        types_used = {"BaseNode", "ProcessingContext", "Field"}
        
        # Generate fields
        field_defs = []
        input_params = []
        
        if node_style == "v1":
            input_types = node_info.get("input_types", {})
            required_inputs = input_types.get("required", {})
            optional_inputs = input_types.get("optional", {})
            
            # Skip nodes with no inputs (might be incomplete metadata)
            if not required_inputs and not optional_inputs:
                return None
            
            # Required parameters
            for param_name, param_info in required_inputs.items():
                field_type, field_def, field_types = self._generate_field_definition(
                    param_name, param_info, is_required=True
                )
                types_used.update(field_types)
                safe_param_name = self._sanitize_name(param_name)
                field_defs.append(f"    {safe_param_name}: {field_type} = {field_def}")
                # Handle both dict and string param_info
                param_type = param_info.get("type", "Any") if isinstance(param_info, dict) else param_info
                input_params.append((safe_param_name, param_name, param_type))
            
            # Optional parameters
            for param_name, param_info in optional_inputs.items():
                field_type, field_def, field_types = self._generate_field_definition(
                    param_name, param_info, is_required=False
                )
                types_used.update(field_types)
                safe_param_name = self._sanitize_name(param_name)
                field_defs.append(f"    {safe_param_name}: {field_type} = {field_def}")
                # Handle both dict and string param_info
                param_type = param_info.get("type", "Any") if isinstance(param_info, dict) else param_info
                input_params.append((safe_param_name, param_name, param_type))
        
        # Generate return type
        return_types = node_info.get("return_types", [])
        if not return_types:
            return_type = "Any"
        elif len(return_types) == 1:
            return_type = get_nodetool_type(return_types[0])
            types_used.add(return_type)
        else:
            # Multiple returns - use tuple
            ret_types = [get_nodetool_type(rt) for rt in return_types]
            types_used.update(ret_types)
            return_type = f"Tuple[{', '.join(ret_types)}]"
            types_used.add("Tuple")
        
        # Generate imports
        imports = get_required_imports(types_used)
        
        # Generate process method
        process_method = self._generate_process_method(
            node_info, input_params, return_types
        )
        
        # Build class code
        class_code = []
        class_code.append(f'class {proxy_class_name}(BaseNode):')
        class_code.append(f'    """')
        if description:
            class_code.append(f'    {description}')
            class_code.append(f'    ')
        class_code.append(f'    ComfyUI Node: {node_id}')
        class_code.append(f'    Category: {category}')
        class_code.append(f'    """')
        class_code.append(f'')
        
        # Add fields
        if field_defs:
            class_code.extend(field_defs)
            class_code.append(f'')
        else:
            class_code.append(f'    pass')
            class_code.append(f'')
        
        # Add process method
        class_code.append(process_method)
        
        return '\n'.join(class_code)
    
    def _generate_process_method(
        self,
        node_info: Dict[str, Any],
        input_params: List[tuple],
        return_types: List[str]
    ) -> str:
        """Generate the process method for a proxy node."""
        node_id = node_info.get("node_id")
        class_name = node_info.get("class_name")
        function_name = node_info.get("function", "process")
        
        method_lines = []
        method_lines.append(f'    async def process(self, context: ProcessingContext) -> {get_nodetool_type(return_types[0]) if return_types else "Any"}:')
        method_lines.append(f'        """Execute the {node_id} ComfyUI node."""')
        method_lines.append(f'        from nodes import {class_name}')
        method_lines.append(f'')
        method_lines.append(f'        # Prepare inputs')
        method_lines.append(f'        inputs = {{}}')
        
        # Convert inputs
        for safe_name, orig_name, param_type in input_params:
            conversion = get_type_conversion_code(f"self.{safe_name}", param_type, "to_comfy")
            if conversion:
                method_lines.append(f'        if self.{safe_name} is not None:')
                method_lines.append(f'            inputs["{orig_name}"] = {conversion}')
            else:
                method_lines.append(f'        if self.{safe_name} is not None:')
                method_lines.append(f'            inputs["{orig_name}"] = self.{safe_name}')
        
        method_lines.append(f'')
        method_lines.append(f'        # Execute ComfyUI node')
        method_lines.append(f'        node = {class_name}()')
        method_lines.append(f'        result = node.{function_name}(**inputs)')
        method_lines.append(f'')
        
        # Convert output
        if return_types and len(return_types) > 0:
            ret_type = return_types[0]
            conversion = get_type_conversion_code("result[0]", ret_type, "from_comfy")
            if conversion:
                method_lines.append(f'        # Convert output')
                method_lines.append(f'        return {conversion}')
            else:
                method_lines.append(f'        return result[0] if isinstance(result, tuple) else result')
        else:
            method_lines.append(f'        return result')
        
        return '\n'.join(method_lines)
    
    def generate_all_nodes(self, categories: Optional[List[str]] = None):
        """
        Generate all proxy nodes.
        
        Args:
            categories: Optional list of categories to generate. If None, generate all.
        """
        nodes_by_category: Dict[str, List[Dict[str, Any]]] = {}
        
        # Group nodes by category
        for node in self.metadata.get("comfy_nodes", []):
            category = node.get("category") or "uncategorized"
            
            # Filter by categories if specified
            if categories and not any(cat in category for cat in categories):
                continue
            
            if category not in nodes_by_category:
                nodes_by_category[category] = []
            nodes_by_category[category].append(node)
        
        print(f"Generating proxy nodes for {len(nodes_by_category)} categories...")
        
        # Generate nodes for each category
        for category, nodes in sorted(nodes_by_category.items()):
            self._generate_category_file(category, nodes)
        
        print(f"Generated proxy nodes in {self.output_dir}")
    
    def _generate_category_file(self, category: str, nodes: List[Dict[str, Any]]):
        """Generate a Python file for a category of nodes."""
        output_path = self._get_category_path(category)
        
        print(f"  {category} -> {output_path.relative_to(self.output_dir)} ({len(nodes)} nodes)")
        
        # Generate file header
        file_lines = []
        file_lines.append('"""')
        file_lines.append(f'Generated ComfyUI proxy nodes for category: {category}')
        file_lines.append('')
        file_lines.append('This file is auto-generated. Do not edit manually.')
        file_lines.append('"""')
        file_lines.append('')
        
        # Add imports
        file_lines.append('from typing import Any, Optional, Tuple, List')
        file_lines.append('from pydantic import Field')
        file_lines.append('')
        file_lines.append('from nodetool.workflows.base_node import BaseNode')
        file_lines.append('from nodetool.workflows.processing_context import ProcessingContext')
        file_lines.append('from nodetool.metadata.types import ImageRef, AudioRef')
        file_lines.append('')
        file_lines.append('')
        
        # Generate classes
        generated_classes = []
        for node in nodes:
            class_code = self._generate_node_class(node)
            if class_code:
                generated_classes.append(class_code)
        
        # Add generated classes (with triple newlines between them)
        file_lines.append('\n\n\n'.join(generated_classes))
        
        # Add __all__ export
        class_names = []
        for node in nodes:
            category = node.get("category") or ""
            if node.get("deprecated", False) or "_for_testing" in category:
                continue
            class_name = self._generate_class_name(
                node.get("node_id"), node.get("class_name")
            )
            class_names.append(f'"{class_name}"')
        
        if class_names:
            file_lines.append('')
            file_lines.append('')
            file_lines.append(f'__all__ = [{", ".join(class_names)}]')
        
        # Write file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(file_lines))
        
        # Create __init__.py files
        self._create_init_files(output_path.parent)
    
    def _create_init_files(self, directory: Path):
        """Create __init__.py files for all parent directories."""
        while directory != self.output_dir.parent:
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Generated ComfyUI proxy nodes."""\n')
            directory = directory.parent
            if directory == self.output_dir.parent:
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Nodetool proxy nodes from ComfyUI metadata"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).parent.parent / "comfy_nodes_metadata.json",
        help="Path to comfy_nodes_metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "src" / "nodetool" / "nodes" / "comfy_generated",
        help="Output directory for generated nodes",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Categories to generate (default: all)",
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = ProxyNodeGenerator(args.metadata, args.output_dir)
    
    # Generate nodes
    generator.generate_all_nodes(args.categories)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
