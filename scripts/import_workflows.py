#!/usr/bin/env python3
"""
ComfyUI Workflow Importer

This script imports ComfyUI workflows as nodetool nodes.

How it works:
1. Each ComfyUI workflow is defined in YAML files
2. Each YAML generates a Python module with node classes
3. Each workflow YAML defines inputs (e.g., text_node.text, load_image.image) and outputs (e.g., SaveImage)
4. The workflow API JSON files live alongside the YAML files
5. Script generates node classes deriving from ComfyWorkflowNode base class
6. Generated nodes have Pydantic fields for inputs and OutputType matching workflow outputs
7. The node handles conversion between nodetool type system (ImageRef, etc.) and ComfyUI types
8. Nodetool manages the ComfyUI server - modified workflow JSON is posted for processing

Usage:
    python scripts/import_workflows.py [workflow_dir]

The workflow_dir should contain:
    - *.yaml files defining workflow configurations
    - *.json files with ComfyUI workflow API format (same base name as YAML)

Example YAML format:
    name: "MyWorkflow"
    description: "Generate images with SDXL"
    category: "comfy.workflows"
    
    inputs:
      prompt:
        node: "6"  # Node ID in workflow JSON
        field: "text"
        type: str
        default: ""
        description: "The prompt for generation"
      
      image:
        node: "10"
        field: "image"
        type: ImageRef
        description: "Input image"
    
    outputs:
      image:
        node: "9"  # SaveImage node ID
        type: ImageRef
        description: "Generated image"
"""

import argparse
import json
import os
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Type mapping from YAML config types to Python/Pydantic types
TYPE_MAPPING = {
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
    "number": "float",
    "bool": "bool",
    "boolean": "bool",
    "ImageRef": "ImageRef",
    "image": "ImageRef",
    "AudioRef": "AudioRef",
    "audio": "AudioRef",
    "VideoRef": "VideoRef",
    "video": "VideoRef",
    "list": "list",
    "dict": "dict",
}

# Default values for types
TYPE_DEFAULTS = {
    "str": '""',
    "int": "0",
    "float": "0.0",
    "bool": "False",
    "ImageRef": "ImageRef()",
    "AudioRef": "AudioRef()",
    "VideoRef": "VideoRef()",
    "list": "[]",
    "dict": "{}",
}


@dataclass
class WorkflowInput:
    """Represents an input field for a workflow node."""

    name: str
    node_id: str
    field_name: str
    python_type: str
    default: Any = None
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[str]] = None


@dataclass
class WorkflowOutput:
    """Represents an output from a workflow node."""

    name: str
    node_id: str
    python_type: str
    description: str = ""
    output_index: int = 0  # For nodes with multiple outputs


@dataclass
class WorkflowConfig:
    """Configuration for a ComfyUI workflow to be converted to a nodetool node."""

    name: str
    description: str
    category: str
    workflow_json: Dict[str, Any]
    inputs: List[WorkflowInput] = field(default_factory=list)
    outputs: List[WorkflowOutput] = field(default_factory=list)
    class_name: Optional[str] = None


def snake_to_pascal(name: str) -> str:
    """Convert snake_case or kebab-case to PascalCase."""
    # Replace hyphens with underscores first
    name = name.replace("-", "_")
    # Split on underscores and capitalize each part
    parts = name.split("_")
    return "".join(part.capitalize() for part in parts)


def pascal_to_snake(name: str) -> str:
    """Convert PascalCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def parse_input_config(name: str, config: Dict[str, Any]) -> WorkflowInput:
    """Parse input configuration from YAML."""
    raw_type = config.get("type", "str")
    python_type = TYPE_MAPPING.get(raw_type, raw_type)

    return WorkflowInput(
        name=name,
        node_id=str(config.get("node", "")),
        field_name=config.get("field", name),
        python_type=python_type,
        default=config.get("default"),
        description=config.get("description", ""),
        min_value=config.get("min"),
        max_value=config.get("max"),
        choices=config.get("choices"),
    )


def parse_output_config(name: str, config: Dict[str, Any]) -> WorkflowOutput:
    """Parse output configuration from YAML."""
    raw_type = config.get("type", "ImageRef")
    python_type = TYPE_MAPPING.get(raw_type, raw_type)

    return WorkflowOutput(
        name=name,
        node_id=str(config.get("node", "")),
        python_type=python_type,
        description=config.get("description", ""),
        output_index=config.get("output_index", 0),
    )


def load_workflow_config(yaml_path: Path) -> Optional[WorkflowConfig]:
    """Load a workflow configuration from a YAML file and its corresponding JSON."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML {yaml_path}: {e}")
        return None

    # Load the corresponding JSON file
    json_path = yaml_path.with_suffix(".json")
    if not json_path.exists():
        print(f"Warning: No JSON file found for {yaml_path}")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            workflow_json = json.load(f)
    except Exception as e:
        print(f"Error loading JSON {json_path}: {e}")
        return None

    # Parse configuration
    name = yaml_config.get("name", yaml_path.stem)
    description = yaml_config.get("description", "")
    category = yaml_config.get("category", "comfy.workflows")

    # Parse inputs
    inputs = []
    for input_name, input_config in yaml_config.get("inputs", {}).items():
        if isinstance(input_config, dict):
            inputs.append(parse_input_config(input_name, input_config))

    # Parse outputs
    outputs = []
    for output_name, output_config in yaml_config.get("outputs", {}).items():
        if isinstance(output_config, dict):
            outputs.append(parse_output_config(output_name, output_config))

    # Generate class name if not specified
    class_name = yaml_config.get("class_name") or snake_to_pascal(name)

    return WorkflowConfig(
        name=name,
        description=description,
        category=category,
        workflow_json=workflow_json,
        inputs=inputs,
        outputs=outputs,
        class_name=class_name,
    )


def generate_field_definition(inp: WorkflowInput) -> str:
    """Generate a Pydantic Field definition for an input."""
    parts = []

    # Determine default value
    if inp.default is not None:
        if inp.python_type == "str":
            default = f'"{inp.default}"'
        elif inp.python_type in ("ImageRef", "AudioRef", "VideoRef"):
            default = f"{inp.python_type}()"
        else:
            default = repr(inp.default)
    else:
        default = TYPE_DEFAULTS.get(inp.python_type, "None")

    parts.append(f"default={default}")

    # Add constraints
    if inp.min_value is not None:
        parts.append(f"ge={inp.min_value}")
    if inp.max_value is not None:
        parts.append(f"le={inp.max_value}")

    # Add description
    if inp.description:
        parts.append(f'description="{inp.description}"')

    return f"Field({', '.join(parts)})"


def generate_output_type(outputs: List[WorkflowOutput]) -> str:
    """Generate the output type annotation for a workflow node."""
    if len(outputs) == 0:
        return "None"
    elif len(outputs) == 1:
        return outputs[0].python_type
    else:
        # Multiple outputs - create a TypedDict or use a tuple
        types = [out.python_type for out in outputs]
        return f"tuple[{', '.join(types)}]"


def generate_node_class(config: WorkflowConfig) -> str:
    """Generate a Python class for a workflow node."""
    lines = []

    # Class definition
    lines.append(f"class {config.class_name}(ComfyWorkflowNode):")

    # Docstring
    doc_lines = ['    """']
    doc_lines.append(f"    {config.description}")
    doc_lines.append(f"    ")
    doc_lines.append(f"    Generated from ComfyUI workflow.")
    doc_lines.append('    """')
    lines.append("\n".join(doc_lines))
    lines.append("")

    # Workflow JSON as class attribute - format for readability
    workflow_json_str = json.dumps(config.workflow_json, indent=4)
    # Use textwrap.indent for clean indentation (skip first line since it's on same line as assignment)
    workflow_json_lines = workflow_json_str.split("\n")
    first_line = workflow_json_lines[0]
    rest_lines = "\n".join(workflow_json_lines[1:])
    indented_rest = textwrap.indent(rest_lines, "        ") if rest_lines else ""
    workflow_json_indented = first_line + ("\n" + indented_rest if indented_rest else "")
    lines.append(f"    _workflow_json: ClassVar[Dict[str, Any]] = {workflow_json_indented}")
    lines.append("")

    # Input field mappings
    input_mappings = {}
    for inp in config.inputs:
        input_mappings[inp.name] = {"node": inp.node_id, "field": inp.field_name}
    lines.append(f"    _input_mappings: ClassVar[Dict[str, Dict[str, str]]] = {repr(input_mappings)}")
    lines.append("")

    # Output node mappings
    output_mappings = {}
    for out in config.outputs:
        output_mappings[out.name] = {"node": out.node_id, "index": out.output_index}
    lines.append(f"    _output_mappings: ClassVar[Dict[str, Dict[str, Any]]] = {repr(output_mappings)}")
    lines.append("")

    # Pydantic fields for inputs
    for inp in config.inputs:
        field_def = generate_field_definition(inp)
        lines.append(f"    {inp.name}: {inp.python_type} = {field_def}")

    if not config.inputs:
        lines.append("    pass")

    lines.append("")

    # get_title class method
    lines.append("    @classmethod")
    lines.append("    def get_title(cls) -> str:")
    lines.append(f'        return "{config.name}"')
    lines.append("")

    # get_namespace class method  
    category_parts = config.category.split(".")
    namespace = ".".join(category_parts[:-1]) if len(category_parts) > 1 else config.category
    lines.append("    @classmethod")
    lines.append("    def get_namespace(cls) -> str:")
    lines.append(f'        return "{namespace}"')
    lines.append("")

    # Process method with output type
    output_type = generate_output_type(config.outputs)
    lines.append(f"    async def process(self, context: ProcessingContext) -> {output_type}:")
    lines.append("        return await self._execute_workflow(context)")
    lines.append("")

    return "\n".join(lines)


def generate_module_code(configs: List[WorkflowConfig], module_name: str) -> str:
    """Generate a complete Python module with workflow node classes."""
    lines = []

    # Module docstring
    lines.append('"""')
    lines.append(f"ComfyUI Workflow Nodes - {module_name}")
    lines.append("")
    lines.append("Auto-generated from ComfyUI workflow definitions.")
    lines.append("Do not edit manually - regenerate using import_workflows.py")
    lines.append('"""')
    lines.append("")

    # Imports
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import json")
    lines.append("from typing import Any, ClassVar, Dict, Optional")
    lines.append("")
    lines.append("from pydantic import Field")
    lines.append("")
    lines.append("from nodetool.metadata.types import ImageRef, AudioRef, VideoRef")
    lines.append("from nodetool.workflows.base_node import BaseNode")
    lines.append("from nodetool.workflows.processing_context import ProcessingContext")
    lines.append("")
    lines.append("from nodetool.nodes.comfy.workflow_base import ComfyWorkflowNode")
    lines.append("")
    lines.append("")

    # Generate classes
    for config in configs:
        lines.append(generate_node_class(config))
        lines.append("")

    # __all__ export
    class_names = [config.class_name for config in configs]
    lines.append(f"__all__ = {repr(class_names)}")
    lines.append("")

    return "\n".join(lines)


def process_workflow_directory(workflow_dir: Path, output_dir: Path) -> None:
    """Process all workflows in a directory and generate Python modules."""
    # Find all YAML files
    yaml_files = list(workflow_dir.glob("*.yaml")) + list(workflow_dir.glob("*.yml"))

    if not yaml_files:
        print(f"No YAML files found in {workflow_dir}")
        return

    # Group workflows by module (based on subdirectory or config)
    configs_by_module: Dict[str, List[WorkflowConfig]] = {}

    for yaml_path in yaml_files:
        config = load_workflow_config(yaml_path)
        if config is None:
            continue

        # Determine module name from category or filename
        category_parts = config.category.split(".")
        if len(category_parts) >= 2:
            module_name = category_parts[-1]
        else:
            module_name = yaml_path.stem

        if module_name not in configs_by_module:
            configs_by_module[module_name] = []
        configs_by_module[module_name].append(config)

    # Generate modules
    os.makedirs(output_dir, exist_ok=True)

    # Copy base class from the source location if not already present
    # The base class file is maintained separately for better maintainability
    base_class_source = Path(__file__).parent.parent / "src" / "nodetool" / "nodes" / "comfy" / "workflow_base.py"
    base_class_dest = output_dir / "workflow_base.py"
    
    if base_class_source.exists() and base_class_source != base_class_dest:
        shutil.copy2(base_class_source, base_class_dest)
        print(f"Copied base class: {base_class_dest}")
    elif not base_class_dest.exists():
        print(f"Warning: Base class not found at {base_class_source}")
        print("Generated modules will need workflow_base.py to be available")

    for module_name, configs in configs_by_module.items():
        module_code = generate_module_code(configs, module_name)
        module_path = output_dir / f"{module_name}.py"

        with open(module_path, "w", encoding="utf-8") as f:
            f.write(module_code)

        print(f"Generated: {module_path} ({len(configs)} workflow(s))")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import ComfyUI workflows as nodetool nodes"
    )
    parser.add_argument(
        "workflow_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).parent.parent / "workflows",
        help="Directory containing workflow YAML and JSON files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for generated Python modules",
    )

    args = parser.parse_args()

    if not args.workflow_dir.exists():
        print(f"Workflow directory not found: {args.workflow_dir}")
        sys.exit(1)

    output_dir = args.output or (Path(__file__).parent.parent / "src" / "nodetool" / "nodes" / "comfy")

    print(f"Processing workflows from: {args.workflow_dir}")
    print(f"Output directory: {output_dir}")
    print()

    process_workflow_directory(args.workflow_dir, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
