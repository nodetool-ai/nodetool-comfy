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
import sys
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

    # Workflow JSON as class attribute
    lines.append(f"    _workflow_json: ClassVar[Dict[str, Any]] = {json.dumps(config.workflow_json)}")
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


def generate_base_class() -> str:
    """Generate the ComfyWorkflowNode base class."""
    return '''"""
ComfyUI Workflow Base Node

This module provides the base class for generated ComfyUI workflow nodes.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import tempfile
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from pydantic import Field

from nodetool.metadata.types import ImageRef, AudioRef, VideoRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class ComfyWorkflowNode(BaseNode):
    """
    Base class for ComfyUI workflow nodes.
    
    This class handles:
    - Converting nodetool types (ImageRef, etc.) to ComfyUI inputs
    - Modifying the workflow JSON with user inputs
    - Executing the workflow via ComfyUI
    - Converting outputs back to nodetool types
    """
    
    # These should be overridden by subclasses
    _workflow_json: ClassVar[Dict[str, Any]] = {}
    _input_mappings: ClassVar[Dict[str, Dict[str, str]]] = {}
    _output_mappings: ClassVar[Dict[str, Dict[str, Any]]] = {}
    
    @classmethod
    def get_title(cls) -> str:
        """Return the display title for this node."""
        return cls.__name__
    
    @classmethod
    def get_namespace(cls) -> str:
        """Return the namespace for this node."""
        return "comfy.workflows"
    
    async def _prepare_inputs(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Prepare inputs for the ComfyUI workflow.
        
        Converts nodetool types to paths/values that ComfyUI can use.
        Returns a dict mapping input names to their prepared values.
        """
        prepared = {}
        
        for input_name, mapping in self._input_mappings.items():
            value = getattr(self, input_name, None)
            
            if value is None:
                continue
            
            # Handle ImageRef - write to temp file and use path
            if isinstance(value, ImageRef) and not value.is_empty():
                pil_image = await context.image_to_pil(value)
                # Save to ComfyUI input folder
                temp_path = await self._save_input_image(pil_image, context)
                prepared[input_name] = temp_path
            # Handle other types directly
            else:
                prepared[input_name] = value
        
        return prepared
    
    async def _save_input_image(
        self, 
        image: PIL.Image.Image, 
        context: ProcessingContext
    ) -> str:
        """Save an input image and return the path for ComfyUI."""
        import folder_paths
        
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"nodetool_input_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(input_dir, filename)
        
        # Save image
        image.save(filepath, format="PNG")
        
        return filename
    
    def _modify_workflow(self, prepared_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify the workflow JSON with prepared input values.
        
        Returns a copy of the workflow with inputs substituted.
        """
        workflow = copy.deepcopy(self._workflow_json)
        
        for input_name, value in prepared_inputs.items():
            mapping = self._input_mappings.get(input_name)
            if not mapping:
                continue
            
            node_id = mapping["node"]
            field_name = mapping["field"]
            
            if node_id in workflow:
                if "inputs" not in workflow[node_id]:
                    workflow[node_id]["inputs"] = {}
                workflow[node_id]["inputs"][field_name] = value
        
        return workflow
    
    async def _execute_comfy_workflow(
        self, 
        workflow: Dict[str, Any],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """
        Execute a ComfyUI workflow and return the results.
        
        This method executes the workflow using ComfyUI's internal execution engine.
        """
        import nodes
        from comfy_execution.graph import DynamicPrompt, ExecutionList
        from comfy_execution.caching import HierarchicalCache
        
        # Create execution context
        dynprompt = DynamicPrompt(workflow)
        cache = HierarchicalCache()
        execution_list = ExecutionList(dynprompt, cache)
        
        # Find output nodes
        output_node_ids = set()
        for output_name, mapping in self._output_mappings.items():
            output_node_ids.add(mapping["node"])
        
        # Add output nodes to execution list
        for node_id in output_node_ids:
            execution_list.add_node(node_id, include_lazy=True)
        
        results = {}
        
        # Execute nodes in topological order
        while not execution_list.is_empty():
            node_id, error, exception = await execution_list.stage_node_execution()
            
            if error:
                raise RuntimeError(f"Workflow execution error: {error}")
            
            if node_id is None:
                break
            
            # Get node info
            node_info = dynprompt.get_node(node_id)
            class_type = node_info["class_type"]
            
            if class_type not in nodes.NODE_CLASS_MAPPINGS:
                raise RuntimeError(f"Unknown node type: {class_type}")
            
            node_class = nodes.NODE_CLASS_MAPPINGS[class_type]
            
            # Gather inputs for this node
            node_inputs = {}
            for input_name, input_value in node_info.get("inputs", {}).items():
                if isinstance(input_value, list) and len(input_value) == 2:
                    # This is a link to another node's output
                    from_node_id, from_output_idx = input_value
                    from_result = cache.get(from_node_id)
                    if from_result is not None:
                        node_inputs[input_name] = from_result[from_output_idx]
                else:
                    node_inputs[input_name] = input_value
            
            # Create and execute node
            node_instance = node_class()
            func_name = getattr(node_class, "FUNCTION", "execute")
            func = getattr(node_instance, func_name)
            
            # Execute the node
            if asyncio.iscoroutinefunction(func):
                result = await func(**node_inputs)
            else:
                result = func(**node_inputs)
            
            # Store result in cache
            if not isinstance(result, tuple):
                result = (result,)
            cache.set(node_id, result)
            
            # Store if this is an output node
            if node_id in output_node_ids:
                results[node_id] = result
            
            execution_list.complete_node_execution()
        
        return results
    
    async def _process_outputs(
        self, 
        results: Dict[str, Any],
        context: ProcessingContext
    ) -> Any:
        """
        Process ComfyUI outputs and convert to nodetool types.
        
        Returns the appropriate type(s) based on _output_mappings.
        """
        outputs = []
        
        for output_name, mapping in self._output_mappings.items():
            node_id = mapping["node"]
            output_idx = mapping.get("index", 0)
            
            result = results.get(node_id)
            if result is None:
                outputs.append(None)
                continue
            
            output_value = result[output_idx] if isinstance(result, tuple) else result
            
            # Convert based on output type
            # Check if it's a tensor (image from SaveImage/PreviewImage)
            if isinstance(output_value, dict) and "images" in output_value:
                # SaveImage returns a dict with images list
                images = output_value["images"]
                if images:
                    # Load the first image
                    import folder_paths
                    output_dir = folder_paths.get_output_directory()
                    image_info = images[0]
                    image_path = os.path.join(output_dir, image_info.get("subfolder", ""), image_info["filename"])
                    
                    pil_image = PIL.Image.open(image_path)
                    image_ref = await context.image_from_pil(pil_image)
                    outputs.append(image_ref)
            elif isinstance(output_value, torch.Tensor):
                # Direct tensor output - convert to ImageRef
                image_ref = await context.image_from_tensor(output_value)
                outputs.append(image_ref)
            elif isinstance(output_value, np.ndarray):
                # Numpy array - convert to tensor then ImageRef
                tensor = torch.from_numpy(output_value)
                image_ref = await context.image_from_tensor(tensor)
                outputs.append(image_ref)
            else:
                outputs.append(output_value)
        
        # Return single value or tuple based on number of outputs
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
    
    async def _execute_workflow(self, context: ProcessingContext) -> Any:
        """
        Main execution method for workflow nodes.
        
        This orchestrates the full workflow:
        1. Prepare inputs (convert nodetool types)
        2. Modify workflow JSON with inputs
        3. Execute via ComfyUI
        4. Process and return outputs
        """
        import torch
        
        with torch.inference_mode():
            # Prepare inputs
            prepared_inputs = await self._prepare_inputs(context)
            
            # Modify workflow with inputs
            modified_workflow = self._modify_workflow(prepared_inputs)
            
            # Execute workflow
            results = await self._execute_comfy_workflow(modified_workflow, context)
            
            # Process and return outputs
            return await self._process_outputs(results, context)
'''


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

    # Generate base class
    base_class_path = output_dir / "workflow_base.py"
    with open(base_class_path, "w", encoding="utf-8") as f:
        f.write(generate_base_class())
    print(f"Generated: {base_class_path}")

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
