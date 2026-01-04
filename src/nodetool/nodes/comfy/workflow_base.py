"""
ComfyUI Workflow Base Node

This module provides the base class for generated ComfyUI workflow nodes.
It handles the conversion between nodetool types and ComfyUI types,
workflow execution, and output processing.
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
    - Executing the workflow via ComfyUI's internal execution engine
    - Converting outputs back to nodetool types
    
    Subclasses should define:
    - _workflow_json: The ComfyUI workflow API JSON
    - _input_mappings: Maps input field names to workflow node IDs and fields
    - _output_mappings: Maps output names to workflow node IDs
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
            # Handle AudioRef - write to temp file and use path
            elif isinstance(value, AudioRef) and not value.is_empty():
                temp_path = await self._save_input_audio(value, context)
                prepared[input_name] = temp_path
            # Handle VideoRef - write to temp file and use path
            elif isinstance(value, VideoRef) and not value.is_empty():
                temp_path = await self._save_input_video(value, context)
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
        """Save an input image and return the filename for ComfyUI."""
        import folder_paths
        
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"nodetool_input_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(input_dir, filename)
        
        # Save image
        image.save(filepath, format="PNG")
        
        return filename
    
    async def _save_input_audio(
        self,
        audio_ref: AudioRef,
        context: ProcessingContext
    ) -> str:
        """Save an input audio file and return the filename for ComfyUI."""
        import folder_paths
        
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"nodetool_input_{uuid.uuid4().hex[:8]}.wav"
        filepath = os.path.join(input_dir, filename)
        
        # Get audio data and save
        audio_data = await context.asset_to_bytes(audio_ref)
        with open(filepath, "wb") as f:
            f.write(audio_data)
        
        return filename
    
    async def _save_input_video(
        self,
        video_ref: VideoRef,
        context: ProcessingContext
    ) -> str:
        """Save an input video file and return the filename for ComfyUI."""
        import folder_paths
        
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        
        # Generate unique filename
        filename = f"nodetool_input_{uuid.uuid4().hex[:8]}.mp4"
        filepath = os.path.join(input_dir, filename)
        
        # Get video data and save
        video_data = await context.asset_to_bytes(video_ref)
        with open(filepath, "wb") as f:
            f.write(video_data)
        
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
    
    def _find_output_nodes(self, workflow: Dict[str, Any]) -> Dict[str, str]:
        """
        Find output nodes in the workflow.
        
        Looks for SaveImage, PreviewImage, and other output nodes.
        Returns a dict mapping node IDs to their class types.
        """
        output_types = {
            "SaveImage", "PreviewImage", "SaveAnimatedWEBP", 
            "SaveAnimatedPNG", "VHS_VideoCombine", "SaveAudio"
        }
        
        output_nodes = {}
        for node_id, node_info in workflow.items():
            if isinstance(node_info, dict):
                class_type = node_info.get("class_type", "")
                if class_type in output_types:
                    output_nodes[node_id] = class_type
        
        return output_nodes
    
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
        
        # Find output nodes from mappings
        output_node_ids = set()
        for output_name, mapping in self._output_mappings.items():
            output_node_ids.add(mapping["node"])
        
        # Also look for SaveImage nodes if no explicit outputs
        if not output_node_ids:
            auto_outputs = self._find_output_nodes(workflow)
            output_node_ids = set(auto_outputs.keys())
        
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
                        # Handle both tuple results (multiple outputs) and single values
                        if isinstance(from_result, (tuple, list)) and len(from_result) > from_output_idx:
                            node_inputs[input_name] = from_result[from_output_idx]
                        elif from_output_idx == 0:
                            # Single output - use directly if requesting index 0
                            node_inputs[input_name] = from_result
                        else:
                            raise RuntimeError(
                                f"Node {from_node_id} output index {from_output_idx} "
                                f"requested but result is not indexable or out of range"
                            )
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
            
            # Safely extract the output value at the specified index
            if isinstance(result, (tuple, list)):
                if output_idx < len(result):
                    output_value = result[output_idx]
                else:
                    raise RuntimeError(
                        f"Output index {output_idx} out of range for node {node_id} "
                        f"which has {len(result)} outputs"
                    )
            elif output_idx == 0:
                output_value = result
            else:
                raise RuntimeError(
                    f"Output index {output_idx} requested but node {node_id} "
                    f"has only a single output"
                )
            
            # Convert based on output type
            output_ref = await self._convert_output_value(output_value, context)
            outputs.append(output_ref)
        
        # Return single value or tuple based on number of outputs
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
    
    async def _convert_output_value(
        self,
        output_value: Any,
        context: ProcessingContext
    ) -> Any:
        """Convert a ComfyUI output value to a nodetool type."""
        import folder_paths
        
        # Check if it's a dict from SaveImage/PreviewImage
        if isinstance(output_value, dict) and "images" in output_value:
            images = output_value["images"]
            if images:
                # Load the first image
                output_dir = Path(folder_paths.get_output_directory())
                image_info = images[0]
                subfolder = image_info.get("subfolder", "")
                filename = image_info["filename"]
                
                # Use pathlib for consistent path handling
                if subfolder:
                    image_path = output_dir / subfolder / filename
                else:
                    image_path = output_dir / filename
                
                pil_image = PIL.Image.open(image_path)
                return await context.image_from_pil(pil_image)
        
        # Check if it's a tensor (direct image output)
        if isinstance(output_value, torch.Tensor):
            return await context.image_from_tensor(output_value)
        
        # Check if it's a numpy array
        if isinstance(output_value, np.ndarray):
            tensor = torch.from_numpy(output_value)
            return await context.image_from_tensor(tensor)
        
        # Return as-is for other types
        return output_value
    
    async def _execute_workflow(self, context: ProcessingContext) -> Any:
        """
        Main execution method for workflow nodes.
        
        This orchestrates the full workflow:
        1. Prepare inputs (convert nodetool types)
        2. Modify workflow JSON with inputs
        3. Execute via ComfyUI
        4. Process and return outputs
        """
        with torch.inference_mode():
            # Prepare inputs
            prepared_inputs = await self._prepare_inputs(context)
            
            # Modify workflow with inputs
            modified_workflow = self._modify_workflow(prepared_inputs)
            
            # Execute workflow
            results = await self._execute_comfy_workflow(modified_workflow, context)
            
            # Process and return outputs
            return await self._process_outputs(results, context)


__all__ = ["ComfyWorkflowNode"]
