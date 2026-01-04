"""Tests for the ComfyUI workflow importer."""

import json
import tempfile
from pathlib import Path

import pytest

# Import from the scripts directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from import_workflows import (
    WorkflowConfig,
    WorkflowInput,
    WorkflowOutput,
    snake_to_pascal,
    pascal_to_snake,
    parse_input_config,
    parse_output_config,
    load_workflow_config,
    generate_field_definition,
    generate_output_type,
    generate_node_class,
    generate_module_code,
)


class TestCaseConversion:
    """Tests for case conversion utilities."""
    
    def test_snake_to_pascal(self):
        assert snake_to_pascal("hello_world") == "HelloWorld"
        assert snake_to_pascal("my_workflow") == "MyWorkflow"
        assert snake_to_pascal("sdxl_text_to_image") == "SdxlTextToImage"
        assert snake_to_pascal("simple") == "Simple"
    
    def test_snake_to_pascal_with_hyphens(self):
        assert snake_to_pascal("hello-world") == "HelloWorld"
        assert snake_to_pascal("my-workflow") == "MyWorkflow"
    
    def test_pascal_to_snake(self):
        assert pascal_to_snake("HelloWorld") == "hello_world"
        assert pascal_to_snake("MyWorkflow") == "my_workflow"
        assert pascal_to_snake("SDXLTextToImage") == "sdxl_text_to_image"


class TestInputParsing:
    """Tests for input configuration parsing."""
    
    def test_parse_string_input(self):
        config = {
            "node": "6",
            "field": "text",
            "type": "str",
            "default": "hello",
            "description": "A text field"
        }
        inp = parse_input_config("prompt", config)
        
        assert inp.name == "prompt"
        assert inp.node_id == "6"
        assert inp.field_name == "text"
        assert inp.python_type == "str"
        assert inp.default == "hello"
        assert inp.description == "A text field"
    
    def test_parse_int_input_with_constraints(self):
        config = {
            "node": "3",
            "field": "steps",
            "type": "int",
            "default": 20,
            "min": 1,
            "max": 100,
            "description": "Steps"
        }
        inp = parse_input_config("steps", config)
        
        assert inp.python_type == "int"
        assert inp.default == 20
        assert inp.min_value == 1
        assert inp.max_value == 100
    
    def test_parse_image_input(self):
        config = {
            "node": "10",
            "field": "image",
            "type": "ImageRef",
            "description": "Input image"
        }
        inp = parse_input_config("input_image", config)
        
        assert inp.python_type == "ImageRef"
        assert inp.default is None
    
    def test_parse_type_aliases(self):
        # Test that common aliases are converted
        assert parse_input_config("x", {"type": "string"}).python_type == "str"
        assert parse_input_config("x", {"type": "integer"}).python_type == "int"
        assert parse_input_config("x", {"type": "number"}).python_type == "float"
        assert parse_input_config("x", {"type": "boolean"}).python_type == "bool"
        assert parse_input_config("x", {"type": "image"}).python_type == "ImageRef"


class TestOutputParsing:
    """Tests for output configuration parsing."""
    
    def test_parse_image_output(self):
        config = {
            "node": "9",
            "type": "ImageRef",
            "output_index": 0,
            "description": "Generated image"
        }
        out = parse_output_config("image", config)
        
        assert out.name == "image"
        assert out.node_id == "9"
        assert out.python_type == "ImageRef"
        assert out.output_index == 0


class TestFieldGeneration:
    """Tests for Pydantic field generation."""
    
    def test_generate_string_field(self):
        inp = WorkflowInput(
            name="prompt",
            node_id="6",
            field_name="text",
            python_type="str",
            default="hello",
            description="The prompt"
        )
        field_def = generate_field_definition(inp)
        
        assert 'default="hello"' in field_def
        assert 'description="The prompt"' in field_def
    
    def test_generate_int_field_with_constraints(self):
        inp = WorkflowInput(
            name="steps",
            node_id="3",
            field_name="steps",
            python_type="int",
            default=20,
            min_value=1,
            max_value=100
        )
        field_def = generate_field_definition(inp)
        
        assert "default=20" in field_def
        assert "ge=1" in field_def
        assert "le=100" in field_def
    
    def test_generate_image_field(self):
        inp = WorkflowInput(
            name="image",
            node_id="10",
            field_name="image",
            python_type="ImageRef"
        )
        field_def = generate_field_definition(inp)
        
        assert "ImageRef()" in field_def


class TestOutputTypeGeneration:
    """Tests for output type annotation generation."""
    
    def test_single_output(self):
        outputs = [WorkflowOutput(name="image", node_id="9", python_type="ImageRef")]
        assert generate_output_type(outputs) == "ImageRef"
    
    def test_no_outputs(self):
        assert generate_output_type([]) == "None"
    
    def test_multiple_outputs(self):
        outputs = [
            WorkflowOutput(name="image", node_id="9", python_type="ImageRef"),
            WorkflowOutput(name="mask", node_id="10", python_type="ImageRef"),
        ]
        assert generate_output_type(outputs) == "tuple[ImageRef, ImageRef]"


class TestClassGeneration:
    """Tests for node class generation."""
    
    def test_generate_simple_class(self):
        config = WorkflowConfig(
            name="Test Workflow",
            description="A test workflow",
            category="comfy.workflows.test",
            workflow_json={"1": {"class_type": "TestNode"}},
            inputs=[
                WorkflowInput(
                    name="prompt",
                    node_id="1",
                    field_name="text",
                    python_type="str",
                    default="hello",
                    description="The prompt"
                )
            ],
            outputs=[
                WorkflowOutput(
                    name="image",
                    node_id="1",
                    python_type="ImageRef"
                )
            ],
            class_name="TestWorkflow"
        )
        
        code = generate_node_class(config)
        
        # Check class definition
        assert "class TestWorkflow(ComfyWorkflowNode):" in code
        
        # Check docstring
        assert "A test workflow" in code
        
        # Check input field
        assert 'prompt: str = Field(default="hello"' in code
        
        # Check output type
        assert "-> ImageRef:" in code
        
        # Check get_title
        assert 'return "Test Workflow"' in code


class TestModuleGeneration:
    """Tests for module code generation."""
    
    def test_generate_module_with_imports(self):
        configs = [
            WorkflowConfig(
                name="Test",
                description="Test",
                category="comfy.test",
                workflow_json={},
                class_name="TestNode"
            )
        ]
        
        code = generate_module_code(configs, "test")
        
        # Check imports
        assert "from __future__ import annotations" in code
        assert "from pydantic import Field" in code
        assert "from nodetool.metadata.types import ImageRef" in code
        assert "from nodetool.nodes.comfy.workflow_base import ComfyWorkflowNode" in code
        
        # Check __all__
        assert "__all__ = ['TestNode']" in code


class TestWorkflowLoading:
    """Tests for loading workflow configurations from files."""
    
    def test_load_workflow_from_yaml_and_json(self, tmp_path):
        # Create test YAML
        yaml_content = """
name: "My Workflow"
description: "Test workflow"
category: "comfy.test"

inputs:
  prompt:
    node: "1"
    field: "text"
    type: str
    default: "hello"

outputs:
  image:
    node: "2"
    type: ImageRef
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)
        
        # Create test JSON
        json_content = {
            "1": {"class_type": "CLIPTextEncode"},
            "2": {"class_type": "SaveImage"}
        }
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(json_content))
        
        # Load config
        config = load_workflow_config(yaml_path)
        
        assert config is not None
        assert config.name == "My Workflow"
        assert config.description == "Test workflow"
        assert len(config.inputs) == 1
        assert config.inputs[0].name == "prompt"
        assert len(config.outputs) == 1
        assert config.outputs[0].name == "image"
    
    def test_load_workflow_missing_json(self, tmp_path):
        yaml_content = "name: test"
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)
        
        # Should return None when JSON is missing
        config = load_workflow_config(yaml_path)
        assert config is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
