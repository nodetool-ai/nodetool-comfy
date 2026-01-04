# Nodetool-Comfy

ComfyUI nodes for Nodetool.

## Description

This package provides ComfyUI nodes for integration with Nodetool, allowing for seamless workflow between the two platforms.

The repository includes:
- ComfyUI as a git submodule (from https://github.com/comfyanonymous/ComfyUI)
- Automated tools to parse and sync ComfyUI nodes
- Nodetool-specific node wrappers and integrations

## Installation

```bash
pip install nodetool-comfy
```

Or install from source:

```bash
git clone --recursive https://github.com/nodetool-ai/nodetool-comfy.git
cd nodetool-comfy
pip install -e .
```

If you cloned without `--recursive`, initialize the submodule:

```bash
git submodule update --init
```

## Development

### Syncing ComfyUI Source

To sync the ComfyUI source code from the submodule to `src/`:

```bash
python scripts/sync_comfyui.py
```

### Parsing ComfyUI Nodes

To parse all ComfyUI nodes and generate metadata:

```bash
python scripts/parse_comfy_nodes.py
```

This generates `comfy_nodes_metadata.json` with information about all available ComfyUI nodes (both V1/classic style and V3/schema style).

### Importing ComfyUI Workflows as Nodetool Nodes

You can import ComfyUI workflows and convert them to Nodetool nodes:

```bash
python scripts/import_workflows.py [workflow_dir] [-o output_dir]
```

**How it works:**
1. Each ComfyUI workflow is defined using a YAML configuration file
2. The YAML file references a corresponding ComfyUI workflow API JSON file
3. The script generates Python modules with node classes for each workflow
4. Generated nodes inherit from `ComfyWorkflowNode` base class
5. The nodes handle conversion between Nodetool types (ImageRef, etc.) and ComfyUI types

**Workflow YAML Format:**

```yaml
name: "My Workflow"
description: "Generate images with SDXL"
category: "comfy.workflows"

inputs:
  prompt:
    node: "6"           # Node ID in workflow JSON
    field: "text"       # Field name in the node's inputs
    type: str
    default: ""
    description: "The prompt for generation"
  
  image:
    node: "10"
    field: "image"
    type: ImageRef      # Nodetool type
    description: "Input image"

outputs:
  image:
    node: "9"           # SaveImage node ID
    type: ImageRef
    description: "Generated image"
```

**Supported Types:**
- `str`, `int`, `float`, `bool` - Basic Python types
- `ImageRef` - Image assets (converted to/from ComfyUI image format)
- `AudioRef` - Audio assets
- `VideoRef` - Video assets

**Example:**

Place workflow files in the `workflows/` directory:
- `workflows/my_workflow.yaml` - Configuration
- `workflows/my_workflow.json` - ComfyUI workflow API JSON

Then run:
```bash
python scripts/import_workflows.py workflows -o src/nodetool/nodes/comfy/
```

### Updating ComfyUI

To update the ComfyUI submodule to the latest version:

```bash
cd ComfyUI
git pull origin master
cd ..
git add ComfyUI
git commit -m "Update ComfyUI submodule"
```

Then sync the source:

```bash
python scripts/sync_comfyui.py
python scripts/parse_comfy_nodes.py
```

## Requirements

- Python 3.10+
- See pyproject.toml for full dependencies

## License

AGPL

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
