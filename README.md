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
