# ComfyUI Proxy Node Generator

This directory contains scripts for parsing ComfyUI nodes and generating Nodetool proxy nodes.

## Scripts

### parse_comfy_nodes.py

Parses ComfyUI node definitions from the ComfyUI submodule and generates `comfy_nodes_metadata.json`.

**Usage:**
```bash
python scripts/parse_comfy_nodes.py
```

This will:
- Parse all nodes from `ComfyUI/nodes.py` and `ComfyUI/comfy_extras/*.py`
- Extract node metadata (inputs, outputs, categories, descriptions)
- Support both V1 (classic) and V3 (schema) node styles
- Generate `comfy_nodes_metadata.json` with 400+ nodes

### generate_proxy_nodes.py

Generates Nodetool proxy nodes from ComfyUI metadata. Each proxy node wraps a ComfyUI node with proper type conversions.

**Usage:**
```bash
# Generate all nodes
python scripts/generate_proxy_nodes.py

# Generate specific categories
python scripts/generate_proxy_nodes.py --categories sampling loaders image

# Specify custom output directory
python scripts/generate_proxy_nodes.py --output-dir path/to/output
```

**Options:**
- `--metadata PATH`: Path to comfy_nodes_metadata.json (default: comfy_nodes_metadata.json)
- `--output-dir PATH`: Output directory (default: src/nodetool/nodes/comfy_generated)
- `--categories CATS...`: Specific categories to generate (default: all)

**Generated Code:**

The generator creates Python modules organized by category:

```
src/nodetool/nodes/comfy_generated/
├── __init__.py
├── sampling.py
├── loaders.py
├── image/
│   ├── __init__.py
│   ├── upscaling.py
│   ├── transform.py
│   └── ...
└── ...
```

Each generated node:
- Extends `BaseNode` from nodetool
- Has proper Pydantic field definitions with constraints
- Handles type conversions (e.g., ImageRef ↔ torch.Tensor)
- Calls the underlying ComfyUI node
- Returns converted results

**Example Generated Node:**

```python
class ComfyImageScale(BaseNode):
    """
    Scales images to specified dimensions.
    
    ComfyUI Node: ImageScale
    Category: image/upscaling
    """
    
    image: ImageRef = Field(..., description="Input image")
    upscale_method: str = Field(..., description="Upscaling algorithm")
    width: int = Field(default=512, ge=0)
    height: int = Field(default=512, ge=0)
    crop: str = Field(...)
    
    async def process(self, context: ProcessingContext) -> ImageRef:
        """Execute the ImageScale ComfyUI node."""
        from nodes import ImageScale
        
        # Convert ImageRef to torch.Tensor
        inputs = {
            "image": await context.tensor_from_image(self.image),
            "upscale_method": self.upscale_method,
            "width": self.width,
            "height": self.height,
            "crop": self.crop
        }
        
        # Execute ComfyUI node
        node = ImageScale()
        result = node.upscale(**inputs)
        
        # Convert result back to ImageRef
        return await context.image_from_tensor(result[0])
```

### type_mappings.py

Type mapping utilities for converting between ComfyUI and Nodetool types.

**Key Functions:**
- `get_nodetool_type(comfy_type)`: Convert ComfyUI type to Nodetool type
- `get_field_constraints(config)`: Extract Pydantic constraints from ComfyUI config
- `get_type_conversion_code(param, type, direction)`: Generate type conversion code
- `needs_type_conversion(comfy_type)`: Check if type needs conversion

**Type Mappings:**
- `IMAGE` → `ImageRef` (needs conversion)
- `MASK` → `ImageRef` (needs conversion)
- `AUDIO` → `AudioRef` (needs conversion)
- `LATENT` → `dict` (no conversion)
- `MODEL`, `CLIP`, `VAE` → `Any` (internal ComfyUI objects, no conversion)
- `INT`, `FLOAT`, `STRING` → `int`, `float`, `str` (no conversion)

## Workflow

1. **Sync ComfyUI source:**
   ```bash
   python scripts/sync_comfyui.py
   ```

2. **Parse ComfyUI nodes:**
   ```bash
   python scripts/parse_comfy_nodes.py
   ```
   
   This generates `comfy_nodes_metadata.json`

3. **Generate proxy nodes:**
   ```bash
   python scripts/generate_proxy_nodes.py
   ```
   
   This generates proxy nodes in `src/nodetool/nodes/comfy_generated/`

## Development

### Adding New Type Mappings

Edit `type_mappings.py` to add new type mappings:

```python
COMFY_TO_NODETOOL_TYPE_MAP = {
    "NEW_TYPE": "NodetoolType",
    ...
}

TYPE_IMPORTS["NodetoolType"] = "from nodetool.types import NodetoolType"
```

### Customizing Generation

Edit `generate_proxy_nodes.py` to customize:
- Class naming: `_generate_class_name()`
- Field generation: `_generate_field_definition()`
- Process method: `_generate_process_method()`
- File organization: `_get_category_path()`

## Notes

- The generator skips deprecated nodes
- Testing nodes (_for_testing/*) are included but can be filtered
- Some ComfyUI nodes may have incomplete metadata
- Generated code is auto-formatted with proper docstrings
- Type conversions are automatically added for IMAGE, MASK, and AUDIO types
