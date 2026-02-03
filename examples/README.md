# ComfyUI Node DSL Examples

This directory contains example workflows demonstrating how to use ComfyUI nodes through the nodetool DSL interface.

## Overview

These examples show common workflows using ComfyUI nodes in a Pythonic, declarative style. The DSL (Domain-Specific Language) makes it easy to:

- Build workflows programmatically
- Connect nodes through property access
- Serialize and execute graphs
- Create reusable workflow templates

## Examples

### 1. text_to_image_basic.py

A simple text-to-image workflow demonstrating the core SD pipeline:

```python
CheckpointLoader -> CLIPTextEncode -> EmptyLatent -> KSampler -> VAEDecode
```

**Key Concepts:**
- Loading checkpoints (model, CLIP, VAE)
- Text prompt encoding
- Latent space operations
- Sampling and decoding

### 2. image_upscale.py

An image upscaling workflow with optional refinement:

```python
LoadImage -> VAEEncode -> LatentUpscale -> KSampler -> VAEDecode
```

**Key Concepts:**
- Image loading and encoding
- Latent space upscaling (faster than pixel space)
- img2img refinement with low denoise
- Preserving original image features

## Prerequisites

### 1. Install nodetool-comfy

```bash
pip install nodetool-comfy
```

### 2. Run Code Generation

The examples use DSL wrappers that need to be generated:

```bash
# This requires nodetool-core to be installed
nodetool codegen --package nodetool-comfy
```

This will generate DSL wrapper classes in `nodetool.dsl.comfy.*`

### 3. Download Test Model

For testing, use the small sd-tiny model (~150MB):

```bash
# Using huggingface-cli (recommended)
pip install -U huggingface_hub
huggingface-cli download segmind/sd-tiny \
  sd_tiny.safetensors \
  --local-dir models/checkpoints \
  --local-dir-use-symlinks False

# Or using curl
curl -L -o models/checkpoints/sd_tiny.safetensors \
  https://huggingface.co/segmind/sd-tiny/resolve/main/sd_tiny.safetensors
```

**Note:** sd-tiny produces low-quality output (this is expected). It's only for testing node functionality.

## Running Examples

After code generation and model download:

```bash
# Basic text-to-image
python examples/text_to_image_basic.py

# Image upscaling
python examples/image_upscale.py
```

## DSL Pattern

The DSL provides a clean, Pythonic way to build workflows:

```python
from nodetool.dsl.graph import create_graph
from nodetool.dsl.comfy.loaders import CheckpointLoaderSimple
from nodetool.dsl.comfy.conditioning import CLIPTextEncode
from nodetool.dsl.comfy.sampling import KSampler
from nodetool.dsl.nodetool.output import Output

# Instantiate nodes
checkpoint = CheckpointLoaderSimple(ckpt_name="model.safetensors")
encoder = CLIPTextEncode(text="a beautiful landscape", clip=checkpoint.clip)
sampler = KSampler(model=checkpoint.model, positive=encoder.output, ...)
output = Output(name="result", value=sampler.output)

# Create graph
graph = create_graph(output)

# Execute (async)
result = await run_graph(graph, user_id="...", auth_token="...")
```

**Key Features:**
- Node outputs are accessible via properties (`.clip`, `.output`, etc.)
- Connections are established by passing outputs to inputs
- The graph is constructed declaratively
- Supports all ComfyUI node types

## CI/Testing

For CI environments, these examples work well with:

- **Model:** segmind/sd-tiny (~150MB)
- **Device:** CPU (slow but works)
- **Purpose:** Smoke tests for node functionality
- **Quality:** Intentionally poor (not for production)

Example test:

```bash
# Download model
huggingface-cli download segmind/sd-tiny sd_tiny.safetensors \
  --local-dir models/checkpoints

# Run integration test
python tests/test_integration_sd_tiny.py
```

## Node Categories

The generated ComfyUI nodes cover these categories:

- **loaders**: CheckpointLoader, VAE, CLIP, LoRA
- **conditioning**: Text encoding, CLIP operations
- **latent**: Latent operations, upscaling, inpainting
- **sampling**: KSampler, schedulers, sigmas
- **image**: Image operations, transformations
- **mask**: Mask operations, compositing
- **And 75+ more categories**

## Custom Workflows

You can create your own workflows following these patterns:

1. **Import DSL components**
   ```python
   from nodetool.dsl.graph import create_graph
   from nodetool.dsl.comfy.* import ...
   ```

2. **Instantiate nodes with parameters**
   ```python
   node = NodeClass(param1=value1, param2=node2.output)
   ```

3. **Connect via outputs**
   ```python
   next_node = NextNode(input=previous_node.output)
   ```

4. **Create graph with outputs**
   ```python
   graph = create_graph(output1, output2, ...)
   ```

5. **Execute**
   ```python
   result = await run_graph(graph, ...)
   ```

## Troubleshooting

### "No module named 'nodetool.dsl.comfy'"

You need to run code generation:
```bash
nodetool codegen --package nodetool-comfy
```

### "Model not found"

Download sd-tiny or point to your model:
```bash
huggingface-cli download segmind/sd-tiny ...
```

### "CUDA out of memory"

Use smaller dimensions or CPU:
```python
EmptyLatentImage(width=256, height=256, ...)  # Smaller
```

### Slow execution

This is normal on CPU. Use:
- Fewer sampling steps
- Smaller image dimensions
- GPU if available

## Contributing

To add new examples:

1. Create a new .py file in this directory
2. Follow the existing pattern
3. Document the workflow structure
4. Add usage notes
5. Submit a PR

## License

AGPL - See LICENSE file
