# ComfyUI Integration Tests

This directory contains integration tests for the generated ComfyUI nodes.

## Test Files

### test_comfy_nodes_basic.py
Basic validation tests that check:
- Generated nodes can be imported
- Wrapper types are available
- Nodes have correct structure (fields, methods)
- Wrapper types function correctly

### test_node_instantiation.py
Tests node instantiation:
- Nodes can be created with default values
- Field types are correct (Model, Clip, Vae, etc.)
- Nodes accept specific values
- Numeric constraints work

### test_comfy_workflows.py
Tests workflow construction:
- Basic text-to-image workflow structure
- Image-to-image workflow structure
- Latent upscale workflow
- Node connections via wrapper types

### test_integration_sd_tiny.py
Full end-to-end integration test:
- Downloads segmind/sd-tiny model (~150MB)
- Runs complete text-to-image pipeline
- Tests all major nodes with real execution
- CPU-compatible (slow but functional)

## Running Tests Locally

### Prerequisites

Install dependencies:
```bash
pip install pydantic torch torchvision transformers safetensors
```

Note: `nodetool-core` is required but may not be publicly available. The tests are designed to work in the CI environment where it's installed.

### Run All Tests

```bash
# Basic validation (no model required)
python tests/test_comfy_nodes_basic.py

# Node instantiation (no model required)
python tests/test_node_instantiation.py

# Workflow construction (no model required)
python tests/test_comfy_workflows.py

# Full integration test (downloads model, runs inference)
python tests/test_integration_sd_tiny.py
```

## GitHub Actions CI

Tests run automatically on push and PR via `.github/workflows/test.yml`:

1. **Environment Setup**
   - Python 3.10 and 3.11
   - CPU-only PyTorch
   - Core dependencies

2. **Model Caching**
   - sd-tiny model cached between runs
   - ~150MB download on first run

3. **Test Execution**
   - Validation tests (fast)
   - Node instantiation tests (fast)
   - Workflow tests (fast)
   - Integration test with sd-tiny (slow on CPU)

4. **Metadata Generation**
   - Validates package metadata can be generated

## Test Philosophy

- **Fast tests** run without model loading
- **Integration tests** use minimal model (sd-tiny)
- **CPU-compatible** for CI environments
- **Graceful degradation** when dependencies missing

## Using segmind/sd-tiny

The integration test uses `segmind/sd-tiny` (~150MB):
- Works with SD 1.x pipelines
- CPU inference supported (slow)
- Output quality is intentionally poor
- Perfect for smoke testing

Download manually:
```bash
pip install huggingface_hub
huggingface-cli download segmind/sd-tiny \
  sd_tiny.safetensors \
  --local-dir models/checkpoints \
  --local-dir-use-symlinks False
```

Or use curl:
```bash
mkdir -p models/checkpoints
curl -L -o models/checkpoints/sd_tiny.safetensors \
  https://huggingface.co/segmind/sd-tiny/resolve/main/sd_tiny.safetensors
```

## Adding New Tests

When adding new tests:

1. Follow the existing pattern
2. Make tests work without nodetool-core when possible
3. Use mock objects for structure tests
4. Keep integration tests minimal (use sd-tiny)
5. Update this README

## Troubleshooting

### "No module named 'nodetool.workflows'"

The tests require nodetool-core which may not be publicly available. In CI, this is installed automatically. For local testing, you may need access to nodetool-core.

### "Model not found"

Download sd-tiny using the commands above.

### Tests timeout on CPU

This is expected. Integration tests can take 1-2 minutes on CPU. GitHub Actions has a 5-minute timeout configured.

### Import errors

Ensure all dependencies are installed:
```bash
pip install pydantic torch transformers safetensors
```
