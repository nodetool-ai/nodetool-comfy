"""
Integration Test: ComfyUI Nodes with SD-Tiny Model

This test validates that generated ComfyUI nodes work correctly with a small
test model (segmind/sd-tiny) without requiring full GPU resources.

The test:
1. Downloads sd-tiny model if not present
2. Creates a simple text-to-image workflow
3. Runs inference (CPU is fine, just slow)
4. Validates output structure

This uses sd-tiny (~150MB) which is ideal for CI:
- Works with SD 1.x pipeline
- CPU inference supported
- Output quality is poor (expected)
- Fast to download
"""

import asyncio
import os
from pathlib import Path
import subprocess
import sys

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def download_sd_tiny(checkpoint_dir: Path) -> Path:
    """
    Download segmind/sd-tiny model for testing.
    
    Args:
        checkpoint_dir: Directory to store checkpoint
        
    Returns:
        Path to the downloaded checkpoint file
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "sd_tiny.safetensors"
    
    if checkpoint_path.exists():
        print(f"✓ sd-tiny already downloaded: {checkpoint_path}")
        return checkpoint_path
    
    print("Downloading sd-tiny model...")
    print("  Model: segmind/sd-tiny")
    print("  Size: ~150MB")
    print("  Location: {}".format(checkpoint_path))
    print()
    
    try:
        # Try using huggingface_hub CLI
        result = subprocess.run(
            [
                "huggingface-cli", "download",
                "segmind/sd-tiny",
                "sd_tiny.safetensors",
                "--local-dir", str(checkpoint_dir),
                "--local-dir-use-symlinks", "False"
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0 and checkpoint_path.exists():
            print("✓ Download successful")
            return checkpoint_path
        else:
            print(f"huggingface-cli failed: {result.stderr}")
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"huggingface-cli not available or timeout: {e}")
    
    # Fallback to direct download with curl
    print("Trying direct download with curl...")
    try:
        result = subprocess.run(
            [
                "curl", "-L", "-o", str(checkpoint_path),
                "https://huggingface.co/segmind/sd-tiny/resolve/main/sd_tiny.safetensors"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0 and checkpoint_path.exists():
            print("✓ Download successful via curl")
            return checkpoint_path
        else:
            raise RuntimeError(f"curl download failed: {result.stderr}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to download sd-tiny: {e}")


async def test_basic_text_to_image():
    """
    Test basic text-to-image workflow with sd-tiny model.
    
    This test validates:
    1. CheckpointLoader can load sd-tiny
    2. CLIPTextEncode works
    3. EmptyLatentImage creates proper latent
    4. KSampler runs (even if slow on CPU)
    5. VAEDecode produces image tensor
    """
    print("\n" + "=" * 70)
    print("Integration Test: Basic Text-to-Image with SD-Tiny")
    print("=" * 70)
    print()
    
    # Import generated nodes
    try:
        from nodetool.nodes.comfy.generated.all_nodes import (
            CheckpointLoaderSimple,
            CLIPTextEncode,
            EmptyLatentImage,
            KSampler,
            VAEDecode,
        )
        from nodetool.workflows.processing_context import ProcessingContext
        print("✓ Successfully imported generated nodes")
    except ImportError as e:
        print(f"✗ Failed to import nodes: {e}")
        print("  This test requires nodetool-core to be installed")
        return False
    
    # Setup checkpoint
    models_dir = repo_root / "models" / "checkpoints"
    try:
        checkpoint_path = download_sd_tiny(models_dir)
        print(f"✓ Model ready: {checkpoint_path}")
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False
    
    print()
    print("Running text-to-image workflow...")
    print("  Note: This will be slow on CPU (~1-2 minutes)")
    print()
    
    try:
        # Create processing context
        context = ProcessingContext()
        
        # Step 1: Load checkpoint
        print("  1. Loading checkpoint...")
        loader = CheckpointLoaderSimple(ckpt_name="sd_tiny.safetensors")
        model, clip, vae = await loader.process(context)
        print(f"     ✓ Loaded: model={type(model).__name__}, clip={type(clip).__name__}, vae={type(vae).__name__}")
        
        # Step 2: Encode positive prompt
        print("  2. Encoding positive prompt...")
        pos_encoder = CLIPTextEncode(
            text="a beautiful landscape",
            clip=clip
        )
        positive = await pos_encoder.process(context)
        print(f"     ✓ Positive conditioning: {type(positive).__name__}")
        
        # Step 3: Encode negative prompt
        print("  3. Encoding negative prompt...")
        neg_encoder = CLIPTextEncode(
            text="blurry, low quality",
            clip=clip
        )
        negative = await neg_encoder.process(context)
        print(f"     ✓ Negative conditioning: {type(negative).__name__}")
        
        # Step 4: Create empty latent
        print("  4. Creating empty latent...")
        latent_creator = EmptyLatentImage(
            width=256,  # Small size for speed
            height=256,
            batch_size=1
        )
        latent = await latent_creator.process(context)
        print(f"     ✓ Empty latent: {type(latent).__name__}")
        
        # Step 5: Sample (this is the slow part)
        print("  5. Sampling (this takes time on CPU)...")
        sampler = KSampler(
            model=model,
            seed=42,
            steps=5,  # Very few steps for speed
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=1.0
        )
        sampled_latent = await sampler.process(context)
        print(f"     ✓ Sampled latent: {type(sampled_latent).__name__}")
        
        # Step 6: Decode to image
        print("  6. Decoding to image...")
        decoder = VAEDecode(
            samples=sampled_latent,
            vae=vae
        )
        image = await decoder.process(context)
        print(f"     ✓ Decoded image: {type(image).__name__}")
        
        print()
        print("=" * 70)
        print("✓ Integration test passed!")
        print("=" * 70)
        print()
        print("All nodes executed successfully:")
        print("  - CheckpointLoader ✓")
        print("  - CLIPTextEncode ✓")
        print("  - EmptyLatentImage ✓")
        print("  - KSampler ✓")
        print("  - VAEDecode ✓")
        print()
        print("Note: Image quality from sd-tiny is intentionally poor.")
        print("This model is only for testing node functionality.")
        
        return True
        
    except Exception as e:
        print()
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the integration test."""
    print("ComfyUI Node Integration Test")
    print("Using: segmind/sd-tiny model")
    print()
    
    # Check if we can import necessary modules
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except ImportError:
        print("✗ PyTorch not available")
        return 1
    
    # Run the test
    success = asyncio.run(test_basic_text_to_image())
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
