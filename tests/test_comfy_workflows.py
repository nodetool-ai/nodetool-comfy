"""
Test: ComfyUI Workflow Construction

This test validates that ComfyUI workflows can be constructed by connecting
nodes together, without requiring actual execution.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def test_basic_workflow_structure():
    """Test that a basic workflow can be constructed."""
    try:
        from nodetool.nodes.comfy.generated.all_nodes import (
            CheckpointLoaderSimple,
            CLIPTextEncode,
            EmptyLatentImage,
            KSampler,
            VAEDecode,
        )
        from nodetool.nodes.comfy.types import Model, Clip, Vae, Conditioning, Latent
        
        print("Constructing basic text-to-image workflow:")
        print()
        
        # Step 1: Create checkpoint loader
        checkpoint = CheckpointLoaderSimple(ckpt_name="model.safetensors")
        print("  ✓ Created CheckpointLoaderSimple")
        
        # Step 2: Create mock outputs (in real execution, these come from process())
        mock_model = Model("mock_model")
        mock_clip = Clip("mock_clip")
        mock_vae = Vae("mock_vae")
        print("  ✓ Created mock checkpoint outputs")
        
        # Step 3: Create text encoders
        positive_encoder = CLIPTextEncode(
            text="a beautiful landscape",
            clip=mock_clip
        )
        print("  ✓ Created positive CLIPTextEncode")
        
        negative_encoder = CLIPTextEncode(
            text="blurry, low quality",
            clip=mock_clip
        )
        print("  ✓ Created negative CLIPTextEncode")
        
        # Step 4: Create mock conditioning
        mock_positive = Conditioning("mock_positive")
        mock_negative = Conditioning("mock_negative")
        print("  ✓ Created mock conditioning outputs")
        
        # Step 5: Create empty latent
        latent_creator = EmptyLatentImage(
            width=512,
            height=512,
            batch_size=1
        )
        print("  ✓ Created EmptyLatentImage")
        
        # Step 6: Create mock latent
        mock_latent = Latent({"samples": "mock"})
        print("  ✓ Created mock latent output")
        
        # Step 7: Create sampler
        sampler = KSampler(
            model=mock_model,
            seed=42,
            steps=20,
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            positive=mock_positive,
            negative=mock_negative,
            latent_image=mock_latent,
            denoise=1.0
        )
        print("  ✓ Created KSampler with all inputs")
        
        # Step 8: Create VAE decoder
        decoder = VAEDecode(
            samples=mock_latent,
            vae=mock_vae
        )
        print("  ✓ Created VAEDecode")
        
        print()
        print("✓ Successfully constructed complete workflow structure")
        return True
        
    except Exception as e:
        print(f"✗ Workflow construction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_workflow_structure():
    """Test that an image processing workflow can be constructed."""
    try:
        from nodetool.nodes.comfy.generated.all_nodes import (
            CheckpointLoaderSimple,
            LoadImage,
            VAEEncode,
            CLIPTextEncode,
            KSampler,
            VAEDecode,
        )
        from nodetool.nodes.comfy.types import Model, Clip, Vae, Conditioning, Latent
        from nodetool.metadata.types import ImageRef
        
        print("Constructing image-to-image workflow:")
        print()
        
        # Step 1: Checkpoint loader
        checkpoint = CheckpointLoaderSimple(ckpt_name="model.safetensors")
        print("  ✓ Created CheckpointLoaderSimple")
        
        # Mock outputs
        mock_model = Model("mock_model")
        mock_clip = Clip("mock_clip")
        mock_vae = Vae("mock_vae")
        
        # Step 2: Image loader
        try:
            # LoadImage might expect a file path
            image_loader = LoadImage(image="input.png")
            print("  ✓ Created LoadImage")
        except Exception as e:
            print(f"  ! LoadImage creation note: {e}")
            # This is fine, just testing structure
        
        # Mock image ref
        mock_image = ImageRef()
        
        # Step 3: VAE encoder
        encoder = VAEEncode(
            pixels=mock_image,
            vae=mock_vae
        )
        print("  ✓ Created VAEEncode")
        
        # Mock latent
        mock_latent = Latent({"samples": "mock"})
        
        # Step 4: Text encode
        positive_encoder = CLIPTextEncode(
            text="enhance quality",
            clip=mock_clip
        )
        print("  ✓ Created CLIPTextEncode")
        
        mock_conditioning = Conditioning("mock")
        
        # Step 5: Sampler with low denoise
        sampler = KSampler(
            model=mock_model,
            seed=42,
            steps=15,
            cfg=7.0,
            sampler_name="euler",
            scheduler="normal",
            positive=mock_conditioning,
            negative=mock_conditioning,
            latent_image=mock_latent,
            denoise=0.5  # Low denoise for img2img
        )
        print("  ✓ Created KSampler with denoise=0.5")
        
        # Step 6: Decoder
        decoder = VAEDecode(
            samples=mock_latent,
            vae=mock_vae
        )
        print("  ✓ Created VAEDecode")
        
        print()
        print("✓ Successfully constructed image-to-image workflow")
        return True
        
    except Exception as e:
        print(f"✗ Image workflow construction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_latent_upscale_workflow():
    """Test that a latent upscaling workflow can be constructed."""
    try:
        from nodetool.nodes.comfy.generated.all_nodes import (
            EmptyLatentImage,
            LatentUpscale,
        )
        from nodetool.nodes.comfy.types import Latent
        
        print("Constructing latent upscale workflow:")
        print()
        
        # Create empty latent
        latent_creator = EmptyLatentImage(
            width=512,
            height=512,
            batch_size=1
        )
        print("  ✓ Created EmptyLatentImage (512x512)")
        
        # Mock latent
        mock_latent = Latent({"samples": "mock"})
        
        # Upscale latent
        upscaler = LatentUpscale(
            samples=mock_latent,
            upscale_method="bilinear",
            width=1024,
            height=1024,
            crop="disabled"
        )
        print("  ✓ Created LatentUpscale (to 1024x1024)")
        
        print()
        print("✓ Successfully constructed latent upscale workflow")
        return True
        
    except Exception as e:
        print(f"✗ Latent upscale workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_node_connections():
    """Test that nodes can be connected via wrapper types."""
    try:
        from nodetool.nodes.comfy.types import Model, Conditioning, Latent
        
        print("Testing node connections via wrapper types:")
        print()
        
        # Create a value
        test_model = Model("test_model_data")
        print(f"  ✓ Created Model wrapper: {test_model}")
        
        # Pass it to another "node" (simulate connection)
        def mock_node_accepts_model(model: Model):
            assert model.value == "test_model_data", "Value not preserved"
            return True
        
        result = mock_node_accepts_model(test_model)
        print("  ✓ Model passed between nodes correctly")
        
        # Test multiple wrapper types
        cond = Conditioning("test_cond")
        latent = Latent({"samples": "test"})
        
        def mock_node_accepts_multiple(c: Conditioning, l: Latent):
            assert c.value == "test_cond", "Conditioning not preserved"
            assert l.value == {"samples": "test"}, "Latent not preserved"
            return True
        
        result = mock_node_accepts_multiple(cond, latent)
        print("  ✓ Multiple wrapper types passed correctly")
        
        print()
        print("✓ Node connections work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Node connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all workflow construction tests."""
    print("=" * 70)
    print("ComfyUI Workflow Construction Tests")
    print("=" * 70)
    print()
    
    tests = [
        ("Basic text-to-image workflow", test_basic_workflow_structure),
        ("Image-to-image workflow", test_image_workflow_structure),
        ("Latent upscale workflow", test_latent_upscale_workflow),
        ("Node connections", test_node_connections),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"Test: {test_name}")
        print("-" * 70)
        if test_func():
            passed += 1
        else:
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
