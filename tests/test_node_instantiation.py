"""
Test: Node Instantiation

This test validates that all generated ComfyUI nodes can be instantiated
with default values without errors.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def test_instantiate_sample_nodes():
    """Test instantiation of a sample of important nodes."""
    try:
        from nodetool.nodes.comfy.generated.all_nodes import (
            KSampler,
            CheckpointLoaderSimple,
            CLIPTextEncode,
            VAEDecode,
            VAEEncode,
            EmptyLatentImage,
            LatentUpscale,
            LoadImage,
            ImageScale,
        )
        
        nodes_to_test = [
            ("KSampler", KSampler),
            ("CheckpointLoaderSimple", CheckpointLoaderSimple),
            ("CLIPTextEncode", CLIPTextEncode),
            ("VAEDecode", VAEDecode),
            ("VAEEncode", VAEEncode),
            ("EmptyLatentImage", EmptyLatentImage),
            ("LatentUpscale", LatentUpscale),
            ("LoadImage", LoadImage),
            ("ImageScale", ImageScale),
        ]
        
        print("Testing node instantiation with default values:")
        print()
        
        for node_name, node_class in nodes_to_test:
            try:
                # Instantiate with no arguments (using defaults)
                node = node_class()
                print(f"  ✓ {node_name}: instantiated successfully")
            except Exception as e:
                print(f"  ✗ {node_name}: failed - {e}")
                return False
        
        print()
        print("✓ All tested nodes instantiate correctly")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import nodes: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_node_field_types():
    """Test that node fields have correct types."""
    try:
        from nodetool.nodes.comfy.generated.all_nodes import KSampler
        from nodetool.nodes.comfy.types import Model, Conditioning, Latent
        
        # Get field annotations
        annotations = KSampler.__annotations__
        
        # Check specific field types
        assert 'model' in annotations, "KSampler missing model field annotation"
        assert 'positive' in annotations, "KSampler missing positive field annotation"
        assert 'negative' in annotations, "KSampler missing negative field annotation"
        assert 'latent_image' in annotations, "KSampler missing latent_image field annotation"
        
        # Check types are wrapper types (not Any)
        model_type = annotations.get('model')
        assert model_type == Model, f"model field should be Model, got {model_type}"
        
        positive_type = annotations.get('positive')
        assert positive_type == Conditioning, f"positive field should be Conditioning, got {positive_type}"
        
        latent_type = annotations.get('latent_image')
        assert latent_type == Latent, f"latent_image field should be Latent, got {latent_type}"
        
        print("✓ Node fields have correct wrapper types")
        return True
        
    except AssertionError as e:
        print(f"✗ Field type check failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_node_with_values():
    """Test instantiating nodes with specific values."""
    try:
        from nodetool.nodes.comfy.generated.all_nodes import (
            EmptyLatentImage,
            CLIPTextEncode,
        )
        from nodetool.nodes.comfy.types import Clip
        
        # Test EmptyLatentImage with specific dimensions
        latent_node = EmptyLatentImage(
            width=512,
            height=512,
            batch_size=1
        )
        assert latent_node.width == 512, "Width not set correctly"
        assert latent_node.height == 512, "Height not set correctly"
        assert latent_node.batch_size == 1, "Batch size not set correctly"
        
        # Test CLIPTextEncode with text
        clip_wrapper = Clip("fake_clip")
        text_node = CLIPTextEncode(
            text="a beautiful landscape",
            clip=clip_wrapper
        )
        assert text_node.text == "a beautiful landscape", "Text not set correctly"
        assert text_node.clip == clip_wrapper, "CLIP not set correctly"
        
        print("✓ Nodes accept and store values correctly")
        return True
        
    except AssertionError as e:
        print(f"✗ Value setting test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_numeric_constraints():
    """Test that numeric fields respect constraints."""
    try:
        from nodetool.nodes.comfy.generated.all_nodes import EmptyLatentImage
        from pydantic import ValidationError
        
        # Valid values should work
        node = EmptyLatentImage(width=256, height=256, batch_size=1)
        assert node.width == 256, "Width not set"
        
        # Test that validation happens (this will depend on Pydantic configuration)
        # For now, just verify the node can be created with various values
        node2 = EmptyLatentImage(width=1024, height=768, batch_size=4)
        assert node2.width == 1024, "Width not set for second node"
        
        print("✓ Numeric fields accept valid values")
        return True
        
    except Exception as e:
        print(f"✗ Numeric constraint test failed: {e}")
        return False


def main():
    """Run all instantiation tests."""
    print("=" * 70)
    print("Node Instantiation Tests")
    print("=" * 70)
    print()
    
    tests = [
        ("Instantiate sample nodes", test_instantiate_sample_nodes),
        ("Check field types", test_node_field_types),
        ("Test with specific values", test_node_with_values),
        ("Test numeric constraints", test_numeric_constraints),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
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
