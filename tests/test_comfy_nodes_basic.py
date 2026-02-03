"""
Test: Basic ComfyUI Node Validation

This test validates that generated ComfyUI nodes have correct structure
without requiring model loading or execution.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def test_nodes_import():
    """Test that generated nodes can be imported."""
    try:
        from nodetool.nodes.comfy.generated import all_nodes
        print("✓ Successfully imported generated nodes module")
        return True
    except ImportError as e:
        print(f"✗ Failed to import nodes: {e}")
        return False


def test_wrapper_types_import():
    """Test that wrapper types can be imported."""
    try:
        from nodetool.nodes.comfy.types import (
            Model, Clip, Vae, Conditioning, Latent, Mask
        )
        print("✓ Successfully imported wrapper types")
        return True
    except ImportError as e:
        print(f"✗ Failed to import wrapper types: {e}")
        return False


def test_node_structure():
    """Test that nodes have required attributes."""
    try:
        from nodetool.nodes.comfy.generated.all_nodes import (
            KSampler,
            CheckpointLoaderSimple,
            CLIPTextEncode,
            VAEDecode,
        )
        
        # Check KSampler has required fields
        assert hasattr(KSampler, 'model'), "KSampler missing model field"
        assert hasattr(KSampler, 'process'), "KSampler missing process method"
        
        # Check CheckpointLoader
        assert hasattr(CheckpointLoaderSimple, 'ckpt_name'), "CheckpointLoader missing ckpt_name"
        assert hasattr(CheckpointLoaderSimple, 'process'), "CheckpointLoader missing process"
        
        # Check CLIPTextEncode
        assert hasattr(CLIPTextEncode, 'text'), "CLIPTextEncode missing text field"
        assert hasattr(CLIPTextEncode, 'clip'), "CLIPTextEncode missing clip field"
        
        # Check VAEDecode
        assert hasattr(VAEDecode, 'samples'), "VAEDecode missing samples field"
        assert hasattr(VAEDecode, 'vae'), "VAEDecode missing vae field"
        
        print("✓ All tested nodes have required structure")
        return True
    except AssertionError as e:
        print(f"✗ Node structure validation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_wrapper_type_functionality():
    """Test that wrapper types work correctly."""
    try:
        from nodetool.nodes.comfy.types import Model, Conditioning, Latent
        
        # Test Model wrapper
        test_value = "test_model"
        model = Model(test_value)
        assert model.value == test_value, "Model wrapper value mismatch"
        assert bool(model), "Model wrapper bool should be True with value"
        
        # Test empty wrapper
        empty_model = Model()
        assert not bool(empty_model), "Empty Model wrapper bool should be False"
        
        # Test Conditioning wrapper
        cond = Conditioning("test_conditioning")
        assert cond.value == "test_conditioning", "Conditioning wrapper value mismatch"
        
        # Test Latent wrapper
        latent = Latent({"samples": "test"})
        assert latent.value == {"samples": "test"}, "Latent wrapper value mismatch"
        
        print("✓ Wrapper types function correctly")
        return True
    except AssertionError as e:
        print(f"✗ Wrapper type test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error in wrapper test: {e}")
        return False


def main():
    """Run all basic validation tests."""
    print("=" * 70)
    print("Basic ComfyUI Node Validation Tests")
    print("=" * 70)
    print()
    
    tests = [
        ("Import generated nodes", test_nodes_import),
        ("Import wrapper types", test_wrapper_types_import),
        ("Validate node structure", test_node_structure),
        ("Test wrapper functionality", test_wrapper_type_functionality),
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
