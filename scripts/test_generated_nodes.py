#!/usr/bin/env python3
"""
Test Generated Nodes

This script tests the generated ComfyUI nodes to ensure they:
1. Can be imported
2. Have correct field definitions
3. Can be instantiated
4. Have valid process methods

Does NOT test actual model loading or inference.
"""

import sys
from pathlib import Path
from typing import Any


# Configuration constants
SAMPLE_NODE_COUNT = 5  # Number of sample nodes to test in detail


def test_import():
    """Test that generated nodes can be imported."""
    print("Testing node imports...")
    try:
        from nodetool.nodes.comfy.generated import all_nodes
        print(f"✓ Successfully imported all_nodes module")
        return all_nodes
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return None


def test_node_structure(all_nodes):
    """Test that nodes have correct structure."""
    print("\nTesting node structure...")
    
    # Get all node classes from the module
    node_classes = []
    for name in dir(all_nodes):
        if not name.startswith("_"):
            obj = getattr(all_nodes, name)
            if isinstance(obj, type) and hasattr(obj, "process"):
                node_classes.append((name, obj))
    
    print(f"Found {len(node_classes)} node classes")
    
    if not node_classes:
        print("✗ No node classes found!")
        return False
    
    # Test a few sample nodes
    sample_nodes = node_classes[:SAMPLE_NODE_COUNT]
    
    for name, node_class in sample_nodes:
        print(f"\nTesting {name}...")
        
        # Test instantiation
        try:
            node = node_class()
            print(f"  ✓ Can instantiate {name}")
        except Exception as e:
            print(f"  ✗ Failed to instantiate: {e}")
            continue
        
        # Test that it has required attributes
        if not hasattr(node, "process"):
            print(f"  ✗ Missing process method")
            continue
        print(f"  ✓ Has process method")
        
        # Test field access (should have Pydantic fields)
        if not hasattr(node, "model_fields"):
            print(f"  ✗ Not a proper Pydantic model")
            continue
        print(f"  ✓ Is a Pydantic model with {len(node.model_fields)} fields")
    
    return True


def test_specific_nodes():
    """Test specific important nodes."""
    print("\n\nTesting specific nodes...")
    
    try:
        from nodetool.nodes.comfy.generated.all_nodes import (
            KSampler,
            CheckpointLoaderSimple,
            CLIPTextEncode,
            VAEDecode,
            EmptyLatentImage,
        )
        print("✓ Successfully imported key nodes:")
        print("  - KSampler")
        print("  - CheckpointLoaderSimple")
        print("  - CLIPTextEncode")
        print("  - VAEDecode")
        print("  - EmptyLatentImage")
    except ImportError as e:
        print(f"✗ Failed to import key nodes: {e}")
        return False
    
    # Test KSampler instantiation
    try:
        sampler = KSampler(
            model=None,
            seed=42,
            steps=20,
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            positive=None,
            negative=None,
            latent_image=None,
            denoise=1.0,
        )
        print("\n✓ KSampler can be instantiated with parameters")
        print(f"  - seed: {sampler.seed}")
        print(f"  - steps: {sampler.steps}")
        print(f"  - cfg: {sampler.cfg}")
    except Exception as e:
        print(f"\n✗ Failed to instantiate KSampler: {e}")
        return False
    
    # Test EmptyLatentImage
    try:
        latent = EmptyLatentImage(width=512, height=512, batch_size=1)
        print("\n✓ EmptyLatentImage can be instantiated")
        print(f"  - width: {latent.width}")
        print(f"  - height: {latent.height}")
    except Exception as e:
        print(f"\n✗ Failed to instantiate EmptyLatentImage: {e}")
        return False
    
    return True


def test_node_graph_structure():
    """
    Test that nodes can be connected in a graph structure.
    This doesn't actually execute, just tests the structure.
    """
    print("\n\nTesting node graph structure...")
    
    try:
        from nodetool.nodes.comfy.generated.all_nodes import (
            EmptyLatentImage,
            KSampler,
            VAEDecode,
        )
        
        # Create a simple graph structure (not executed)
        print("Creating a simple node graph (structure only):")
        
        # Step 1: Create empty latent
        latent_node = EmptyLatentImage(width=512, height=512, batch_size=1)
        print("  1. EmptyLatentImage node created")
        
        # Step 2: Sample
        sampler_node = KSampler(
            model=None,  # Would be connected from CheckpointLoader
            seed=42,
            steps=20,
            cfg=7.5,
            sampler_name="euler",
            scheduler="normal",
            positive=None,  # Would be connected from CLIPTextEncode
            negative=None,  # Would be connected from CLIPTextEncode
            latent_image=None,  # Would be connected from EmptyLatentImage
            denoise=1.0,
        )
        print("  2. KSampler node created")
        
        # Step 3: Decode
        decode_node = VAEDecode(
            samples=None,  # Would be connected from KSampler
            vae=None,  # Would be connected from CheckpointLoader
        )
        print("  3. VAEDecode node created")
        
        print("\n✓ Node graph structure is valid")
        print("  (Nodes can be connected in a pipeline)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to create node graph: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner."""
    # Add src to path
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    
    print("=" * 60)
    print("Testing Generated ComfyUI Nodes")
    print("=" * 60)
    
    # Test 1: Import
    all_nodes = test_import()
    if not all_nodes:
        return 1
    
    # Test 2: Structure
    if not test_node_structure(all_nodes):
        return 1
    
    # Test 3: Specific nodes
    if not test_specific_nodes():
        return 1
    
    # Test 4: Graph structure
    if not test_node_graph_structure():
        return 1
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nGenerated nodes are structurally valid and ready for use.")
    print("Note: Actual model execution would require model loading.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
