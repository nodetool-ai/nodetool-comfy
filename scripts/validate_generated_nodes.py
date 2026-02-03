#!/usr/bin/env python3
"""
Simple validation of generated nodes

This script validates the generated nodes without requiring nodetool to be installed.
It checks:
1. Python syntax is valid
2. File structure looks correct
3. All expected nodes are present
"""

import ast
import json
from pathlib import Path


# Configuration constants
NODE_COUNT_THRESHOLD = 0.9  # Minimum ratio of expected to actual nodes


def validate_syntax(file_path: Path) -> bool:
    """Validate Python syntax of generated file."""
    print(f"Validating syntax of {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        print("✓ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False


def count_classes(file_path: Path) -> int:
    """Count the number of classes in the generated file."""
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    tree = ast.parse(source)
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    return len(classes)


def validate_structure(file_path: Path, expected_nodes: int) -> bool:
    """Validate the structure of the generated file."""
    print(f"\nValidating structure...")
    
    num_classes = count_classes(file_path)
    print(f"  Found {num_classes} classes")
    
    if num_classes == 0:
        print("  ✗ No classes found!")
        return False
    
    if num_classes < expected_nodes * NODE_COUNT_THRESHOLD:
        print(f"  ⚠ Expected around {expected_nodes} nodes, found {num_classes}")
    else:
        print(f"  ✓ Class count looks good")
    
    # Check for imports
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    required_imports = [
        "from nodetool.workflows.base_node import BaseNode",
        "from nodetool.workflows.processing_context import ProcessingContext",
        "from pydantic import Field",
    ]
    
    for imp in required_imports:
        if imp in content:
            print(f"  ✓ Found: {imp}")
        else:
            print(f"  ✗ Missing: {imp}")
            return False
    
    return True


def check_sample_nodes(file_path: Path) -> bool:
    """Check that some key nodes are present."""
    print("\nChecking for key nodes...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    key_nodes = [
        "KSampler",
        "CheckpointLoaderSimple",
        "CLIPTextEncode",
        "VAEDecode",
        "EmptyLatentImage",
    ]
    
    all_found = True
    for node_name in key_nodes:
        if f"class {node_name}(BaseNode):" in content:
            print(f"  ✓ Found {node_name}")
        else:
            print(f"  ✗ Missing {node_name}")
            all_found = False
    
    return all_found


def validate_process_methods(file_path: Path) -> bool:
    """Validate that classes have process methods."""
    print("\nValidating process methods...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    classes_without_process = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            has_process = any(
                isinstance(item, ast.AsyncFunctionDef) and item.name == "process"
                for item in node.body
            )
            if not has_process:
                classes_without_process.append(node.name)
    
    if classes_without_process:
        print(f"  ✗ {len(classes_without_process)} classes missing process method:")
        for name in classes_without_process[:5]:
            print(f"    - {name}")
        if len(classes_without_process) > 5:
            print(f"    ... and {len(classes_without_process) - 5} more")
        return False
    else:
        print("  ✓ All classes have process method")
        return True


def main():
    """Main validation."""
    print("=" * 60)
    print("Validating Generated ComfyUI Nodes")
    print("=" * 60)
    
    repo_root = Path(__file__).parent.parent
    generated_file = repo_root / "src" / "nodetool" / "nodes" / "comfy" / "generated" / "all_nodes.py"
    
    if not generated_file.exists():
        print(f"✗ Generated file not found: {generated_file}")
        return 1
    
    # Load metadata to get expected node count
    metadata_file = repo_root / "comfy_nodes_metadata.json"
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    expected_nodes = len(metadata["comfy_nodes"])
    # Subtract deprecated nodes
    expected_nodes -= sum(1 for n in metadata["comfy_nodes"] if n.get("deprecated", False))
    
    print(f"Expected approximately {expected_nodes} nodes (non-deprecated)")
    print()
    
    # Run validations
    all_valid = True
    
    if not validate_syntax(generated_file):
        all_valid = False
    
    if not validate_structure(generated_file, expected_nodes):
        all_valid = False
    
    if not check_sample_nodes(generated_file):
        all_valid = False
    
    if not validate_process_methods(generated_file):
        all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("All validations passed! ✓")
        print("=" * 60)
        print("\nGenerated nodes are valid and ready to use.")
        return 0
    else:
        print("Some validations failed ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
