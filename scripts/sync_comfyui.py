#!/usr/bin/env python3
"""
Sync ComfyUI Source

This script syncs the ComfyUI source code from the submodule to the src/ directory
for use by the nodetool-comfy package.

It copies:
- comfy/ - Core comfy library
- comfy_api/ - API types and helpers
- comfy_extras/ - Extra node definitions
- comfy_config/ - Configuration types
- comfy_execution/ - Execution engine
- nodes.py - Base node definitions
- node_helpers.py - Node helper functions
- folder_paths.py - File path utilities
- latent_preview.py - Latent preview utilities
- protocol.py - Protocol definitions
"""

import shutil
import sys
from pathlib import Path


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def sync_comfyui_source():
    """Sync ComfyUI source from submodule to src/."""
    repo_root = get_repo_root()
    comfyui_path = repo_root / "ComfyUI"
    src_path = repo_root / "src"
    
    if not comfyui_path.exists():
        print(f"Error: ComfyUI submodule not found at {comfyui_path}")
        print("Please run: git submodule update --init")
        return False
    
    # Directories to sync
    dirs_to_sync = [
        "comfy",
        "comfy_api",
        "comfy_extras",
        "comfy_config",
        "comfy_execution",
    ]
    
    # Files to sync
    files_to_sync = [
        "nodes.py",
        "node_helpers.py",
        "folder_paths.py",
        "latent_preview.py",
        "protocol.py",
    ]
    
    print(f"Syncing ComfyUI source from {comfyui_path} to {src_path}")
    
    # Sync directories
    for dir_name in dirs_to_sync:
        src_dir = comfyui_path / dir_name
        dst_dir = src_path / dir_name
        
        if not src_dir.exists():
            print(f"  Warning: Source directory not found: {src_dir}")
            continue
        
        # Remove existing directory if it exists
        if dst_dir.exists():
            print(f"  Removing existing: {dst_dir}")
            shutil.rmtree(dst_dir)
        
        print(f"  Copying: {dir_name}/")
        shutil.copytree(src_dir, dst_dir)
    
    # Sync files
    for file_name in files_to_sync:
        src_file = comfyui_path / file_name
        dst_file = src_path / file_name
        
        if not src_file.exists():
            print(f"  Warning: Source file not found: {src_file}")
            continue
        
        print(f"  Copying: {file_name}")
        shutil.copy2(src_file, dst_file)
    
    print("\nSync complete!")
    return True


def main():
    """Main entry point."""
    success = sync_comfyui_source()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
