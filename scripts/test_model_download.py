#!/usr/bin/env python3
"""
Test script to demonstrate model download for CI

This script simulates what happens in the GitHub Actions workflow
when downloading the sd-tiny model.
"""

print("=" * 70)
print("Testing Model Download for CI")
print("=" * 70)
print()

# Test 1: Check huggingface_hub is available
print("1. Checking huggingface_hub installation...")
try:
    from huggingface_hub import hf_hub_download
    print("   ✓ huggingface_hub is installed")
except ImportError:
    print("   ✗ huggingface_hub not found")
    print("   Install with: pip install huggingface_hub")
    exit(1)

print()

# Test 2: Show what the download command would do
print("2. Model download command (simulated):")
print()
print("   from huggingface_hub import hf_hub_download")
print("   file_path = hf_hub_download(")
print("       repo_id='segmind/sd-tiny',")
print("       filename='sd_tiny.safetensors',")
print("       local_dir='models/checkpoints',")
print("       local_dir_use_symlinks=False")
print("   )")
print()

# Test 3: Show expected output
print("3. Expected output when download succeeds:")
print()
print("   Downloading sd-tiny model from HuggingFace...")
print("   Fetching 1 files: 100%|████████████████| 1/1 [00:15<00:00, 15.23s/it]")
print("   ✓ Downloaded successfully to: models/checkpoints/sd_tiny.safetensors")
print("   ✓ File size: 153.4 MB")
print()

# Test 4: Show fallback behavior
print("4. Fallback behavior if Python download fails:")
print()
print("   ✗ Download failed: [Network error or timeout]")
print("   Trying curl fallback...")
print("   Downloading with curl...")
print("   % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current")
print("                                  Dload  Upload   Total   Spent    Left  Speed")
print("   100  153M  100  153M    0     0  15.2M      0  0:00:10  0:00:10 --:--:-- 18.1M")
print("   ✓ Downloaded via curl")
print()

# Test 5: Show caching behavior
print("5. When model is already cached:")
print()
print("   ✓ Model already cached: models/checkpoints/sd_tiny.safetensors")
print("   -rw-r--r-- 1 runner runner 153M Feb  2 10:00 models/checkpoints/sd_tiny.safetensors")
print()

print("=" * 70)
print("Summary")
print("=" * 70)
print()
print("The workflow uses a two-stage download approach:")
print("  1. Try Python API (hf_hub_download) - faster, better error handling")
print("  2. Fallback to curl if Python API fails - more reliable for CI")
print()
print("Model is cached between CI runs for efficiency.")
print("Cache key: ${{ runner.os }}-models-sd-tiny")
print()
