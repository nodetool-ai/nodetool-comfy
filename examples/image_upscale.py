"""
Example: Image Upscaling Workflow with ComfyUI Nodes

This example demonstrates an image upscaling workflow using ComfyUI nodes.
It shows:

1. Loading an existing image
2. Using VAE to encode to latent space
3. Upscaling the latent
4. Optional img2img refinement
5. Decoding back to image

The workflow pattern:
    [LoadImage] -> [VAEEncode] -> [LatentUpscale] -> [KSampler] -> [VAEDecode] -> [Output]

Useful for enhancing low-resolution images.
"""

# Note: This is a template for DSL-style usage
# Actual DSL imports would come from nodetool.dsl.comfy.* after running codegen


def build_image_upscale_workflow():
    """
    Build an image upscaling workflow using ComfyUI nodes.
    
    This function demonstrates:
    1. Loading an input image
    2. Encoding to latent space
    3. Upscaling in latent space (faster than pixel space)
    4. Optional refinement with img2img
    5. Decoding to high-res image
    
    Returns:
        Graph: A graph object representing the workflow
    """
    
    # For actual implementation, this would use:
    # from nodetool.dsl.graph import create_graph
    # from nodetool.dsl.comfy.loaders import CheckpointLoaderSimple
    # from nodetool.dsl.comfy.image import LoadImage
    # from nodetool.dsl.comfy.latent import VAEEncode, LatentUpscale, VAEDecode
    # from nodetool.dsl.comfy.conditioning import CLIPTextEncode
    # from nodetool.dsl.comfy.sampling import KSampler
    # from nodetool.dsl.nodetool.input import ImageInput, FloatInput
    # from nodetool.dsl.nodetool.output import Output
    
    print("Image Upscaling Workflow Template")
    print("=" * 50)
    print()
    print("Workflow Structure:")
    print()
    print("  1. CheckpointLoaderSimple(ckpt_name='model.safetensors')")
    print("     -> outputs: model, clip, vae")
    print()
    print("  2. LoadImage(image=input_image_path)")
    print("     -> outputs: image, mask")
    print()
    print("  3. VAEEncode(pixels=loaded_image.image, vae=checkpoint.vae)")
    print("     -> outputs: latent")
    print()
    print("  4. LatentUpscale(")
    print("       samples=encoded_latent.output,")
    print("       upscale_method='bilinear',")
    print("       width=1024,  # 2x upscale")
    print("       height=1024,")
    print("       crop='disabled'")
    print("     )")
    print("     -> outputs: upscaled_latent")
    print()
    print("  5. CLIPTextEncode(")
    print("       text='high quality, detailed, sharp',")
    print("       clip=checkpoint.clip")
    print("     )")
    print("     -> outputs: positive_conditioning")
    print()
    print("  6. CLIPTextEncode(")
    print("       text='blurry, low quality, artifacts',")
    print("       clip=checkpoint.clip")
    print("     )")
    print("     -> outputs: negative_conditioning")
    print()
    print("  7. KSampler(")
    print("       model=checkpoint.model,")
    print("       seed=42,")
    print("       steps=15,")
    print("       cfg=7.0,")
    print("       sampler_name='euler',")
    print("       scheduler='normal',")
    print("       positive=positive_encode.output,")
    print("       negative=negative_encode.output,")
    print("       latent_image=upscaled_latent.output,")
    print("       denoise=0.4  # Low denoise to preserve original")
    print("     )")
    print("     -> outputs: refined_latent")
    print()
    print("  8. VAEDecode(samples=refined_latent.output, vae=checkpoint.vae)")
    print("     -> outputs: upscaled_image")
    print()
    print("  9. Output(name='upscaled_image', value=decoder.output)")
    print()
    
    # Pseudo-code for actual DSL implementation:
    """
    # Inputs
    input_image = ImageInput(
        name="input_image",
        description="Image to upscale",
    )
    
    upscale_factor = FloatInput(
        name="upscale_factor",
        description="Upscaling factor (2.0 = double size)",
        value=2.0,
        min=1.0,
        max=4.0
    )
    
    denoise_strength = FloatInput(
        name="denoise_strength",
        description="Refinement strength (0 = no change, 1 = full redraw)",
        value=0.4,
        min=0.0,
        max=1.0
    )
    
    # Load checkpoint
    checkpoint = CheckpointLoaderSimple(
        ckpt_name="sd_tiny.safetensors"
    )
    
    # Load input image
    loaded_image = LoadImage(
        image=input_image.output
    )
    
    # Encode to latent
    encoded = VAEEncode(
        pixels=loaded_image.image,
        vae=checkpoint.vae
    )
    
    # Calculate target dimensions
    # (In real DSL, would need dimension calculation nodes)
    target_width = 1024
    target_height = 1024
    
    # Upscale in latent space
    upscaled_latent = LatentUpscale(
        samples=encoded.output,
        upscale_method="bilinear",
        width=target_width,
        height=target_height,
        crop="disabled"
    )
    
    # Encode prompts for refinement
    positive_encode = CLIPTextEncode(
        text="high quality, detailed, sharp, professional photograph",
        clip=checkpoint.clip
    )
    
    negative_encode = CLIPTextEncode(
        text="blurry, low quality, compression artifacts, pixelated",
        clip=checkpoint.clip
    )
    
    # Refine with img2img
    refined = KSampler(
        model=checkpoint.model,
        seed=42,
        steps=15,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive=positive_encode.output,
        negative=negative_encode.output,
        latent_image=upscaled_latent.output,
        denoise=denoise_strength.output  # Low denoise preserves original
    )
    
    # Decode to image
    decoder = VAEDecode(
        samples=refined.output,
        vae=checkpoint.vae
    )
    
    # Output
    output = Output(
        name="upscaled_image",
        value=decoder.output,
        description="High-resolution upscaled image"
    )
    
    return create_graph(output)
    """


if __name__ == "__main__":
    """
    To run this example (after DSL codegen):
    
    1. Download sd-tiny model:
       huggingface-cli download segmind/sd-tiny sd_tiny.safetensors --local-dir models/checkpoints
    
    2. Prepare an input image (e.g., input.png)
    
    3. Run the workflow:
       python examples/image_upscale.py
    
    The workflow will:
    - Encode your image to latent space
    - Upscale in latent space (efficient)
    - Optionally refine with img2img
    - Decode to high-res image
    """
    
    build_image_upscale_workflow()
    
    print()
    print("Key Benefits:")
    print("  - Upscaling in latent space is faster than pixel space")
    print("  - Low denoise strength preserves original image features")
    print("  - Can add detail enhancement via img2img refinement")
    print("  - Works with standard SD checkpoints")
    print()
    print("Notes:")
    print("  - This is a template showing the DSL structure")
    print("  - Actual implementation requires running 'nodetool codegen'")
    print("  - Adjust denoise strength to balance quality vs preservation")
    print("  - For best results, use a checkpoint trained for upscaling")
