"""
Example: Simple Text-to-Image with ComfyUI Nodes

This example demonstrates a basic text-to-image workflow using ComfyUI nodes
through the nodetool DSL interface. It shows:

1. Loading a checkpoint (model + CLIP + VAE)
2. Encoding text prompts (positive and negative)
3. Creating an empty latent
4. Sampling/denoising
5. Decoding to image

The workflow pattern:
    [CheckpointLoader] -> [CLIPTextEncode] -> [EmptyLatent] -> [KSampler] -> [VAEDecode] -> [Output]

This is the most basic Stable Diffusion workflow.
"""

# Note: This is a template for DSL-style usage
# Actual DSL imports would come from nodetool.dsl.comfy.* after running codegen

# Example of what the DSL usage would look like:

def build_text_to_image_workflow():
    """
    Build a simple text-to-image workflow using ComfyUI nodes.
    
    This function demonstrates:
    1. Loading a checkpoint (returns model, clip, vae)
    2. Encoding positive and negative prompts
    3. Creating empty latent space
    4. Sampling with KSampler
    5. Decoding latent to image
    
    Returns:
        Graph: A graph object representing the workflow
    """
    
    # For actual implementation, this would use:
    # from nodetool.dsl.graph import create_graph
    # from nodetool.dsl.comfy.loaders import CheckpointLoaderSimple
    # from nodetool.dsl.comfy.conditioning import CLIPTextEncode
    # from nodetool.dsl.comfy.latent import EmptyLatentImage
    # from nodetool.dsl.comfy.sampling import KSampler
    # from nodetool.dsl.comfy.latent import VAEDecode
    # from nodetool.dsl.nodetool.output import Output
    
    print("Text-to-Image Workflow Template")
    print("=" * 50)
    print()
    print("Workflow Structure:")
    print("  1. CheckpointLoaderSimple(ckpt_name='model.safetensors')")
    print("     -> outputs: model, clip, vae")
    print()
    print("  2. CLIPTextEncode(text='positive prompt', clip=checkpoint.clip)")
    print("     -> outputs: positive_conditioning")
    print()
    print("  3. CLIPTextEncode(text='negative prompt', clip=checkpoint.clip)")
    print("     -> outputs: negative_conditioning")
    print()
    print("  4. EmptyLatentImage(width=512, height=512, batch_size=1)")
    print("     -> outputs: latent")
    print()
    print("  5. KSampler(")
    print("       model=checkpoint.model,")
    print("       seed=42,")
    print("       steps=20,")
    print("       cfg=7.5,")
    print("       sampler_name='euler',")
    print("       scheduler='normal',")
    print("       positive=positive_encode.output,")
    print("       negative=negative_encode.output,")
    print("       latent_image=empty_latent.output,")
    print("       denoise=1.0")
    print("     )")
    print("     -> outputs: sampled_latent")
    print()
    print("  6. VAEDecode(samples=sampler.output, vae=checkpoint.vae)")
    print("     -> outputs: image")
    print()
    print("  7. Output(name='generated_image', value=decoder.output)")
    print()
    
    # Pseudo-code for actual DSL implementation:
    """
    # Load checkpoint
    checkpoint = CheckpointLoaderSimple(
        ckpt_name="sd_tiny.safetensors"
    )
    
    # Encode positive prompt
    positive_encode = CLIPTextEncode(
        text="a beautiful landscape with mountains and a lake",
        clip=checkpoint.clip
    )
    
    # Encode negative prompt  
    negative_encode = CLIPTextEncode(
        text="blurry, low quality, bad anatomy",
        clip=checkpoint.clip
    )
    
    # Create empty latent
    empty_latent = EmptyLatentImage(
        width=512,
        height=512,
        batch_size=1
    )
    
    # Sample
    sampler = KSampler(
        model=checkpoint.model,
        seed=42,
        steps=20,
        cfg=7.5,
        sampler_name="euler",
        scheduler="normal",
        positive=positive_encode.output,
        negative=negative_encode.output,
        latent_image=empty_latent.output,
        denoise=1.0
    )
    
    # Decode to image
    decoder = VAEDecode(
        samples=sampler.output,
        vae=checkpoint.vae
    )
    
    # Output
    output = Output(
        name="generated_image",
        value=decoder.output,
        description="Generated image from text prompt"
    )
    
    return create_graph(output)
    """


if __name__ == "__main__":
    """
    To run this example (after DSL codegen):
    
    1. Download sd-tiny model:
       huggingface-cli download segmind/sd-tiny sd_tiny.safetensors --local-dir models/checkpoints
    
    2. Run the workflow:
       python examples/text_to_image_basic.py
    """
    
    build_text_to_image_workflow()
    
    print()
    print("Notes:")
    print("  - This is a template showing the DSL structure")
    print("  - Actual implementation requires running 'nodetool codegen'")
    print("  - The DSL provides a Pythonic way to build ComfyUI workflows")
    print("  - Node outputs are accessible via properties (e.g., checkpoint.clip)")
    print("  - The graph is constructed declaratively and can be serialized")
