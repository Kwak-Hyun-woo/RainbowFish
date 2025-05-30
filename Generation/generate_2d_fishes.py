import torch
import numpy as np
import random
import os
import time
from PIL import Image
from IPython import display as IPdisplay
from tqdm.auto import tqdm
import argparse

from diffusers import StableDiffusionPipeline
from diffusers import (
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    AutoPipelineForText2Image
)
from diffusers.utils import load_image
from transformers import logging
os.environ["HF_HOME"] = "./cache"
logging.set_verbosity_error()

def save_images(images, save_path):
    os.makedirs(save_path, exist_ok=True)
    try:
        # Convert each image in the 'images' list from an array to an Image object.
        images = [Image.fromarray(np.array(image[0], dtype=np.uint8)) for image in images]

        for i, image in enumerate(images):
            # Generate a file name based on the current time, replacing colons with hyphens
            # to ensure the filename is valid for file systems that don't allow colons.
            # Save each image in the list as a PNG file at the 'save_path' location.
            images[i].save(f"{save_path}/fish_{i:04d}.png")

    except Exception as e:
        # If there is an error during the process, print the exception message.
        print(e)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guidance_scale", type=int, default=8, help="The guidance scale is set to its normal range (7 - 10).")
    parser.add_argument("--num_inference_steps", type=int, default=15, help="The number of inference steps was chosen empirically to generate an acceptable picture within an acceptable time.")
    parser.add_argument("--height", type=int, default=512, help="height")
    parser.add_argument("--width", type=int, default=512, help="width")
    parser.add_argument("--save_path", type=str, default="../data/2d_fishes_test", help="The guidance scale is set to its normal range (7 - 10).")
    parser.add_argument("--reference_image_path", type=str, default="reference_fish.png", help="The guidance scale is set to its normal range (7 - 10).")
    parser.add_argument("--num_fishes", type=int, default=150, help="The number of fishes to be generated")
    return parser
    
if __name__ == "__main__":  
    parser = get_parser()
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    guidance_scale = args.guidance_scale
    # The number of inference steps was chosen empirically to generate an acceptable picture within an acceptable time.
    num_inference_steps = args.num_inference_steps
    height = args.height
    width = args.width
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    reference_image_path = args.reference_image_path
    num_fishes = args.num_fishes
    
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", cache_dir="./cache", torch_dtype=torch.float16).to("cuda")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", cache_dir="./cache")

    pipe.set_ip_adapter_scale(0.6)

    # Disable image generation progress bar, we'll display our own
    pipe.set_progress_bar_config(disable=True)

    # Offloading the weights to the CPU and only loading them on the GPU can reduce memory consumption to less than 3GB.
    pipe.enable_model_cpu_offload()

    # Tighter ordering of memory tensors.
    pipe.unet.to(memory_format=torch.channels_last)

    # Decoding large batches of images with limited VRAM or batches with 32 images or more by decoding the batches of latents one image at a time.
    pipe.enable_vae_slicing()

    # Splitting the image into overlapping tiles, decoding the tiles, and then blending the outputs together to compose the final image.
    pipe.enable_vae_tiling()

    # Using Flash Attention; If you have PyTorch >= 2.0 installed, you should not expect a speed-up for inference when enabling xformers.
    pipe.enable_xformers_memory_efficient_attention()

    # The seed is set to "None", because we want different results each time we run the generation.
    seed = None
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    # Load the reference image and convert it to RGB format.
    reference_image = Image.open(reference_image_path).convert("RGB")
    reference_image = reference_image.resize((width, height))

    # List of fish species, colors, and descriptions to generate prompts

    fish_colors = [
        "vibrant orange", "neon blue", "deep yellow", "bright purple", "shimmering green",
        "striped red and white", "spotted turquoise", "metallic silver", "gradient pink and yellow",
        "iridescent teal", "glossy red", "translucent white", "patterned black and gold"
    ]

    fish_descriptions = [
        "studio lighting", "highly detailed", "realistic textures",
        "white backdrop", "centered composition", "detailed scales", "realistic lighting"
    ]

    # Negative prompts to avoid unwanted elements in the generated images
    negative_prompt = "multiple fishes, cartoon, anime, sketch, painting, 2d, drawing, low resolution, blurry, extra limbs, extra fins, deformed, bad anatomy, shadow, reflection, watermark, logo,\
                        noise, duplicate, background, close-up, stylized"

    # Generate 140 prompts and negative prompts
    prompts, negative_prompts = [], []
    for _ in range(num_fishes):
        color = random.choice(fish_colors)
        desc = ", ".join(random.sample(fish_descriptions, 4))
        prompt = f"A photorealistic side view of only one {color} fish full body with transparent background, {desc}"
        prompts.append(prompt)
        negative_prompts.append(negative_prompt)

    variants_images = []
    for idx, (prompt, negative_prompt) in tqdm(
        enumerate(zip(prompts, negative_prompts)),
        total=len(prompts),
    ):
        variants_images.append(
            pipe(
                height=height,
                width=width,
                num_images_per_prompt=1,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                ip_adapter_image=reference_image,
            ).images
        )
    save_images(variants_images, save_path)
    reference_image.save(f"{save_path}/reference_fish.png")
    # The prompt for the image generation. You can change this to whatever you want.
    import pickle
    with open("prompts.pkl", "wb") as f:
        pickle.dump({"prompts": prompts, "negative_prompts": negative_prompts}, f)

