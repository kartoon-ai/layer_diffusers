import os

from IPython.display import display
from PIL import Image
import torch

import diffusers
from diffusers import StableDiffusionXLPipeline

from layer_diffusers.utils import get_torch_device, load_file_from_url, load_torch_file, tensor_to_numpy_img
from layer_diffusers.transparency_manager import TransparencyManager


def create_image_grid(images, cols=None, rows=1):
    if not isinstance(images, list) or len(images) == 0:
        return None

    if cols is None:
        cols = len(images)
    if len(images) != cols * rows:
        raise ValueError("Number of images does not match grid size")

    image_width, image_height = images[0].size

    grid_width = cols * image_width
    grid_height = rows * image_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Paste each image into the grid
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        grid_image.paste(image, (col * image_width, row * image_height))

    return grid_image


print(f"using diffusers version {diffusers.__version__}")

device = get_torch_device()
SEED = 123
gen = torch.Generator().manual_seed(SEED)

model_name = "RunDiffusion/Juggernaut-XL-v9"


prompts = [
    "a glass bottle, high quality",
    "a woman with messy hair",
    "a teenage boy with messy hair",
    "an octopus with many tentacles",
]
prompt_append = ", best quality"
lora_weight = 1.

pipe = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,).to(device)

try:
    import xformers
    pipe.enable_vae_tiling()
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers is installed")
except ImportError:
    print("xformers is not installed, skipping some optimization")

transparency_manager = TransparencyManager(pipe, lora_weight)
transparency_manager.patch_pipe()

alpha_output_images = []
final_output_images = []
simple_pixel_output_images = []
checkerboard_output_images = []

for prompt in prompts:
    prompt += prompt_append
    print(prompt)
    images = pipe(prompt=prompt, negative_prompt="bad, ugly", num_inference_steps=20,
                  width=1024, height=1024, generator=gen).images

    pixels, pixels_rgb, alpha, checkerboard_image = transparency_manager.post_process_transparency()

    simple_pixel_output_images.append(Image.fromarray(tensor_to_numpy_img(pixels)))
    final_output_images.append(Image.fromarray(tensor_to_numpy_img(pixels_rgb)))
    alpha_output_images.append(Image.fromarray(tensor_to_numpy_img(alpha)))
    checkerboard_output_images.append(Image.fromarray(tensor_to_numpy_img(checkerboard_image)))

# whether to save to disk
save = True
output_folder = f"outputs/{model_name.split('/')[-1].split('.')[0]}"

pairs = [
    ('pixels', simple_pixel_output_images),
    ('alpha', alpha_output_images),
    ('checkerboard', checkerboard_output_images),
    ('rgb', final_output_images),  # <- RGB is actually not a very interesting output (due to the gaussian blur)
]

for (s, imlist) in pairs:
    if save:
        os.makedirs(output_folder, exist_ok=True)
        for i in range(len(prompts)):
            safe_filename = prompts[i].split(',')[0].replace(' ', '_')
            imlist[i].save(f"{output_folder}/{safe_filename}_{s}.png")
        print(f"Saved {len(prompts)} images to {output_folder}/xxx_{s}.png")