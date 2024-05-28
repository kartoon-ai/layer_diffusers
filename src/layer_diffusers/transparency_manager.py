import os
from PIL import Image

from .utils import (get_torch_device, load_file_from_url, load_torch_file, save_latent, latents_to_img,
                    tensor_to_numpy_img, generate_img_with_checkerboard)
from .patcher import UnetPatcher
from .models import TransparentVAEDecoder


class TransparencyManager:
    def __init__(self, pipe, lora_weight=1., device=get_torch_device()):
        self.pipe = pipe
        self.lora_weight = lora_weight
        self.device = device
        self.latents_save_list = []

        # instantiate the transparent VAE decoder
        layer_model_root = os.path.join(os.path.expanduser("~"), ".cache", "layer_model")
        os.makedirs(layer_model_root, exist_ok=True)
        vae_decoder_model_path = load_file_from_url(
            url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors",
            model_dir=layer_model_root, file_name="vae_transparent_decoder.safetensors",
        )
        self.vae_transparent_decoder = TransparentVAEDecoder(load_torch_file(vae_decoder_model_path), fp16=False)

        # load LoRA that modifies attention to encode alpha information
        lora_model_path = load_file_from_url(
            url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors",
            model_dir=layer_model_root,
            file_name="layer_xl_transparent_attn.safetensors",
        )
        self.layer_lora_model = load_torch_file(lora_model_path, safe_load=True)

    def patch_pipe(self):
        self.pipe.vae_transparent_decoder = self.vae_transparent_decoder

        # add wrapper to save latent output
        self.pipe.vae.decode = save_latent(self.pipe.vae.decode, self.latents_save_list)

        unet_patcher = UnetPatcher(self.pipe.unet, self.device)
        unet_patcher.load_frozen_patcher(self.layer_lora_model, self.lora_weight)
        unet = unet_patcher.patch_model(device_to=self.device)
        self.pipe.unet = unet

    def post_process_transparency(self):
        latents = self.latents_save_list[-1]

        pixels = latents_to_img(self.pipe, latents)
        pixel_with_alpha = self.pipe.vae_transparent_decoder.decode_pixel(pixels, latents)

        # [B, C, H, W] => [B, H, W, C]
        pixel_with_alpha = pixel_with_alpha.movedim(1, -1)
        pixels_rgb = pixel_with_alpha[0, ..., 1:]

        alpha = pixel_with_alpha[..., 0]
        checkerboard_image = generate_img_with_checkerboard(pixels_rgb, alpha.movedim(0, -1))

        return pixels[0].movedim(0, -1), pixels_rgb, alpha[0], checkerboard_image[0]
