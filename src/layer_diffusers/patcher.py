import torch

from .utils import cast_to_device, copy_to_param, set_attr

import re

# Define mappings for different parts of the name.
lora_to_diffusers_mapping = {
    'diffusion_model.middle_block': 'mid_block',
    'diffusion_model.input_blocks': 'down_blocks',
    'diffusion_model.output_blocks': 'up_blocks',
}


def convert_lora_to_diffusers(lora_name):
    """
    Converts a LoRA weight name to a Diffusers weight name.

    Args:
        lora_name (str): The LoRA weight name.

    Returns:
        str: The Diffusers weight name.
    """

    # Extract the numbers from the LoRA name.
    match = re.match(r'.*\.(\d+).(\d+)\..*\.(\d+)\.attn.*', lora_name)
    if match:
        head_num = int(match.group(3))
    else:
        # for middle block
        match = re.match(r'.*\.(\d+)\..*\.(\d+)\.attn.*', lora_name)
        head_num = int(match.group(2))
    layer_num = int(match.group(1))
    end_str = lora_name.split('attn')[-1]

    # Construct the Diffusers name.
    for k, v in lora_to_diffusers_mapping.items():
        if k in lora_name:
            diffusers_name = v
            break
    else:
        raise ValueError(f"{lora_name}")

    if diffusers_name == 'mid_block':
        diffusers_name += f'.attentions.{layer_num - 1}.transformer_blocks.{head_num}'
    elif diffusers_name == 'down_blocks':
        diffusers_name += f'.{(layer_num) // 3}.attentions.{(layer_num - 4) % 3}.transformer_blocks.{head_num}'
    elif diffusers_name == 'up_blocks':
        diffusers_name += f'.{layer_num // 3}.attentions.{layer_num % 3}.transformer_blocks.{head_num}'

    diffusers_name += f'.attn{end_str}'

    return diffusers_name


class UnetPatcher:
    def __init__(self, model, offload_device):
        model_sd = model.state_dict()
        self.model = model
        self.model_keys = set(model_sd.keys())
        self.patches = {}
        self.backup = {}
        self.offload_device = offload_device

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p_count = 0
        p_app_count = 0

        for k in patches:
            diffusers_name = convert_lora_to_diffusers(k)
            p_count += 1
            if diffusers_name in self.model_keys:
                p_app_count += 1
                current_patches = self.patches.get(diffusers_name, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[diffusers_name] = current_patches

    def load_frozen_patcher(self, state_dict, strength):
        patch_dict = {}
        for k, w in state_dict.items():
            model_key, patch_type, weight_index = k.split("::")
            if model_key not in patch_dict:
                patch_dict[model_key] = {}
            if patch_type not in patch_dict[model_key]:
                patch_dict[model_key][patch_type] = [None] * 16
            patch_dict[model_key][patch_type][int(weight_index)] = w

        patch_flat = {}
        for model_key, v in patch_dict.items():
            for patch_type, weight_list in v.items():
                patch_flat[model_key] = (patch_type, weight_list)

        self.add_patches(patches=patch_flat, strength_patch=float(strength), strength_model=1.0)
        return

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_model(self, device_to=None, patch_weights=True):
        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    print("could not patch. key doesn't exist in model:", key)
                    continue

                weight = model_sd[key]

                inplace_update = True  # condition? maybe

                if key not in self.backup:
                    self.backup[key] = weight.to(device=self.offload_device, copy=inplace_update)

                if device_to is not None:
                    temp_weight = cast_to_device(weight, device_to, torch.float32, copy=True)
                else:
                    temp_weight = weight.to(torch.float32, copy=True)
                out_weight = self.calculate_weight(self.patches[key], temp_weight, key).to(weight.dtype)
                if inplace_update:
                    copy_to_param(self.model, key, out_weight)
                else:
                    set_attr(self.model, key, out_weight)
                del temp_weight

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                raise NotImplementedError

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]
            else:
                raise Exception("Could not detect patch_type")

            if patch_type == "lora":  # lora/locon
                mat1 = cast_to_device(v[0], weight.device, torch.float32)
                mat2 = cast_to_device(v[1], weight.device, torch.float32)
                if v[2] is not None:
                    raise NotImplementedError
                if v[3] is not None:
                    raise NotImplementedError
                try:
                    weight += ((alpha * torch.mm(mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)))
                               .reshape(weight.shape).type(weight.dtype))
                except Exception as e:
                    print("ERROR", key, e)
            else:
                print("patch type not recognized", patch_type, key)

        return weight
