# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import math
import torch
import safetensors.torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import model_info
from transformers.models.clip.modeling_clip import CLIPTextModel


# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label


def get_base_model_name(model_name):
    return model_info(model_name).cardData.get("base_model", model_name)


def is_lora_model(model_name):
    # TODO: might be a better way to detect (e.g. presence of LORA weights file)
    return model_name != get_base_model_name(model_name)


# Merges LoRA weights into the layers of a base model
def merge_lora_weights(base_model, lora_model_path: str, submodel_name="unet", scale=1.0):
    from collections import defaultdict
    from functools import reduce

    from diffusers.models.attention_processor import LoRAAttnProcessor

    # Load LoRA weights
    if lora_model_path.split('.')[-1].lower() == "safetensors":
        lora_state_dict = safetensors.torch.load_file(lora_model_path, device="cpu")
    else:
        lora_state_dict = torch.load(lora_model_path, map_location="cpu")

    """
    Merging LoRA
    kohya-ss/sd-scripts
    networks/merge_lora.py 37-102
    https://github.com/kohya-ss/sd-scripts/blob/main/networks/merge_lora.py
    http://www.apache.org/licenses/LICENSE-2.0
    Copyright [2022] [kohya-ss]
    """
    # create module map
    name_to_module = {}
    if submodel_name == "text_encoder":
        prefix = "lora_te"
        target_replace_modules = ["CLIPAttention", "CLIPMLP"]
    if submodel_name == "unet":
        prefix = "lora_unet"
        target_replace_modules = ["Transformer2DModel", "Attention", "ResnetBlock2D", "Downsample2D", "Upsample2D"]

    for name, module in base_model.named_modules():
        if module.__class__.__name__ in target_replace_modules:
            for child_name, child_module in module.named_modules():
                if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                    lora_name = prefix + "." + name + "." + child_name
                    lora_name = lora_name.replace(".", "_")
                    name_to_module[lora_name] = child_module

    ratio = 1.0 # 병합비

    for key in lora_state_dict.keys():
        if "lora_down" in key:
            up_key = key.replace("lora_down", "lora_up")
            alpha_key = key[: key.index("lora_down")] + "alpha"

            # find original module for this lora
            module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
            if module_name not in name_to_module:
                print(f"no module found for LoRA weight: {key}")
                continue
            module = name_to_module[module_name]
            # print(f"apply {key} to {module}")

            down_weight = lora_state_dict[key].type(torch.float32)
            up_weight = lora_state_dict[up_key].type(torch.float32)

            dim = down_weight.size()[0]
            alpha = lora_state_dict.get(alpha_key, dim)
            scale = alpha / dim

            # W <- W + U * D
            weight = module.weight
            # print(module_name, down_weight.size(), up_weight.size())
            if len(weight.size()) == 2:
                # linear
                weight = weight + ratio * torch.mm(up_weight, down_weight) * scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + ratio
                    * torch.mm(up_weight.squeeze(3).squeeze(2), down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                # print(conved.size(), weight.size(), module.stride, module.padding)
                weight = weight + ratio * conved * scale

            module.weight = torch.nn.Parameter(weight)
            print(module.weight)
    """
    # All keys in the LoRA state dictionary should have 'lora' somewhere in the string.
    keys = list(lora_state_dict.keys())
    assert all("lora" in k for k in keys)

    if all(key.startswith(submodel_name) for key in keys):
        # New format (https://github.com/huggingface/diffusers/pull/2918) supports LoRA weights in both the
        # unet and text encoder where keys are prefixed with 'unet' or 'text_encoder', respectively.
        submodel_state_dict = {k: v for k, v in lora_state_dict.items() if k.startswith(submodel_name)}
    else:
        # Old format. Keys will not have any prefix. This only applies to unet, so exit early if this is
        # optimizing the text encoder.
        if submodel_name != "unet":
            return
        submodel_state_dict = lora_state_dict

    # Group LoRA weights into attention processors
    attn_processors = {}
    lora_grouped_dict = defaultdict(dict)
    for key, value in submodel_state_dict.items():
        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
        lora_grouped_dict[attn_processor_key][sub_key] = value

    for key, value_dict in lora_grouped_dict.items():
        rank = value_dict["to_k_lora.down.weight"].shape[0]
        cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
        hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

        attn_processors[key] = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
        )
        attn_processors[key].load_state_dict(value_dict)

    # Merge LoRA attention processor weights into existing Q/K/V/Out weights
    for name, proc in attn_processors.items():
        attention_name = name[: -len(".processor")]
        attention = reduce(getattr, attention_name.split(sep="."), base_model)
        attention.to_q.weight.data += scale * torch.mm(proc.to_q_lora.up.weight, proc.to_q_lora.down.weight)
        attention.to_k.weight.data += scale * torch.mm(proc.to_k_lora.up.weight, proc.to_k_lora.down.weight)
        attention.to_v.weight.data += scale * torch.mm(proc.to_v_lora.up.weight, proc.to_v_lora.down.weight)
        attention.to_out[0].weight.data += scale * torch.mm(proc.to_out_lora.up.weight, proc.to_out_lora.down.weight)
    """


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------


def text_encoder_inputs(batchsize, torch_dtype):
    return torch.zeros((batchsize, 77), dtype=torch_dtype)


def text_encoder_load(model_name):
    checkpoint_path = os.environ.get("OLIVE_CKPT_PATH")
    loras: list[str] = os.environ.get("OLIVE_LORAS", '').split('$')
    model = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder")
    for lora in loras:
        if lora:
            filename = lora.split('\\')[-1]
            print(f"Merging LoRA {filename}...")
            merge_lora_weights(model, os.path.join(os.environ.get("OLIVE_LORA_BASE_PATH"), lora), "text_encoder")
    return model


def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int32)


def text_encoder_data_loader(data_dir, batchsize):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int32)


# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------


def unet_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 4, int(os.environ.get("OLIVE_SAMPLE_HEIGHT_DIM", 64)), int(os.environ.get("OLIVE_SAMPLE_WIDTH_DIM", 64))), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, 77, 768), dtype=torch_dtype),
        "return_dict": False,
    }


def unet_load(model_name):
    checkpoint_path = os.environ.get("OLIVE_CKPT_PATH")
    loras: list[str] = os.environ.get("OLIVE_LORAS", '').split('$')
    model = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")
    for lora in loras:
        if lora:
            filename = lora.split('\\')[-1]
            print(f"Merging LoRA {filename}...")
            merge_lora_weights(model, os.path.join(os.environ.get("OLIVE_LORA_BASE_PATH"), lora), "unet")
    return model


def unet_conversion_inputs(model):
    return tuple(unet_inputs(1, torch.float32).values())


def unet_data_loader(data_dir, batchsize):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------


def vae_encoder_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 3, int(os.environ.get("OLIVE_SAMPLE_HEIGHT", 512)), int(os.environ.get("OLIVE_SAMPLE_WIDTH", 512))), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_encoder_load(model_name):
    subfolder = os.environ.get("OLIVE_VAE_SUBFOLDER") or None
    model = AutoencoderKL.from_pretrained(os.environ.get("OLIVE_VAE"), subfolder=subfolder)
    model.forward = lambda sample, return_dict: model.encode(sample, return_dict)[0].sample()
    return model


def vae_encoder_conversion_inputs(model):
    return tuple(vae_encoder_inputs(1, torch.float32).values())


def vae_encoder_data_loader(data_dir, batchsize):
    return RandomDataLoader(vae_encoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------


def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand((batchsize, 4, int(os.environ.get("OLIVE_SAMPLE_HEIGHT_DIM", 64)), int(os.environ.get("OLIVE_SAMPLE_WIDTH_DIM", 64))), dtype=torch_dtype),
        "return_dict": False,
    }


def vae_decoder_load(model_name):
    subfolder = os.environ.get("OLIVE_VAE_SUBFOLDER") or None
    model = AutoencoderKL.from_pretrained(os.environ.get("OLIVE_VAE"), subfolder=subfolder)
    model.forward = model.decode
    return model


def vae_decoder_conversion_inputs(model):
    return tuple(vae_decoder_inputs(1, torch.float32).values())


def vae_decoder_data_loader(data_dir, batchsize):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# SAFETY CHECKER
# -----------------------------------------------------------------------------


def safety_checker_inputs(batchsize, torch_dtype):
    return {
        "clip_input": torch.rand((batchsize, 3, 224, 224), dtype=torch_dtype),
        "images": torch.rand((batchsize, int(os.environ.get("OLIVE_SAMPLE_HEIGHT", 512)), int(os.environ.get("OLIVE_SAMPLE_WIDTH", 512)), 3), dtype=torch_dtype),
    }


def safety_checker_load(model_name):
    model = StableDiffusionSafetyChecker.from_pretrained(os.environ.get("OLIVE_CKPT_PATH"), subfolder="safety_checker")
    model.forward = model.forward_onnx
    return model


def safety_checker_conversion_inputs(model):
    return tuple(safety_checker_inputs(1, torch.float32).values())


def safety_checker_data_loader(data_dir, batchsize):
    return RandomDataLoader(safety_checker_inputs, batchsize, torch.float16)
