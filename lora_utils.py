# This Python script is sourced from the following:
# 1) diffusers/examples/dreambooth/train_dreambooth_lora.py by Hugging Face
# 2) Shi, Yujun et al. DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing, CVPR 2024.

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, PretrainedConfig

import torch
import torch.nn.functional as F
from torchvision import transforms

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.utils import check_min_version
from diffusers.optimization import get_scheduler
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.utils.import_utils import is_xformers_available


check_min_version("0.17.0")

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=False):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

def train_lora(image,
               prompt,
               model_path,
               vae_path,
               save_lora_path,
               lora_step,
               lora_lr,
               lora_batch_size,
               lora_rank,
               progress,
               save_interval=-1):

    set_seed(0)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    noise_scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    text_encoder_cls = import_model_class_from_model_name_or_path(model_path, revision=None)
    text_encoder = text_encoder_cls.from_pretrained(
        model_path, subfolder="text_encoder", revision=None
    )
    if vae_path == "default":
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", revision=None
        )
    else:
        vae = AutoencoderKL.from_pretrained(vae_path)
    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", revision=None
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(device, dtype=torch.float16)
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    unet_lora_attn_procs = {}

    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]

        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise NotImplementedError("name must start with up_blocks, mid_blocks, or down_blocks")

        if isinstance(attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = (
                LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
            )
        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
        )

    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    params_to_optimize = (unet_lora_layers.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=lora_step,
        num_cycles=1,
        power=1.0,
    )

    unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    with torch.no_grad():
        text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None)
        text_embedding = encode_prompt(
            text_encoder,
            text_inputs.input_ids,
            text_inputs.attention_mask,
            text_encoder_use_attention_mask=False
        )
        text_embedding = text_embedding.repeat(lora_batch_size, 1, 1)

    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    for step in tqdm(range(lora_step), desc="LoRA Fine-tuning"):
        unet.train()
        image_batch = []
        for _ in range(lora_batch_size):
            image_transformed = image_transforms(Image.fromarray(image)).to(device, dtype=torch.float16)
            image_transformed = image_transformed.unsqueeze(dim=0)
            image_batch.append(image_transformed)

        image_batch = torch.cat(image_batch, dim=0)
        latents_dist = vae.encode(image_batch).latent_dist
        model_input = latents_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(model_input)
        bsz, channels, height, width = model_input.shape

        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
        )
        timesteps = timesteps.long()

        noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

        model_pred = unet(noisy_model_input, timesteps, text_embedding).sample

        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if save_interval > 0 and (step + 1) % save_interval == 0:
            save_lora_path_intermediate = os.path.join(save_lora_path, str(step+1))
            if not os.path.isdir(save_lora_path_intermediate):
                os.mkdir(save_lora_path_intermediate)

            LoraLoaderMixin.save_lora_weights(
                save_directory=save_lora_path_intermediate,
                unet_lora_layers=unet_lora_layers,
                text_encoder_lora_layers=None,
            )

    LoraLoaderMixin.save_lora_weights(
        save_directory=save_lora_path,
        unet_lora_layers=unet_lora_layers,
        text_encoder_lora_layers=None,
    )

    return
