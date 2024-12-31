import os
import re
import cv2
import glob
import time
import shutil
import einops
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from accelerate import Accelerator
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.nn import functional as F 

from controlnet_aux import HEDdetector
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetPipeline

from main_utils import *
from noise_init import *
from lora_utils import train_lora


logging.set_verbosity_error()

class Pipeline(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_key = "runwayml/stable-diffusion-v1-5"

        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_key, torch_dtype=torch.float16).to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet_pre = deepcopy(self.pipe.unet) # Opted for deepcopy() due to time constraints; revisit if time permits.
        self.unet_post = deepcopy(self.pipe.unet) # Opted for deepcopy() due to time constraints; revisit if time permits.

        self.scheduler = DDIMScheduler.from_pretrained(self.model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.config["device"])

        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16).cuda()

        self.controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None, torch_dtype=torch.float16
        )
        self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.unet_pre.parameters():
            param.requires_grad = False
        for param in self.unet_post.parameters():
            param.requires_grad = False

        self.lora_image = None
        self.controlnet_image = None
        self.is_preprocess_stage = True
        self.self_attn_buffer = {}

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.config["device"]))[0]
        uncond_input = self.tokenizer(negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                      return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.config["device"]))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def load_img(self, image_path):
        lora_image = Image.open(image_path).convert("RGB")
        lora_image = np.array(lora_image)
        current_average = np.mean(lora_image)
        target_average = 40.0
        scaling_factor = target_average / current_average
        if current_average < target_average:
            lora_image = (lora_image * scaling_factor).clip(0,255).astype(np.uint8)
        lora_image = Image.fromarray(lora_image).resize((512, 512))
        lora_image = np.array(lora_image)
        
        image_np = lora_image.copy()
        input_tensor = T.ToTensor()(image_np).unsqueeze(0).to(self.config["device"])

        edge_img = cv2.imread(config["data_path"])
        current_average = np.mean(edge_img)
        target_average = 70.0
        scaling_factor = target_average / current_average
        if current_average < target_average:
            edge_img = (edge_img * scaling_factor).clip(0, 255).astype(np.uint8)
        edge_img = cv2.resize(edge_img, (512, 512))
        edge_img = Image.fromarray(edge_img)
        edge_img = self.hed(edge_img)
        self.controlnet_image = self.controlnet_pipeline.prepare_image(edge_img, 
                                                                       width=None, 
                                                                       height=None, 
                                                                       batch_size=2, 
                                                                       num_images_per_prompt=1, 
                                                                       device=torch.device("cuda"), 
                                                                       dtype=torch.float16, 
                                                                       do_classifier_free_guidance=True, 
                                                                       guess_mode=False)

        return lora_image, input_tensor
    
    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type=self.config["device"], dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type=self.config["device"], dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path):
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type=self.config["device"], dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps, desc="DDIM-Inversion")):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet_pre(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
            
        key = (f"noisy_latents_{t}", )
        self.self_attn_buffer[key] = latent
        return latent

    # Adopted from diffusers/utils.torch_utils.py
    def randn_tensor(
        self,
        shape: Union[Tuple, List],
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
        layout: Optional["torch.layout"] = None,
    ):
        rand_device = device
        batch_size = shape[0]

        layout = layout or torch.strided
        device = device or torch.device("cpu")

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
                for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

        return latents
    
    @torch.no_grad()
    def denoise_step(self, t, x=None):
        register_time(self.unet_pre, t.item())
        register_time(self.unet_post, t.item())
        
        if self.is_preprocess_stage:
            noise_pred = self.unet_pre(x, 
                                       t, 
                                       encoder_hidden_states=self.get_text_embeds("", "").chunk(2)[0]
                                       )["sample"]
            
            denoised_latent = self.scheduler.step(noise_pred, t, x)["prev_sample"]
        else:
            if t==self.scheduler.timesteps[0].item():
                key = (f"noisy_latents_{t}", )
                latent_model_input = self.self_attn_buffer.get(key, None)
                shape = torch.Size([1, 4, 64, 64])
                dtype = torch.float32
                device = config["device"]
                generator = [torch.Generator(device="cuda").manual_seed(145) for _ in range(1)] 

                style_latent = self.randn_tensor(shape=shape, generator=generator, device=device, dtype=dtype)
                content_latent = latent_model_input

                latent_model_input = adaptive_instance_normalization(content_latent, style_latent.clone())
                latent_model_input = torch.cat((style_latent, style_latent, latent_model_input, latent_model_input), dim=0)
            else:
                latent_model_input = torch.cat([x[0].unsqueeze(0)]*2 + [x[1].unsqueeze(0)]*2) 

            text_embed_input = self.get_text_embeds(config["prompt"], config["negative_prompt"])
            text_embed_input = torch.cat([text_embed_input]*2)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embed_input,
                    controlnet_cond=self.controlnet_image,
                    guess_mode = False,
                    return_dict=False
                )
                
            noise_pred = self.unet_post(latent_model_input, 
                                        t, 
                                        encoder_hidden_states=text_embed_input,
                                        down_block_additional_residuals=[
                                                sample.to(dtype=torch.float32) for sample in down_block_res_samples
                                            ],
                                        mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float32)
                                    )["sample"]
            

            controlnet_noise_pred_uncond, controlnet_noise_pred_cond, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(4)
            noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)
            controlnet_noise_pred = controlnet_noise_pred_uncond + self.config["guidance_scale"] * (controlnet_noise_pred_cond - controlnet_noise_pred_uncond)

            noise_pred = torch.cat([controlnet_noise_pred] + [noise_pred])
            scheduler_output = self.scheduler.step(noise_pred, t, torch.cat([latent_model_input[0].unsqueeze(0)] + [latent_model_input[2].unsqueeze(0)]))
            denoised_latent = scheduler_output["prev_sample"]

        return denoised_latent

    def init_pnp(self):
        self.qk_injection_timesteps = self.scheduler.timesteps
        register_LoRA_attention_control_efficient(self, self.qk_injection_timesteps, self.config["save_dir"])
        register_attention_control_efficient(self, self.qk_injection_timesteps, self.config["save_dir"])

    def run_pnp(self, inverted_x=None, original_dims=None):
        self.init_pnp()
        reconstructed_input = self.sample_loop(inverted_x)

        filename = os.path.basename(config["data_path"])

        if self.is_preprocess_stage:
            reconstructed_image_pil = T.ToPILImage()(reconstructed_input[0]).convert("RGB")

            self.lora_image = Image.fromarray(self.lora_image.astype(np.uint8))

            horizontal_gap = 10  

            total_width = self.lora_image.width + reconstructed_image_pil.width + horizontal_gap
            total_height = max(self.lora_image.height, reconstructed_image_pil.height) + 50  

            new_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))
            new_image.paste(self.lora_image, (0, 0))
            new_image.paste(reconstructed_image_pil, (self.lora_image.width + horizontal_gap, 0))

            draw = ImageDraw.Draw(new_image)
            font = ImageFont.load_default()

            draw.text((10, self.lora_image.height + 10), "scaled input (if below avg. 40)", font=font, fill="black")
            draw.text((self.lora_image.width + horizontal_gap + 10, reconstructed_image_pil.height + 10), "LoRA reconstructed", font=font, fill="black")

            new_image.save(f"{self.config['save_dir']}/LoRA_Reconstructed/{filename}")
        else:
            output_image_pil = T.ToPILImage()(reconstructed_input[1])
            output_image_pil.save(f"{self.config['save_dir']}/{filename}")

            output_image_pil = output_image_pil.resize(original_dims)
            resized_dir = os.path.join(self.config["save_dir"], "Resized")
            output_image_pil.save(f"{resized_dir}/{filename}")

    def sample_loop(self, x):
        with torch.autocast(device_type=self.config["device"], dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM-Sampling")):
                x = self.denoise_step(t, x=x)

            decoded_latent = self.decode_latents(x)

        return decoded_latent

    def extract_latents(self, num_steps, data_path, save_path):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds("", "")
        cond = cond[1].unsqueeze(0)

        self.lora_image, image = self.load_img(data_path)

        lora_path = "./lora_tmp"
        if os.path.exists(lora_path) and os.path.isdir(lora_path):
            shutil.rmtree(lora_path)

        train_lora(self.lora_image, "", self.model_key, "default", lora_path, self.config["lora_steps"], 0.0005, self.config["lora_batch_size"], 16, -1)
        self.unet_pre.load_attn_procs(lora_path)
        for param in self.unet_pre.parameters():
            param.requires_grad = False

        latent = self.encode_imgs(image)
        inverted_x = self.ddim_inversion(cond, latent, save_path)

        return inverted_x


def sort_files(filepath):
    numbers = re.findall(r"\d+", filepath)
    if numbers:
        return [int(num) for num in numbers]
    else:
        return [filepath]

def run(config):
    seed_everything(config["seed"])
    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["save_dir"], "LoRA_Reconstructed"), exist_ok=True)
    os.makedirs(os.path.join(config["save_dir"], "Resized"), exist_ok=True)

    img_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        img_paths.extend(glob.glob(os.path.join(config["img_dir_path"], ext)))
    img_paths = sorted(img_paths, key=sort_files)

    print(img_paths)

    model = Pipeline(config)

    for img_path in img_paths:
        filename = os.path.basename(img_path)
        model.config["data_path"] = img_path

        original_image = Image.open(img_path)
        original_dims = original_image.size

        print(f"\n> Processing {filename}...")
        start_time = time.time()

        model.unet_pre = deepcopy(model.pipe.unet) # Opted for deepcopy() due to time constraints; revisit if time permits.
        inverted_x = model.extract_latents(
                                        num_steps=config["n_timesteps"],
                                        data_path=config["data_path"],
                                        save_path=config["save_dir"],
                                        )

        model.is_preprocess_stage = True
        model.run_pnp(inverted_x, original_dims)

        model.is_preprocess_stage = False
        model.run_pnp(inverted_x, original_dims)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nElapsed Time: {elapsed_time:.2f} (s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_timesteps", type=int, default=50)
    parser.add_argument("--img_dir_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./output")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lora_batch_size", type=int, default=4, help="number of replication of the input image.")
    parser.add_argument("--lora_steps", type=int, default=80, help="numer of LoRA fine-tuning iterations with input image itself.")
    # Below prompts are not removed only because we did not remove below prompts settings at the time of submission of our paper. 
    # Therefore, below prompts could be replaced with emptry strings due to their negligble effects.
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--prompt", type=str, default="best quality, super-resolution, highest quality, highest texture, realistic, intricate details, realistic photo, photorealistic, sharp focus, hyperrealistic, film, professional, amazing, aesthetic, realistic stock photo, beautiful, luxury, sophisticated, cinematic, elegant and refined, magazine photoshoot, cinematic masterpiece, instagram post, 8k, 4K UHD quality")
    parser.add_argument("--negative_prompt", type=str, default="dark, gloomy, dim, generated image, noisy, artifacts, deformed, ugly, disfigured, blurry, poor facial details, disfigured, bad anatomy, fake, worst quality, low quality, lowres, unreal, monochrome, jpegartifacts, blurry, cropped, comic art, anime, fantasy art")
    opt = parser.parse_args()

    config = vars(opt)
    run(config)