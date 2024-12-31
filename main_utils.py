import os
import random
import numpy as np
from PIL import Image
from einops import rearrange
from math import sqrt, floor, ceil

import torch
from torchvision import transforms as T

import xformers
from diffusers.models.attention_processor import  Attention, LoRAXFormersAttnProcessor


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

@torch.no_grad()
def register_time(unet, t):
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, "t", t)
            module = unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, "t", t)

def LoRAAttnOverrideForward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None):
        is_cross = encoder_hidden_states is not None
        residual = hidden_states

        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states) + scale * self.processor.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states) + scale * self.processor.to_k_lora(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states) + scale * self.processor.to_v_lora(encoder_hidden_states)
        
        query = self.head_to_batch_dim(query).contiguous()
        key = self.head_to_batch_dim(key).contiguous()
        value = self.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.processor.attention_op, scale=self.scale
        )

        hidden_states = self.batch_to_head_dim(hidden_states)

        if (hasattr(self, "place_in_unet")) and (self.place_in_unet=="up"):
            if is_cross:
                pass
            else:
                key = (self.place_in_unet, self.attn_name_value, self.t, self.res, self.block)
                self.our_pipeline.self_attn_buffer[key] = hidden_states.detach().clone()

        hidden_states = self.to_out[0](hidden_states) + scale * self.processor.to_out_lora(hidden_states)

        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states

def register_LoRA_attention_control_efficient(model, injection_schedule, save_dir):
    with torch.no_grad():

        res_dict = {0: [], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
        for res in res_dict:
            for block in res_dict[res]:
                for attn_name in ["attn1", "attn2"]:
                    module = getattr(model.unet_pre.up_blocks[res].attentions[block].transformer_blocks[0], attn_name)
                    module.forward = LoRAAttnOverrideForward.__get__(module, type(module))
                    setattr(module, "our_pipeline", model)
                    setattr(module, "injection_schedule", injection_schedule)
                    setattr(module, "res", res)
                    setattr(module, "block", block)
                    setattr(module, "attn_name_value", attn_name)
                    setattr(module, "save_dir", save_dir)
                    setattr(module, "place_in_unet", "up")

def AttnOverrideForward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, temb=None):
        is_cross = encoder_hidden_states is not None
        residual = hidden_states

        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = self.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.head_to_batch_dim(query).contiguous()
        key = self.head_to_batch_dim(key).contiguous()
        value = self.head_to_batch_dim(value).contiguous()
        
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.processor.attention_op, scale=self.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.batch_to_head_dim(hidden_states)

        if (hasattr(self, "place_in_unet")) and (self.place_in_unet=="up"):
            if is_cross:
                pass
            else:
                key = (self.place_in_unet, self.attn_name_value, self.t, self.res, self.block)
                hidden_states_inject = self.our_pipeline.self_attn_buffer.get(key, None)
                hidden_states[2] = hidden_states_inject.clone()
                hidden_states[3] = hidden_states_inject.clone()

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states

def register_attention_control_efficient(model, injection_schedule, save_dir):
    with torch.no_grad():
        res_dict = {0: [], 1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
        for res in res_dict:
            for block in res_dict[res]:
                for attn_name in ["attn1", "attn2"]:
                    module = getattr(model.unet_post.up_blocks[res].attentions[block].transformer_blocks[0], attn_name)
                    module.forward = AttnOverrideForward.__get__(module, type(module))
                    setattr(module, "our_pipeline", model)
                    setattr(module, "injection_schedule", injection_schedule) 
                    setattr(module, "res", res)
                    setattr(module, "block", block)
                    setattr(module, "attn_name_value", attn_name)
                    setattr(module, "save_dir", save_dir)
                    setattr(module, "place_in_unet", "up")