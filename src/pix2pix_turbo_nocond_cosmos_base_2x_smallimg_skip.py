# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import requests
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
from einops import rearrange, repeat
import time
import torch.nn.functional as F
import time

import argparse
import json
import os

import time

import torch
from megatron.core import parallel_state

from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
from cosmos_predict2.tokenizers.tokenizer import ResidualBlock, CausalConv3d
from cosmos_predict2.configs.base.config_text2image import (
    get_cosmos_predict2_text2image_pipeline,
)
from cosmos_predict2.conditioner import DataType
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline

from imaginaire.lazy_config import LazyDict, instantiate


config = get_cosmos_predict2_text2image_pipeline(model_size="0.6B", fast_tokenizer=False)

### MiniTrainDIT
config.dit_path = '/work/models/base/cosmos_ablation_2B_0502_t2i_309_1024res_multidataset_v4_synthetic_ratio_1_3.pth'
config.tokenizer["vae_pth"] = '/work/models/base/tokenizer.pth'
config.guardrail_config.enabled=False


from model import make_1step_sched_base as make_1step_sched 


def load_ckpt_from_state_dict(net_pix2pix, net_disc, optimizer, optimizer_disc, pretrained_path):
    sd = torch.load(pretrained_path, map_location="cuda")
        
    net_pix2pix.unet.load_state_dict(sd["state_dict_unet"])
    net_pix2pix.vae.load_state_dict(sd["state_dict_vae"])
    
    net_disc.load_state_dict({k[7:]: v for k, v in sd["net_disc"].items()})
    
    optimizer.load_state_dict(sd["optimizer"])
    optimizer_disc.load_state_dict(sd["optimizer_disc"])
    
    print()
    print('!!!! loading, load pretrained weight from', pretrained_path)
    print()
    return net_pix2pix, net_disc, optimizer, optimizer_disc

def save_ckpt(net_pix2pix, net_disc, optimizer, optimizer_disc, outf, train_full_unet=False, freeze_vae=False):
    sd = {}

    sd["state_dict_unet"] = net_pix2pix.unet.state_dict()
    sd["state_dict_vae"] = net_pix2pix.vae.state_dict()
        
    sd["net_disc"] = net_disc.state_dict()
    sd["optimizer"] = optimizer.state_dict()
    sd["optimizer_disc"] = optimizer_disc.state_dict()
    
    torch.save(sd, outf)


CACHE_T = 2

def my_vae_encoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to("cuda"), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    l_blocks = []
    # downsamples
    for layer in self.downsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
            
        l_blocks.append(x) 

    # middle
    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    # head
    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to("cuda"), cache_x], dim=2
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
            
    self.current_down_blocks = l_blocks
    return x


def my_vae_decoder_fwd(self, x, feat_cache=None, feat_idx=[0]):
    # conv1
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to("cuda"), cache_x], dim=2)
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

        
    skip_convs = [
        self.skip_conv_0, 
        self.skip_conv_1, 
        self.skip_conv_2, 
        self.skip_conv_3, 
        self.skip_conv_4, 
        self.skip_conv_5,
        self.skip_conv_6,
        self.skip_conv_7,
        self.skip_conv_8
    ]
    skip_acts = self.incoming_skip_acts
    
    
    # middle
    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

            
    enc_dec_mapping =  {0:3, 1:4, 2:5, 5:6, 6:7, 8:9, 10:10, 12:9, 14:10}
    
    for dec_idx, layer in enumerate(self.upsamples):
        
        if dec_idx in enc_dec_mapping:
            enc_indx = enc_dec_mapping[dec_idx]
            layer_index = list(enc_dec_mapping.keys()).index(dec_idx)
            
            skip_input = skip_acts[::-1][enc_indx]
            skip_in = skip_convs[layer_index](skip_input)  # 1x1 conv
            
            if dec_idx in [12, 14]:
                B, C, T, H, W = skip_in.shape
                SCALE = 2
                
                skip_in = skip_in.view((B, C * T, H, W))
                skip_in = F.interpolate(skip_in, 
                                        scale_factor=(SCALE, SCALE),
                                        mode='bilinear',
                                        align_corners=False)
                skip_in = skip_in.view((B, C, T, H * SCALE, W * SCALE))


            x = x + skip_in  # add skip

        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)
            

    # head
    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to("cuda"), cache_x], dim=2
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x



class Pix2Pix_Turbo(torch.nn.Module):
    
    def __init__(self, pretrained_path = None, freeze_vae_encoder=False, 
                 freeze_vae=False, train_full_unet=True, timestep=999, 
                 use_sched = False, vae_skip_connection = False):
        super().__init__()
                
        self.timesteps = torch.tensor([timestep]).to("cuda")
        self.timesteps_int =  timestep
        
        self.train_full_unet = train_full_unet
        self.freeze_vae = freeze_vae
        self.freeze_vae_encoder = freeze_vae_encoder
        self.vae_skip_connection = vae_skip_connection    
        
        self.use_sched = use_sched
        if self.use_sched:
            self.sched = None #make_1step_sched()
         

        self.initialize_cosmos_model()
        self.set_train()

        if pretrained_path is not None:
            print('loading from', pretrained_path)
            sd = torch.load(pretrained_path, map_location="cuda", weights_only=False)

            self.unet.load_state_dict(sd["state_dict_unet"], strict=False)
            self.vae.load_state_dict(sd["state_dict_vae"], strict=False)

            
        # print number of trainable parameters
        print("="*50)
        print(f"Number of trainable parameters in UNet: {sum(p.numel() for p in self.unet.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(f"Number of trainable parameters in VAE: {sum(p.numel() for p in self.vae.parameters() if p.requires_grad) / 1e6:.2f}M")
        print("="*50)

    def sample_batch_image(self, batch_size: int = 1):
        #h, w = 384, 640
        #h, w = 768, 1360
        h, w = 288, 512
        
        data_batch = {
            "dataset_name": "image_data",
            "images": torch.zeros(batch_size, 3, h, w).cuda(),
            "t5_text_embeddings": torch.zeros(batch_size, 512, 1024).cuda(),
            "fps": torch.ones((batch_size,)).cuda() * 24,
            "padding_mask": torch.zeros(batch_size, 1, h, w).cuda(),
        }
        return data_batch


    def initialize_cosmos_model(self):

        model = Text2ImagePipeline.from_config(
            config,
            dit_path=config.dit_path,
            use_text_encoder = False
        )
        
        
        ##### conditioning
        conditioner = model.conditioner
        data_batch = self.sample_batch_image()
        is_image_batch = True
        
        condition, uncondition = conditioner.get_condition_uncondition(data_batch)
        del condition
        
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        self.condition = uncondition

        self.unet = model#.net # MiniTrainDIT
        vae = model.tokenizer.model #model.tokenizer <projects.cosmos.diffusion.v2.tokenizers.wan2pt1.Wan2pt1VAEInterface object at 0x155401d12c20>
        
        self.img_mean = vae.img_mean
        self.img_std = vae.img_std
        self.video_mean = vae.video_mean
        self.video_std = vae.video_std
        self.scale = vae.scale
        
        
        
        ##### vae add skip
        #vae =  vae.model
        vae =  self.unet.tokenizer.model.model
        
        if self.vae_skip_connection:
            print('------adding skip connection')
            vae.decoder.skip_conv_0 = torch.nn.Conv3d(384, 384, kernel_size=1, stride=1, bias=False).cuda()  # <-- CHANGED
            vae.decoder.skip_conv_1 = torch.nn.Conv3d(384, 384, kernel_size=1, stride=1, bias=False).cuda()  # <-- CHANGED
            vae.decoder.skip_conv_2 = torch.nn.Conv3d(192, 384, kernel_size=1, stride=1, bias=False).cuda()  # <-- CHANGED
            vae.decoder.skip_conv_3 = torch.nn.Conv3d(192, 384, kernel_size=1, stride=1, bias=False).cuda()  # <-- CHANGED
            vae.decoder.skip_conv_4 = torch.nn.Conv3d(192, 384, kernel_size=1, stride=1, bias=False).cuda()  # <-- CHANGED
            vae.decoder.skip_conv_5 = torch.nn.Conv3d(96, 192, kernel_size=1, stride=1, bias=False).cuda()  # <-- CHANGED
            vae.decoder.skip_conv_6 = torch.nn.Conv3d(96, 192, kernel_size=1, stride=1, bias=False).cuda()  # <-- CHANGED
            vae.decoder.skip_conv_7 = torch.nn.Conv3d(96, 96, kernel_size=1, stride=1, bias=False).cuda()  # <-- CHANGED
            vae.decoder.skip_conv_8 = torch.nn.Conv3d(96, 96, kernel_size=1, stride=1, bias=False).cuda()  # <-- CHANGED

            torch.nn.init.constant_(vae.decoder.skip_conv_0.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_5.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_6.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_7.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_8.weight, 1e-5)

            vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)  # <-- CHANGED
            vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)  # <-- CHANGED 

        self.vae = vae
              
        print(' new self.vae', self.vae)
        print('-------------------------------------')
        
        


        print('-------------------------------------')
        print('SUCCESS in Initialize COSMOS MODEL')
        print(f"Number of parameters in UNet: {sum(p.numel() for p in self.unet.parameters() ) / 1e6:.2f}M")
        print(f"Number of parameters in VAE: {sum(p.numel() for p in self.vae.parameters()) / 1e6:.2f}M")
        print('-------------------------------------')

        self.unet.to("cuda")
        self.vae.to("cuda")
        
        
        
    def vae_encode(self, state: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(state, self.scale)
        num_frames = latents.shape[2]
        if num_frames == 1:
            return (latents - self.img_mean.type_as(latents)) / self.img_std.type_as(latents)
        else:
            return (latents - self.video_mean[:, :, :num_frames].type_as(latents)) / self.video_std[
                :, :, :num_frames
            ].type_as(latents)

    def vae_decode(self, latent: torch.Tensor) -> torch.Tensor:
        num_frames = latent.shape[2]
        if num_frames == 1:
            return self.vae.decode(
                (latent * self.img_std.type_as(latent)) + self.img_mean.type_as(latent), self.scale
            )
        else:
            return self.vae.decode(
                (latent * self.video_std[:, :, :num_frames].type_as(latent))
                + self.video_mean[:, :, :num_frames].type_as(latent), self.scale
            )
        
    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        
        if self.train_full_unet:
            self.unet.requires_grad_(True)
            #self.unet.net_ema.requires_grad_(False)
        else:
            raise ValueError('!! train partial Unet not implemented')
            
            
        self.vae.train()
        self.vae.requires_grad_(True)
        
        for name, param in self.vae.named_parameters():
            if "time_conv" in name:
                param.requires_grad = False
                print("set ", name, "grad to be false")
                
            
        if self.freeze_vae:
            self.vae.requires_grad_(False)
            self.vae.eval()

        if self.freeze_vae_encoder:
            self.vae.encoder.eval()
            self.vae.encoder.requires_grad_(False)

    def forward(self, x, timesteps=None):

        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"
        assert len(x.shape) == 4 
              
        unet_input = x[:, :, None, :, :]
        
        SCALE = 2
        B, C, T, H, W = unet_input.shape
        
        unet_input = unet_input.view((B, C * T, H, W))
        
        unet_input = F.interpolate(unet_input, 
                                   scale_factor=(1/SCALE, 1/SCALE),
                                   mode='bilinear',
                                   align_corners=False)
        
        unet_input = unet_input.view((B, C, T, H // SCALE, W // SCALE))


        unet_input = self.vae_encode(unet_input)
        
        
        
        B, C, T, H, W = unet_input.shape
        unet_input = unet_input.view((B, C * T, H, W))
        unet_input = F.interpolate(unet_input, 
                                   scale_factor=(SCALE, SCALE),
                                   mode='bilinear',
                                   align_corners=False)
        unet_input = unet_input.view((B, C, T, H * SCALE, W * SCALE))
        
        
        
        # ----- denoising
        sigma_B_T = self.timesteps.to(dtype=unet_input.dtype) / 1000
        
        model_pred = self.unet.denoise(xt_B_C_T_H_W = unet_input, 
                               sigma = sigma_B_T,
                               condition = self.condition ).x0
        
        z_denoised = model_pred
        
        if self.vae_skip_connection:
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks

        output_image = self.vae_decode(z_denoised)        
        
        output_image = output_image[:, :, 0]

        return output_image

    def save_model(self, outf, net_disc, optimizer, optimizer_disc):
        sd = {}
        
        sd["state_dict_unet"] = self.unet.state_dict()
        sd["state_dict_vae"] = self.vae.state_dict()
        
        sd["net_disc"] = net_disc.state_dict()
        sd["optimizer"] = optimizer.state_dict()
        sd["optimizer_disc"] = optimizer_disc.state_dict()
        
        torch.save(sd, outf)
