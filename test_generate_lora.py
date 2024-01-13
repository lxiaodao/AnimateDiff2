# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf
from einops import rearrange

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
#LoRAXFormersAttnProcessor
from diffusers.models.attention_processor import YangLoRAXFormersAttnProcessor,LoRAXFormersAttnProcessor,XFormersAttnProcessor
from animatediff.models.unet import UNet3DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from animatediff.pipelines.pipeline_animation import AnimationPipeline
import torchvision
import imageio

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
#check_min_version("0.21.0.dev0")

def main():
    # Final inference
    # Load previous pipeline
    inference_config = OmegaConf.load("configs/inference/inference.yaml")
    pretrained_model_name_or_path="/hy-tmp/models/runwayml/stable-diffusion-v1-5"

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=None)
    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_name_or_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            inference_config.unet_additional_kwargs),
    )

    pipeline = AnimationPipeline.from_pretrained(
        pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )
    pipeline = pipeline.to("cuda")

    # load attention processors
    pipeline.unet.load_attn_procs("outputs")

    # run inference
    validation_prompt="model3, model3-pose-00009, a woman in a black dress with a beaded collar and sleeves, standing in a white room with a white wall"
    generator = torch.Generator("cuda")
   
    generator = generator.manual_seed(42)
   
    for index in range(4):
       sample=pipeline(
            validation_prompt,
            num_inference_steps=30,
            generator=generator,
            temporal_context=9,
            video_length=9,
            fp16= False ,
        ).videos
       #image.save(f"outputs/{index}.png")
       save_videos_grid(sample, f"outputs/{index}.gif")

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)
  


if __name__ == "__main__":
    main()
