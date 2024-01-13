# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:21:44 2023

@author: Administrator
"""
import os

from omegaconf import OmegaConf

PROJECT="model3"

#pretrained_model_name_or_path="C:/Users/Administrator/Downloads/stable-diffusion-v1-5"
#linux
pretrained_model_name_or_path="/hy-tmp/models/runwayml/stable-diffusion-v1-5"
video_path="C:/software/ffmpeg-6.1-full_build/model3-pose/model3-1.mp4"
#rename Default train config
config_yaml = f"""
pretrained_model_path: {pretrained_model_name_or_path}
output_dir: outputs/{PROJECT}
train_data:
  video_path: {video_path}
  prompt: train prompt
  n_sample_frames: 16
  width: 512
  height: 512
  sample_start_idx: 0
  sample_frame_rate: 4
validation_data:
  prompts:
  - prompt
  video_length: 16
  width: 512
  height: 512
  num_inference_steps: 20
  guidance_scale: 12.5
  use_inv_latent: true
  num_inv_steps: 50
learning_rate: 3.0e-05
train_batch_size: 1
max_train_steps: 300
checkpointing_steps: 1000
validation_steps: 100
trainable_modules:
- attn1.to_q
- attn2.to_q
- attn_temp
seed: 33
mixed_precision: fp16
use_8bit_adam: false
gradient_checkpointing: true
enable_xformers_memory_efficient_attention: true
"""

#with open(f"configs/{PROJECT}.yaml", 'w') as f:
#  f.write(config_yaml)


CONFIG_NAME = f"configs/{PROJECT}.yaml" 
MOTION_MODULE_NAME="models/Motion_Module/mm_sd_v15.ckpt" 
train_video_path="model3/model3-1.mp4"
train_prompt="model3, model3-pose-00009"
video_length = 16 #@param {type:"number"}
width = 720 #@param {type:"number"}
height = 960 #@param {type:"number"}
learning_rate = 3e-5 #@param {type:"number"}
train_steps = 1 #@param {type:"number"}
validation_steps = 1 #@param {type:"number"}

config = {
  "pretrained_model_path": pretrained_model_name_or_path,
  "motion_module": MOTION_MODULE_NAME,
  "output_dir": "outputs/model3",
  "train_data": {
    "video_path": train_video_path,
    "prompt": train_prompt,
    "n_sample_frames": 16,
    "width": width,
    "height": height,
    "sample_start_idx": 0,
    "sample_frame_rate": 1,
  },
  "validation_data": {
    "prompts": [
      f"{train_prompt}",
    ],
    "video_length": video_length,
    "width": width,
    "height": height,
    "num_inference_steps": 20,
    "guidance_scale": 12.5,
    "use_inv_latent": True,
    "num_inv_steps": 50,
  },
  "learning_rate": learning_rate,
  "train_batch_size": 1,
  "max_train_steps": train_steps,
  "checkpointing_steps": 100,
  "validation_steps": validation_steps,
  "trainable_modules": [
    "to_q",
  ],
  "seed": 33,
  "mixed_precision": "fp16",
  "use_8bit_adam": False,
  "gradient_checkpointing": True,
  "enable_xformers_memory_efficient_attention": True,
}

OmegaConf.save(config, CONFIG_NAME)


print(f"------write the config successfully------{CONFIG_NAME}")