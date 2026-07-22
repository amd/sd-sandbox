#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import logging
import time
import json
import os

import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.utils import load_image

from src.pipeline import StableDiffusionXLPipelineAMD
from src.utils import common
from src.utils.model_loader import ModelLoadConfig, ModelType, load_model
from src.trigger.base_trigger import BaseOnnxPipelineTrigger

logger = logging.getLogger(__name__)


class StableDiffusionXLONNXPipelineAMDTrigger(BaseOnnxPipelineTrigger):
    # pr-231: Accept args namespace directly — eliminates runner getter boilerplate
    def __init__(self, args):
        super().__init__(args)
        self.control_image_path = args.control_image_path
        self.is_dynamic = args.dynamic_shape
        root_model_path = args.model_path
        custom_op_path = args.custom_op_path
        enable_compile = args.enable_compile
        control_image_path = args.control_image_path
        is_dynamic = args.dynamic_shape

        self.tokenizer = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "tokenizer"),
                model_type=ModelType.CLIP_TOKENIZER,
            ),
            self.load_metrics_collector,
        )

        self.tokenizer_2 = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "tokenizer_2"),
                model_type=ModelType.CLIP_TOKENIZER,
            ),
            self.load_metrics_collector,
        )

        self.text_encoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "text_encoder"),
                model_type=ModelType.CLIP_TEXT_MODEL,
            ),
            self.load_metrics_collector,
        )

        self.text_encoder_2 = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "text_encoder_2"),
                model_type=ModelType.CLIP_TEXT_WITH_PROJECTION,
            ),
            self.load_metrics_collector,
        )

        self.scheduler = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "scheduler"),
                model_type=ModelType.SCHEDULER,
            ),
            self.load_metrics_collector,
        )

        self.unet = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "unet"),
                model_type=ModelType.ONNX_MODEL,
                disable_dd_env="DISABLE_UNET_DD",
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
                is_dynamic=is_dynamic,
            ),
            self.load_metrics_collector,
        )

        self.vae_decoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "vae_decoder"),
                model_type=ModelType.ONNX_MODEL,
                disable_dd_env="DISABLE_VAE_DD",
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
                is_dynamic=is_dynamic,
            ),
            self.load_metrics_collector,
        )

        if control_image_path is not None:
            self.vae_encoder = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(root_model_path, "vae_encoder"),
                    model_type=ModelType.ONNX_MODEL,
                    disable_dd_env="DISABLE_VAE_DD",
                    custom_op_path=custom_op_path,
                    enable_compile=enable_compile,
                    execution_providers=self.execution_providers,
                ),
                self.load_metrics_collector,
            )
        else:
            self.vae_encoder = None

        if "scaling_factor" not in self.vae_decoder.config:
            self.vae_decoder.config["scaling_factor"] = 0.18215

    # pr-231: Accept args + prompt + shape — eliminates runner get_run_kwargs boilerplate
    def run(self, args, prompt, shape):
        height, width = shape.height, shape.width
        n_prompt = args.n_prompt
        num_inference_steps = args.num_inference_steps
        num_images_per_prompt = args.num_images_per_prompt
        guidance_scale = args.guidance_scale
        seed = args.seed
        control_image_path = args.control_image_path
        strength = getattr(args, "strength", 0.3)

        pipe = StableDiffusionXLPipelineAMD(
            scheduler=self.scheduler,
            vae_decoder=self.vae_decoder,
            vae_encoder=self.vae_encoder,
            unet=self.unet,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
        )
        if control_image_path is not None:
            control_image = load_image(control_image_path)
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            num_inference_steps_real = num_inference_steps - t_start
        else:
            control_image = None
            num_inference_steps_real = num_inference_steps
        common.print_config(
            {
                "height": height,
                "width": width,
                "prompt": prompt,
                "n_prompt": n_prompt,
                "num_inference_steps": num_inference_steps_real,
                "num_images_per_prompt": num_images_per_prompt,
                "guidance_scale": guidance_scale,
                "control_image_path": control_image_path,
                "strength": strength,
                "seed": seed,
            }
        )

        def run_fn():
            g = torch.Generator().manual_seed(seed) if seed else torch.Generator()
            return pipe(
                prompt=prompt,
                n_prompt=n_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                control_image=control_image,
                strength=strength,
                generator=g,
            )

        return super().run(pipe, run_fn, height, width, enable_profile=self.enable_profile)
