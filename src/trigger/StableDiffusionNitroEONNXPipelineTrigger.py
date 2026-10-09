#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import time
import os
import torch
import logging as Logger

from src.pipeline import StableDiffusionNitroEONNXPipelineAMD
from src.utils import common
from src.utils.model_loader import ModelLoadConfig, ModelType, load_model
from src.trigger.base_trigger import BaseOnnxPipelineTrigger


class StableDiffusionNitroEONNXPipelineTrigger(BaseOnnxPipelineTrigger):
    # pr-231: Accept args namespace directly — eliminates runner getter boilerplate
    def __init__(self, args):
        super().__init__(args)
        root_model_path = args.model_path
        custom_op_path = args.custom_op_path
        enable_compile = args.enable_compile

        self.tokenizer = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "llamafp32_optimum"),
                model_type=ModelType.AUTO_TOKENIZER,
            ),
            self.load_metrics_collector,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_encoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "llamafp32_optimum"),
                model_type=ModelType.ONNX_MODEL,
            ),
            self.load_metrics_collector,
        )

        from diffusers import FlowMatchEulerDiscreteScheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

        self.vae_decoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "vae_decoder"),
                model_type=ModelType.ONNX_MODEL,
            ),
            self.load_metrics_collector,
        )

        self.transformer = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "transformer"),
                model_type=ModelType.ONNX_MODEL,
                disable_dd_env="DISABLE_TRANSFORMER_DD",
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
            ),
            self.load_metrics_collector,
        )

    # pr-231: Accept args + prompt + shape — eliminates runner get_run_kwargs boilerplate
    def run(self, args, prompt, shape):
        height, width = shape.height, shape.width
        n_prompt = args.n_prompt
        num_inference_steps = args.num_inference_steps
        num_images_per_prompt = args.num_images_per_prompt
        guidance_scale = args.guidance_scale
        seed = args.seed

        pipe = StableDiffusionNitroEONNXPipelineAMD(
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            vae_decoder=self.vae_decoder,
        )

        common.print_config(
            {
                "height": height,
                "width": width,
                "prompt": prompt,
                "n_prompt": n_prompt,
                "num_inference_steps": num_inference_steps,
                "num_images_per_prompt": num_images_per_prompt,
                "guidance_scale": guidance_scale,
                "seed": seed,
            }
        )

        def run_fn():
            g = torch.Generator().manual_seed(seed) if seed else torch.Generator()
            return pipe(
                prompt=prompt,
                negative_prompt=n_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                generator=g,
            )

        return super().run(pipe, run_fn, height, width, enable_profile=self.enable_profile)
