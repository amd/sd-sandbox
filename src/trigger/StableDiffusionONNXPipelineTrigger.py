# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.

import logging
import os

import torch

from src.pipeline import StableDiffusionONNXPipelineAMD
from src.utils import common
from src.utils.model_loader import ModelLoadConfig, ModelType, load_model
from src.trigger.base_trigger import BaseOnnxPipelineTrigger

logger = logging.getLogger(__name__)


class StableDiffusionONNXPipelineAMDTrigger(BaseOnnxPipelineTrigger):
    # pr-231: Accept args namespace directly — eliminates runner getter boilerplate
    def __init__(self, args):
        super().__init__(args)
        root_model_path = args.model_path
        custom_op_path = args.custom_op_path
        enable_compile = args.enable_compile

        self.tokenizer = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "tokenizer"),
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
            ),
            self.load_metrics_collector,
        )

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

        pipe_obj = StableDiffusionONNXPipelineAMD(
            vae=self.vae_decoder,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=None,
            feature_extractor=None,
            image_encoder=None,
            requires_safety_checker=False,
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

        def _run():
            g = (
                torch.Generator().manual_seed(seed)
                if seed
                else torch.Generator()
            )
            return pipe_obj(
                prompt=prompt,
                n_prompt=n_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                generator=g,
            )

        return super().run(pipe_obj, _run, height, width, enable_profile=self.enable_profile)
