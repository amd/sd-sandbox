#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import time
import json
import os
import sys
import torch
import importlib
import logging as Logger
# Add parent directory to path for Kolors imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kolors.modeling_chatglm import ChatGLMModel
from src.kolors.tokenization_chatglm import ChatGLMTokenizer

from src.pipeline import StableDiffusionKolorsPipelineAMD
from diffusers.utils import load_image
from src.utils import common
from src.utils.model_loader import ModelLoadConfig, ModelType, load_model
from src.trigger.base_trigger import BaseOnnxPipelineTrigger


class StableDiffusionKolorsONNXPipelineAMDTrigger(BaseOnnxPipelineTrigger):
    # pr-231: Accept args namespace directly — eliminates runner getter boilerplate
    def __init__(self, args):
        super().__init__(args)
        self.controlnet_str = args.controlnet
        self.is_dynamic = args.dynamic_shape
        self._inpainting_origin_size = None
        if args.control_mask_path and args.control_image_path:
            self._inpainting_origin_size = load_image(args.control_image_path).size
        root_model_path = args.model_path
        custom_op_path = args.custom_op_path
        enable_compile = args.enable_compile
        controlnet_str = args.controlnet


        self.tokenizer = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "tokenizer"),
                model_type=ModelType.CHATGLM_TOKENIZER,
            ),
            self.load_metrics_collector,
        )
        self.text_encoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "text_encoder"),
                model_type=ModelType.CHATGLM_MODEL,
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

        # Load VAE Encoder
        if controlnet_str is not None and controlnet_str.lower() != "none":
            self.vae_encoder = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(root_model_path, "vae_encoder"),
                    model_type=ModelType.ONNX_MODEL,
                    disable_dd_env="DISABLE_VAE_DD",
                    custom_op_path=custom_op_path,
                    enable_compile=enable_compile,
                    execution_providers=self.execution_providers,
                    is_dynamic=self.is_dynamic,
                ),
                self.load_metrics_collector,
            )
        else:
            self.vae_encoder = None

        # Load ControlNet
        if controlnet_str is not None and controlnet_str.lower() not in ("none", "inpainting"):
            self.controlnet = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(root_model_path, f"controlnet-{controlnet_str}"),
                    model_type=ModelType.ONNX_MODEL,
                    disable_dd_env="DISABLE_CONTROLNET_DD",
                    custom_op_path=custom_op_path,
                    enable_compile=enable_compile,
                    execution_providers=self.execution_providers,
                    is_dynamic=self.is_dynamic,
                ),
                self.load_metrics_collector,
            )
        else:
            self.controlnet = None

        # Load UNet
        self.unet = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "unet"),
                model_type=ModelType.ONNX_MODEL,
                disable_dd_env="DISABLE_UNET_DD",
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
                is_dynamic=self.is_dynamic,
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
                is_dynamic=self.is_dynamic,
            ),
            self.load_metrics_collector,
        )

        if "scaling_factor" not in self.vae_decoder.config:
            self.vae_decoder.config["scaling_factor"] = (
                0.18215  # default value in AutoencoderKL
            )

    # pr-231: Accept args + prompt + shape — eliminates runner get_run_kwargs boilerplate
    def run(self, args, prompt, shape):
        height, width = shape.height, shape.width
        n_prompt = args.n_prompt
        num_inference_steps = args.num_inference_steps
        num_images_per_prompt = args.num_images_per_prompt
        guidance_scale = args.guidance_scale
        seed = args.seed
        strength = getattr(args, "strength", 0.8)
        control_mask_path = args.control_mask_path
        control_image_path = args.control_image_path
        controlnet_conditioning_scale = getattr(args, "controlnet_conditioning_scale", 1.0)
        control_guidance_start = getattr(args, "control_guidance_start", 0.0)
        control_guidance_end = getattr(args, "control_guidance_end", 1.0)

        pipe = StableDiffusionKolorsPipelineAMD(
            scheduler=self.scheduler,
            vae_decoder=self.vae_decoder,
            vae_encoder=self.vae_encoder,
            unet=self.unet,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            controlnet=self.controlnet if hasattr(self, 'controlnet') else None,
        )

        # Load images if provided (image and control_image merged: use control_image_path or image_path as single control_image)
        mask_image = load_image(control_mask_path) if control_mask_path is not None else None
        control_image = load_image(control_image_path) if control_image_path is not None else None
        strength = strength if control_image_path is not None else None

        common.print_config(
            {
                "height": height,
                "width": width,
                "prompt": prompt,
                "n_prompt": n_prompt,
                "num_inference_steps": num_inference_steps,
                "num_images_per_prompt": num_images_per_prompt,
                "guidance_scale": guidance_scale,
                "strength": strength,
                "control_mask_path": control_mask_path,
                "control_image_path": control_image_path,
                "seed": seed,
                "controlnet_conditioning_scale": controlnet_conditioning_scale if hasattr(self, 'controlnet') and self.controlnet else None,
                "control_guidance_start": control_guidance_start if hasattr(self, 'controlnet') and self.controlnet else None,
                "control_guidance_end": control_guidance_end if hasattr(self, 'controlnet') and self.controlnet else None,
            }
        )

        pipe_kwargs = {
            "height": height,
            "width": width,
            "prompt": prompt,
            "negative_prompt": n_prompt,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images_per_prompt,
            "guidance_scale": guidance_scale,
        }
        if hasattr(self, "controlnet") and self.controlnet is not None:
            pipe_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            pipe_kwargs["control_guidance_start"] = control_guidance_start
            pipe_kwargs["control_guidance_end"] = control_guidance_end
        if control_image is not None:
            pipe_kwargs["control_image"] = control_image
            pipe_kwargs["strength"] = strength
        if mask_image is not None:
            pipe_kwargs["mask_image"] = mask_image

        common.print_config(pipe_kwargs)

        def run_fn():
            kw = dict(pipe_kwargs)
            kw["generator"] = torch.Generator().manual_seed(seed) if seed else torch.Generator()
            return pipe(**kw)

        return super().run(pipe, run_fn, height, width, enable_profile=self.enable_profile)
