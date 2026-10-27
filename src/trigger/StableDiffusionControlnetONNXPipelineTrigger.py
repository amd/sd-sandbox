# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import time
import json
import os
import torch
import importlib
import logging as Logger

import cv2
import numpy as np
from diffusers.utils import load_image
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor

from src.pipeline import StableDiffusionControlNetONNXPipelineAMD
from src.utils import common
from src.utils.model_loader import ModelLoadConfig, ModelType, load_model
from src.trigger.base_trigger import BaseOnnxPipelineTrigger


class StableDiffusionControlnetONNXPipelineTrigger(BaseOnnxPipelineTrigger):
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

        # Prefer the fp16 ONNX text encoder (model.onnx); fall back to the
        # fp32 PyTorch safetensors version when it is missing / fails to load.
        try:
            self.text_encoder = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(root_model_path, "text_encoder"),
                    model_type=ModelType.ONNX_TEXT_ENCODER,
                    execution_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
                ),
                self.load_metrics_collector,
            )
        except Exception as e:
            Logger.warning("ONNX text_encoder unavailable (%s); using PyTorch.", e)
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

        self.feature_extractor = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "feature_extractor"),
                model_type=ModelType.CLIP_IMAGE_PROCESSOR,
            ),
            self.load_metrics_collector,
        )
        
        self.unet = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "unet"),
                model_type=ModelType.ONNX_MODEL,
                execution_providers=self.execution_providers,
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                disable_dd_env="DISABLE_UNET_DD",
            ),
            self.load_metrics_collector,
        )
        
        self.controlnet = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "controlnet"),
                model_type=ModelType.ONNX_MODEL,
                execution_providers=["DmlExecutionProvider", "RyzenAILightExecutionProvider", "CPUExecutionProvider"],
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                disable_dd_env="DISABLE_CONTROLNET_DD",
            ),
            self.load_metrics_collector,
        )

        self.vae_decoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "vae_decoder"),
                model_type=ModelType.ONNX_MODEL,
                execution_providers=self.execution_providers,
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                disable_dd_env="DISABLE_VAE_DD",
            ),
            self.load_metrics_collector,
        )

    def __enter__(self):
        Logger.info(
            "Initializing resources for StableDiffusionControlnetONNXPipelineTrigger."
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.unet.model
        del self.controlnet.model
        del self.vae_decoder.model
        import gc

        gc.collect()
        Logger.debug(
            "Models in StableDiffusionControlnetONNXPipelineTrigger are released"
        )

    @staticmethod
    def _load_canny_image(image_path, low_threshold=100, high_threshold=200):
        """Moved from runner — canny preprocessing belongs with the controlnet trigger."""
        try:
            image = np.array(load_image(image_path))
            image = cv2.Canny(image, low_threshold, high_threshold)
            image = np.concatenate([image[:, :, None]] * 3, axis=2)
            return Image.fromarray(image)
        except Exception as exc:
            Logger.warning("Failed to load control image: %s", exc)
            return None

    # pr-231: Accept args + prompt + shape — eliminates runner get_run_kwargs boilerplate
    def run(self, args, prompt, shape):
        height, width = shape.height, shape.width
        n_prompt = args.n_prompt
        num_inference_steps = args.num_inference_steps
        num_images_per_prompt = args.num_images_per_prompt
        guidance_scale = args.guidance_scale
        seed = args.seed
        control_image = self._load_canny_image(args.control_image_path)
        controlnet_conditioning_scale = args.controlnet_conditioning_scale

        pipe = StableDiffusionControlNetONNXPipelineAMD(
            unet_onnx=self.unet,
            controlnet_onnx=self.controlnet,
            vae_decoder_onnx=self.vae_decoder,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
        )

        config_dict = {
            "height": height,
            "width": width,
            "prompt": prompt,
            "n_prompt": n_prompt,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images_per_prompt,
            "guidance_scale": guidance_scale,
            "control_image": control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "seed": seed,
        }
        common.print_config(config_dict)

        def run_fn():
            g = (
                torch.Generator().manual_seed(seed)
                if seed is not None
                else torch.Generator()
            )
            return pipe.run(
                prompt=prompt,
                negative_prompt=n_prompt,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                generator=g,
                image=control_image,
                height=height,
                width=width,
                guess_mode=False,
                return_dict=False,
                device=torch.device("cpu"),
            )

        return super().run(pipe, run_fn, height, width, enable_profile=self.enable_profile)