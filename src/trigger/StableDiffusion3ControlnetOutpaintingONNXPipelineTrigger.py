# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import time
import json
import os
import torch
import numpy as np
import importlib
import logging as Logger
from typing import List

from diffusers.utils import load_image
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
)
from skimage.morphology import binary_dilation
import PIL

from src.pipeline import StableDiffusion3ControlNetOutpaintingPipeline
from src.utils import common
from src.utils.model_loader import ModelLoadConfig, ModelType, load_model
from src.utils.common import _get_provider_label
from src.trigger.base_trigger import BaseOnnxPipelineTrigger


class StableDiffusion3ControlnetOutpaintingONNXPipelineTrigger(BaseOnnxPipelineTrigger):
    # pr-231: Accept args namespace directly — eliminates runner getter boilerplate
    def __init__(self, args):
        tea_caching_coef = getattr(args, "tea_caching_coef", None)
        if tea_caching_coef is None:
            tea_caching_coef = [1.0, 0.0]
        super().__init__(args)
        self.enable_tea_caching = args.tea_caching
        self.tea_caching_delta = args.tea_caching_delta
        self.tea_caching_coef = tea_caching_coef

        root_model_path = args.model_path
        custom_op_path = args.custom_op_path
        enable_compile = args.enable_compile
        control_image_path = args.control_image_path
        control_mask_path = args.control_mask_path
        image_pads = args.image_pads
        controlnet_name = args.controlnet
        width = args.width
        t5_sequence_len = args.t5_sequence_len
        tea_caching = args.tea_caching
        sub_model_path = getattr(args, "sub_model_path", "normal/")
        common_model_path = getattr(args, "common_model_path", "common/")
      
        self.image_preprocess_time = 0
        image_preprocess_start = time.perf_counter()
        self.control_name = controlnet_name.lower()
        controlnet_model_name = common.get_controlnet_model_name(controlnet_name)
        self.control_image = load_image(control_image_path)
        if controlnet_model_name == "controlnet-outpainting":
            image_width, image_height = self.control_image.size
            control_mask = np.zeros(
                (
                    image_height + image_pads[0] + image_pads[1],
                    image_width + image_pads[2] + image_pads[3],
                ),
                dtype=np.uint8,
            )
            control_mask[:, image_width + image_pads[2] :] = 255
            control_mask[:, : image_pads[2]] = 255
            control_mask[: image_pads[0], :] = 255
            control_mask[image_height + image_pads[0] :, :] = 255
            control_image = np.zeros(
                (
                    image_height + image_pads[0] + image_pads[1],
                    image_width + image_pads[2] + image_pads[3],
                    3,
                ),
                dtype=np.uint8,
            )
            control_image[
                image_pads[0] : image_pads[0] + image_height,
                image_pads[2] : image_pads[2] + image_width,
            ] = np.array(self.control_image)
            self.control_image = PIL.Image.fromarray(control_image)
            self.control_mask = PIL.Image.fromarray(control_mask)

        elif controlnet_model_name == "controlnet-removal":
            self.control_mask = load_image(control_mask_path)
        elif controlnet_model_name == "controlnet-inpainting":
            self.control_mask = load_image(control_mask_path)
        else:
            raise ValueError(f"{controlnet_model_name} not supported.")
        self.control_mask = self.dynamic_dilated_mask(self.control_mask)
        self.origin_width, self.origin_height = self.control_mask.size
        self.image_preprocess_time = time.perf_counter() - image_preprocess_start

        Logger.info(f"Image preprocess time: {self.image_preprocess_time:.2f}s")

        abs_sub_model_path = os.path.join(root_model_path, sub_model_path) if sub_model_path else root_model_path
        abs_common_model_path = os.path.join(root_model_path, common_model_path) if common_model_path else root_model_path

        self.tokenizer = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_common_model_path, "tokenizer"),
                model_type=ModelType.CLIP_TOKENIZER,
            ),
            self.load_metrics_collector,
        )
        self.tokenizer_2 = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_common_model_path, "tokenizer_2"),
                model_type=ModelType.CLIP_TOKENIZER,
            ),
            self.load_metrics_collector,
        )
        self.tokenizer_3 = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_common_model_path, "tokenizer_3"),
                model_type=ModelType.T5_TOKENIZER,
            ),
            self.load_metrics_collector,
        )
        self.text_encoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_common_model_path, "text_encoder"),
                model_type=ModelType.ONNX_MODEL,
                model_file="model.onnx",
            ),
            self.load_metrics_collector,
        )
        self.text_encoder_2 = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_common_model_path, "text_encoder_2"),
                model_type=ModelType.ONNX_MODEL,
                model_file="model.onnx",
            ),
            self.load_metrics_collector,
        )
        self.scheduler = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_common_model_path, "scheduler"),
                model_type=ModelType.SCHEDULER,
            ),
            self.load_metrics_collector,
        )

        self.vae_decoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_common_model_path, "vae_decoder"),
                model_type=ModelType.ONNX_MODEL,
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
                disable_dd_env="DISABLE_VAE_DD",
                width=width,
                t5_sequence_len=t5_sequence_len,
            ),
            self.load_metrics_collector,
        )

        try:
            self.text_encoder_3 = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(abs_common_model_path, "text_encoder_3_gptq_v2"),
                    model_type=ModelType.T5_NPU_MODEL,
                    execution_providers=self.execution_providers,
                ),
                self.load_metrics_collector,
            )
        except Exception as e:
            Logger.warning(f"Failed to load text_encoder_3_gptq_v2, using text_encoder_3 instead: {e}")
            self.text_encoder_3 = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(abs_common_model_path, "text_encoder_3"),
                    model_type=ModelType.ONNX_MODEL,
                    execution_providers=self.execution_providers,
                ),
                self.load_metrics_collector,
            )

        self.vae_encoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_common_model_path, "vae_encoder"),
                model_type=ModelType.ONNX_MODEL,
                execution_providers=self.execution_providers,
                enable_compile=enable_compile,
                custom_op_path=custom_op_path,
                width=width,
                t5_sequence_len=t5_sequence_len,
            ),
            self.load_metrics_collector,
        )

        self.controlnet = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_sub_model_path, controlnet_model_name),
                model_type=ModelType.ONNX_MODEL,
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
                width=width,
                t5_sequence_len=t5_sequence_len,
                disable_dd_env="DISABLE_CONTROLNET_DD",
            ),
            self.load_metrics_collector,
        )

        self.transformer = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(abs_sub_model_path, "transformer"),
                    model_type=ModelType.ONNX_MODEL,
                    execution_providers=self.execution_providers,
                    custom_op_path=custom_op_path,
                    enable_compile=enable_compile,
                    width=width,
                    t5_sequence_len=t5_sequence_len,
                    disable_dd_env="DISABLE_MMDIT_DD",
                ),
                self.load_metrics_collector,
            )

        if tea_caching:
            self.tea_caching_model = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(abs_sub_model_path, "tea_caching"),
                    model_type=ModelType.ONNX_MODEL,
                    execution_providers=self.execution_providers,
                    custom_op_path=custom_op_path,
                    enable_compile=enable_compile,
                    width=width,
                    t5_sequence_len=t5_sequence_len,
                    disable_dd_env="DISABLE_TEA_CACHING_DD",
                ),
                self.load_metrics_collector,
            )
        else:
            self.tea_caching_model = None

        if "scaling_factor" not in self.vae_decoder.config:
            self.vae_decoder.config["scaling_factor"] = (
                0.18215  # default value in AutoencoderKL
            )
        
    def mask_dilation(self, mask, kernel_size=8):
        kernel = np.ones((kernel_size, kernel_size))
        mask = binary_dilation(mask, kernel)
        return mask

    def dynamic_dilated_mask(self, mask):
        ori_mask_size = mask.size
        mask_uper_bound = np.sqrt(ori_mask_size[0] * ori_mask_size[1]) * 0.04
        mask = np.array(mask)
        if len(mask.shape) == 3:
            mask = np.mean(mask, axis=2).astype(np.uint8)
        mask = np.array(mask)
        mask = np.where(mask > 0, 255, 0)
        mask_ratio = (np.sum(mask) // 255) / (mask.shape[0] * mask.shape[1])

        mask_ratio = np.sqrt(mask_ratio)
        dilated_ratio = max(int(mask_ratio * mask_uper_bound), 15)
        dilated_image = self.mask_dilation(mask, dilated_ratio)
        dilated_image = PIL.Image.fromarray(dilated_image)
        return dilated_image

    # pr-231: Accept args + prompt + shape — eliminates runner get_run_kwargs boilerplate
    def run(self, args, prompt, shape):
        height, width = shape.height, shape.width
        n_prompt = args.n_prompt
        num_inference_steps = args.num_inference_steps
        num_images_per_prompt = args.num_images_per_prompt
        guidance_scale = args.guidance_scale
        seed = args.seed
        controlnet_conditioning_scale = args.controlnet_conditioning_scale
        t5_sequence_len = shape.extras.get("t5_sequence_len", args.t5_sequence_len)

        # Resolve controlnet_use_pooled_projections from inpainting controlnet output count when not explicit
        controlnet_use_pooled_projections = None
        if "inpainting" in self.control_name.lower():
            num_outputs = len(self.controlnet.model.get_outputs())
            # https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting has 23 outputs
            # and controlnet_use_pooled_projections is False by default
            # Some customer inpainting controlnet has 8 outputs and need to set controlnet_use_pooled_projections to True
            controlnet_use_pooled_projections = num_outputs != 23
        else:
            controlnet_use_pooled_projections = False

        pipe = StableDiffusion3ControlNetOutpaintingPipeline(
            vae_encoder=self.vae_encoder,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            tokenizer_3=self.tokenizer_3,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            text_encoder_3=self.text_encoder_3,
            controlnet=self.controlnet,
            transformer=self.transformer,
            vae_decoder=self.vae_decoder,
            tea_caching=self.tea_caching_model,
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
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_image_path": args.control_image_path,
                "control_mask_path": args.control_mask_path,
                "seed": seed,
            }
        )

        def run_fn():
            g = torch.Generator().manual_seed(seed) if seed else torch.Generator()
            return pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                negative_prompt=n_prompt,
                control_name=self.control_name,
                control_image=self.control_image,
                control_mask=self.control_mask,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                controlnet_use_pooled_projections=controlnet_use_pooled_projections,
                generator=g,
                max_sequence_length=t5_sequence_len,
                guidance_scale=guidance_scale,
                enable_tea_caching=self.enable_tea_caching,
                tea_caching_delta=self.tea_caching_delta,
                tea_caching_coef=self.tea_caching_coef,
            )

        return super().run(pipe, run_fn, height, width, enable_profile=self.enable_profile)
