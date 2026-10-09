#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import time
import json
import os
import torch
import importlib
import logging as Logger
from typing import List
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast
from diffusers.utils import load_image
from diffusers import FlowMatchEulerDiscreteScheduler
from src.pipeline import OnnxStableDiffusion3ControlNetPipelineAMD
from src.utils import common
from src.utils.model_loader import ModelLoadConfig, ModelType, load_model
from src.trigger.base_trigger import BaseOnnxPipelineTrigger


class StableDiffusion3PipelineTrigger(BaseOnnxPipelineTrigger):
    # pr-231: Accept args namespace directly — eliminates runner getter boilerplate
    def __init__(self, args):
        tea_caching_coef = getattr(args, "tea_caching_coef", None)
        if tea_caching_coef is None:
            tea_caching_coef = [1.0, 0.0]
        super().__init__(args)
        self.is_dynamic = args.dynamic_shape
        self.mem_dict = {
            "t5": 0,
            "vae_encoder": 0,
            "controlnet": 0,
            "mmdit": 0,
            "vae_decoder": 0,
            "tea_caching": 0,
        }
        self.enable_tea_caching = args.tea_caching
        self.tea_caching_delta = args.tea_caching_delta
        self.tea_caching_coef = tea_caching_coef

        root_model_path = args.model_path
        custom_op_path = args.custom_op_path
        enable_compile = args.enable_compile
        controlnet_str = args.controlnet
        width = args.width
        t5_sequence_len = args.t5_sequence_len
        is_dynamic = args.dynamic_shape
        tea_caching = args.tea_caching
        sub_model_path = getattr(args, "sub_model_path", "normal/")
        common_model_path = getattr(args, "common_model_path", "common/")
        self.root_model_path = root_model_path

        abs_sub_model_path = os.path.join(root_model_path, sub_model_path) if sub_model_path else root_model_path
        abs_common_model_path = os.path.join(root_model_path, common_model_path) if common_model_path else root_model_path
        self.t5_sequence_len = t5_sequence_len

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
                execution_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
            ),
            self.load_metrics_collector,
        )

        self.text_encoder_2 = load_model(
            ModelLoadConfig(
                model_path=os.path.join(abs_common_model_path, "text_encoder_2"),
                model_type=ModelType.ONNX_MODEL,
                execution_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
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

        npu_platform = getattr(args, "npu_platform", "unknown")
        if npu_platform == "mds":
            self.text_encoder_3 = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(abs_common_model_path, "text_encoder_3_int4"),
                    model_type=ModelType.ONNX_MODEL,
                    execution_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
                ),
                self.load_metrics_collector,
            )
        else:
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
                Logger.warning(
                    f"Failed to load text_encoder_3_gptq_v2, using text_encoder_3 instead: {e}"
                )
                self.text_encoder_3 = load_model(
                    ModelLoadConfig(
                        model_path=os.path.join(abs_common_model_path, "text_encoder_3"),
                        model_type=ModelType.ONNX_MODEL,
                        execution_providers=self.execution_providers,
                    ),
                    self.load_metrics_collector,
                )

        if controlnet_str is not None and controlnet_str.lower() != "none":
            controlnet_model_name = common.get_controlnet_model_name(controlnet_str)
            self.vae_encoder = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(abs_common_model_path, "vae_encoder"),
                    model_type=ModelType.ONNX_MODEL,
                    custom_op_path=custom_op_path,
                    enable_compile=enable_compile,
                    execution_providers=self.execution_providers,
                    disable_dd_env="DISABLE_VAE_ENCODER_DD",
                    width=width,
                    t5_sequence_len=t5_sequence_len,
                    is_dynamic=is_dynamic,
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
                    disable_dd_env="DISABLE_CONTROLNET_DD",
                    width=width,
                    t5_sequence_len=t5_sequence_len,
                    is_dynamic=is_dynamic,
                ),
                self.load_metrics_collector,
            )
        else:
            self.vae_encoder = None
            self.controlnet = None
        if npu_platform == "mds":
            self.transformer = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(abs_sub_model_path, "transformer"),
                    model_type=ModelType.ONNX_MODEL,
                    # Ryzen first: NPU takes fused ops; CPU last is fallback for unassigned nodes.
                    execution_providers=self.execution_providers,
                    disable_dd_env="DISABLE_MMDIT_DD",
                    width=width,
                    t5_sequence_len=t5_sequence_len,
                    custom_op_path=custom_op_path,
                    enable_compile=enable_compile,
                    is_dynamic=is_dynamic,
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
                    is_dynamic=is_dynamic,
                ),
                self.load_metrics_collector,
            )
        else:
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
                    is_dynamic=is_dynamic,
                ),
                self.load_metrics_collector,
            )
            self.transformer = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(abs_sub_model_path, "transformer"),
                    model_type=ModelType.ONNX_MODEL,
                    # Ryzen first: NPU takes fused ops; CPU last is fallback for unassigned nodes.
                    execution_providers=self.execution_providers,
                    disable_dd_env="DISABLE_MMDIT_DD",
                    width=width,
                    t5_sequence_len=t5_sequence_len,
                    custom_op_path=custom_op_path,
                    enable_compile=enable_compile,
                    is_dynamic=is_dynamic,
                ),
                self.load_metrics_collector,
            )

        if tea_caching:
            self.tea_caching_model = load_model(
                ModelLoadConfig(
                    model_path=os.path.join(abs_sub_model_path, "tea_caching"),
                    model_type=ModelType.ONNX_MODEL,
                    execution_providers=self.execution_providers,
                    disable_dd_env="DISABLE_TEA_CACHING_DD",
                    width=width,
                    t5_sequence_len=t5_sequence_len,
                    custom_op_path=custom_op_path,
                    enable_compile=enable_compile,
                    is_dynamic=is_dynamic,
                ),
                self.load_metrics_collector,
            )
        else:
            self.tea_caching_model = None

    def __enter__(self):
        Logger.info("Initializing resources for StableDiffusion3PipelineTrigger.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.text_encoder.model
        del self.text_encoder_2.model
        del self.transformer.model
        del self.vae_decoder.model
        del self.text_encoder_3
        if self.controlnet is not None:
            del self.controlnet.model
            del self.vae_encoder.model
        if self.tea_caching_model is not None:
            del self.tea_caching_model.model
        import gc
        gc.collect()
        Logger.debug("Models in StableDiffusion3PipelineTrigger are released")

    # pr-231: Accept args + prompt + shape — eliminates runner get_run_kwargs boilerplate
    def run(self, args, prompt, shape):
        height, width = shape.height, shape.width
        t5_sequence_len = shape.extras.get("t5_sequence_len", args.t5_sequence_len)
        n_prompt = args.n_prompt
        num_inference_steps = args.num_inference_steps
        num_images_per_prompt = args.num_images_per_prompt
        guidance_scale = args.guidance_scale
        seed = args.seed
        control_image_path = args.control_image_path
        controlnet_conditioning_scale = args.controlnet_conditioning_scale

        pipe = OnnxStableDiffusion3ControlNetPipelineAMD(
            vae_encoder=self.vae_encoder,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            tokenizer_3=self.tokenizer_3,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            text_encoder_3=self.text_encoder_3,
            transformer=self.transformer,
            vae_decoder=self.vae_decoder,
            controlnet=self.controlnet,
            tea_caching=self.tea_caching_model,
        )
        config_dict = {}
        config_dict["height"] = height
        config_dict["width"] = width
        config_dict["prompt"] = prompt
        config_dict["n_prompt"] = n_prompt
        config_dict["num_inference_steps"] = num_inference_steps
        config_dict["num_images_per_prompt"] = num_images_per_prompt
        config_dict["guidance_scale"] = guidance_scale
        config_dict["seed"] = seed
        if self.controlnet:
            config_dict["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            config_dict["control_img_url"] = control_image_path
            start_mem = common.measure_mem()
            control_image = load_image(control_image_path)
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["control_image"] = mem_change
            Logger.debug(f"Control image Mem: {mem_change}MB")
            Logger.debug(f"Load control image from : {control_image_path}")
        else:
            control_image = None

        common.print_config(config_dict)

        def run_fn():
            g = (
                torch.Generator().manual_seed(seed)
                if seed is not None
                else torch.Generator()
            )
            return pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                negative_prompt=n_prompt,
                control_image=control_image,
                num_images_per_prompt=num_images_per_prompt,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=g,
                max_sequence_length=t5_sequence_len,
                guidance_scale=guidance_scale,
                enable_tea_caching=self.enable_tea_caching,
                tea_caching_delta=self.tea_caching_delta,
                tea_caching_coef=self.tea_caching_coef,
            )

        return super().run(pipe, run_fn, height, width, enable_profile=self.enable_profile)
