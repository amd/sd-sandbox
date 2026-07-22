#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import os
import torch
import logging as Logger

from src.pipeline import OnnxFluxPipelineAMD
from src.utils import common
from src.utils.model_loader import ModelLoadConfig, ModelType, load_model
from src.trigger.base_trigger import BaseOnnxPipelineTrigger


class FluxPipelineTrigger(BaseOnnxPipelineTrigger):
    """
    Pipeline trigger for Flux text-to-image generation using ONNX Runtime on AMD hardware.

    Uses the refactored ModelLoadConfig-based loader.
    """

    # pr-231: Accept args namespace directly — eliminates runner getter boilerplate
    def __init__(self, args):
        super().__init__(args)
        self.is_dynamic = args.dynamic_shape
        root_model_path = args.model_path
        custom_op_path = args.custom_op_path
        enable_compile = args.enable_compile
        width = args.width
        max_sequence_length = getattr(args, "t5_sequence_len", 256)
        is_dynamic = args.dynamic_shape
        self.max_sequence_length = max_sequence_length
        self.root_model_path = root_model_path

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
                model_type=ModelType.T5_TOKENIZER,
            ),
            self.load_metrics_collector,
        )

        self.text_encoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "text_encoder"),
                model_type=ModelType.ONNX_MODEL,
                execution_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
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

        self.text_encoder_2 = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "text_encoder_3_gptq_v2"),
                model_type=ModelType.T5_NPU_MODEL,
                execution_providers=self.execution_providers,
            ),
            self.load_metrics_collector,
        )

        self.vae_decoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "vae_decoder"),
                model_type=ModelType.ONNX_MODEL,
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
                disable_dd_env="DISABLE_VAE_DD",
                width=width,
                t5_sequence_len=max_sequence_length,
                is_dynamic=is_dynamic,
            ),
            self.load_metrics_collector,
        )

        self.transformer = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "transformer"),
                model_type=ModelType.ONNX_MODEL,
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
                disable_dd_env="DISABLE_TRANSFORMER_DD",
                width=width,
                t5_sequence_len=max_sequence_length,
                is_dynamic=is_dynamic,
                low_memory_threshold_gb=25.0,
            ),
            self.load_metrics_collector,
        )

    def __enter__(self):
        Logger.info("Initializing resources for FluxPipelineTrigger.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.text_encoder.model
        del self.text_encoder_2
        del self.transformer.model
        del self.vae_decoder.model
        import gc
        gc.collect()
        Logger.debug("Models in FluxPipelineTrigger are released")

    # pr-231: Accept args + prompt + shape — eliminates runner get_run_kwargs boilerplate
    def run(self, args, prompt, shape):
        height, width = shape.height, shape.width
        prompt_2 = getattr(args, "prompt_2", None)
        num_inference_steps = args.num_inference_steps
        num_images_per_prompt = args.num_images_per_prompt
        guidance_scale = args.guidance_scale
        max_sequence_length = shape.extras.get("txt_seq_len", 256)
        seed = args.seed

        pipe = OnnxFluxPipelineAMD(
            scheduler=self.scheduler,
            vae_decoder=self.vae_decoder,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            transformer=self.transformer,
        )

        config_dict = {
            "height": height,
            "width": width,
            "prompt": prompt,
            "prompt_2": prompt_2,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": num_images_per_prompt,
            "guidance_scale": guidance_scale,
            "max_sequence_length": max_sequence_length,
            "seed": seed,
        }
        common.print_config(config_dict)

        def run_fn():
            g = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
            return pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=g,
            )

        return super().run(pipe, run_fn, height, width, enable_profile=self.enable_profile)
