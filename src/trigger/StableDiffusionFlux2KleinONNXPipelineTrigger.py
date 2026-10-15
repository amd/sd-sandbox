#
# Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
#

import logging as Logger
import os
from typing import List, Optional

import torch
from diffusers.utils import load_image
from safetensors import safe_open

from src.pipeline.pipeline_stable_diffusion_flux2_klein_onnx_amd import (
    StableDiffusionFlux2KleinPipelineAMD,
)
from src.trigger.base_trigger import BaseOnnxPipelineTrigger
from src.utils import common
from src.utils.model_loader import ModelLoadConfig, ModelType, detect_device, load_model


class StableDiffusionFlux2KleinONNXPipelineTrigger(BaseOnnxPipelineTrigger):
    """Trigger for FLUX.2-klein text-to-image / image-edit on AMD ONNX Runtime."""

    def __init__(self, args):
        super().__init__(args)
        root_model_path = args.model_path
        custom_op_path = args.custom_op_path
        enable_compile = args.enable_compile
        edit_image_path = getattr(args, "edit_image_path", None)
        is_dynamic = args.dynamic_shape
        model_id_lower = (args.model_id or "").lower()
        self.is_distilled = any(t in model_id_lower for t in ("4b", "distill"))
        self.is_dynamic = is_dynamic
        self.root_model_path = root_model_path

        self.tokenizer = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "tokenizer"),
                model_type=ModelType.QWEN2_TOKENIZER,
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

        gpu_text_encoder_cfg = ModelLoadConfig(
            model_path=os.path.join(root_model_path, "text_encoder"),
            model_type=ModelType.ONNX_MODEL,
            model_file="qwen3_text_encoder_prompt_embeds_matmulnbits.onnx",
            execution_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        npu_platform = getattr(args, "npu_platform", "unknown")
        if npu_platform == "mds":
            try:
                self.text_encoder = load_model(
                    ModelLoadConfig(
                        model_path=os.path.join(root_model_path, "text_encoder"),
                        model_type=ModelType.QWEN3_NPU_TEXT_ENCODER,
                        custom_op_path=custom_op_path,
                        t5_sequence_len=256,
                    ),
                    self.load_metrics_collector,
                )
            except Exception as e:
                Logger.warning(
                    "Failed to load NPU Qwen3 text encoder, falling back to GPU/DML: %s",
                    e,
                )
                self.text_encoder = load_model(
                    gpu_text_encoder_cfg, self.load_metrics_collector
                )
        else:
            self.text_encoder = load_model(
                gpu_text_encoder_cfg, self.load_metrics_collector
            )

        tensor_path = os.path.join(
            root_model_path, "vae_decoder", "bn.running_x.safetensors"
        )
        with safe_open(tensor_path, framework="pt", device="cpu") as f:
            self.bn_running_var = f.get_tensor("bn.running_var")
            self.bn_running_mean = f.get_tensor("bn.running_mean")

        self.vae_decoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "vae_decoder"),
                model_type=ModelType.ONNX_MODEL,
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
                disable_dd_env="DISABLE_VAE_DD",
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
                is_dynamic=is_dynamic,
                low_memory_threshold_gb=-1.0,
            ),
            self.load_metrics_collector,
        )

        self.vae_encoder = None
        if (
            # detect_device(self.execution_providers) != "NPU"
            edit_image_path is not None
            and len(edit_image_path) > 0
        ):
            self.vae_encoder = load_model(
            ModelLoadConfig(
                model_path=os.path.join(root_model_path, "vae_encoder"),
                model_type=ModelType.ONNX_MODEL,
                custom_op_path=custom_op_path,
                enable_compile=enable_compile,
                execution_providers=self.execution_providers,
                disable_dd_env="DISABLE_VAE_DD",
                is_dynamic=is_dynamic,
            ),
            self.load_metrics_collector,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.text_encoder.model
        del self.transformer.model
        del self.vae_decoder.model
        if self.vae_encoder is not None:
            del self.vae_encoder.model
        import gc

        gc.collect()

    def run(self, args, prompt, shape):
        height, width = shape.height, shape.width
        edit_image_path = getattr(args, "edit_image_path", None)
        num_inference_steps = args.num_inference_steps
        num_images_per_prompt = args.num_images_per_prompt
        guidance_scale = args.guidance_scale
        max_sequence_length = shape.extras.get("txt_seq_len", 256)
        seed = args.seed
        pipe = StableDiffusionFlux2KleinPipelineAMD(
            scheduler=self.scheduler,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            vae_encoder=self.vae_encoder,
            vae_decoder=self.vae_decoder,
            bn_running_mean=self.bn_running_mean,
            bn_running_var=self.bn_running_var,
            is_distilled=self.is_distilled,
        )

        common.print_config(
            {
                "height": height,
                "width": width,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "num_images_per_prompt": num_images_per_prompt,
                "guidance_scale": guidance_scale,
                "max_sequence_length": max_sequence_length,
                "seed": seed,
                "edit_image_path": edit_image_path,
            }
        )

        edit_images = None
        if edit_image_path:
            edit_images = [load_image(p) for p in edit_image_path]

        def run_fn():
            g = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
            return pipe(
                prompt=prompt,
                height=height,
                width=width,
                image=edit_images,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=g,
            )

        return super().run(
            pipe, run_fn, height, width, enable_profile=self.enable_profile
        )
