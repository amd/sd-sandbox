#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.
#

import time
import json
import os
import torch
import importlib
import logging as Logger
import copy
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.utils import load_image
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
)
from .pipeline_stable_diffusion_3_controlnet_onnx_amd import (
    OnnxStableDiffusion3ControlNetPipelineAMD,
)
from .utils import common


class StableDiffusion3PipelineTrigger:
    def __init__(
        self,
        model_id: str = None,
        custom_op_path: str = None,
        root_path: str = None,
        model_path: str = None,
        sub_model_path: str = None,
        common_model_path: str = None,
        controlnet_str: str = None,
        control_image_path: str = None,
        enable_compile=False,
        enable_profile=False,
        profiling_rounds=4,
        width=1024,
        t5_sequence_len=83,
        is_dynamic=False,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.enable_profile = enable_profile
        self.profiling_rounds = profiling_rounds
        self.is_dynamic = is_dynamic
        self.mem_dict = {
            "t5": 0,
            "vae_encoder": 0,
            "controlnet": 0,
            "mmdit": 0,
            "vae_decoder": 0,
        }

        abs_sub_model_path = os.path.join(model_path, sub_model_path)
        abs_common_model_path = os.path.join(model_path, common_model_path)
        self.t5_sequence_len = t5_sequence_len
        self.t0_start = time.perf_counter()
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                os.path.join(abs_common_model_path, "tokenizer")
            )
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_id, subfolder="tokenizer"
            )
        try:
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                os.path.join(abs_common_model_path, "tokenizer_2")
            )
        except:
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                self.model_id, subfolder="tokenizer_2"
            )
        try:
            self.tokenizer_3 = T5TokenizerFast.from_pretrained(
                os.path.join(abs_common_model_path, "tokenizer_3")
            )
        except:
            self.tokenizer_3 = T5TokenizerFast.from_pretrained(
                self.model_id, subfolder="tokenizer_3"
            )

        self.text_encoder = common.LoadModel(
            abs_common_model_path, "text_encoder", "text_encoder"
        )
        self.text_encoder_2 = common.LoadModel(
            abs_common_model_path, "text_encoder_2", "text_encoder_2"
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(abs_common_model_path, "scheduler")
        )

        self.t0_npu_start = time.perf_counter()

        # Load T5 model
        start_mem = common.measure_mem()
        self.text_encoder_3 = common.LoadT5NPUTorchModel(
            root_path, abs_common_model_path, "text_encoder_3_gptq_v1"
        )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["t5"] = mem_change
        Logger.debug(f"T5 Mem: {mem_change}MB")
        self.control_image = None
        self.control_img_url = ""
        if controlnet_str.lower() != "none":
            controlnet_model_name, default_control_img_url, prompt, n_prompt = (
                common.get_normal_controlnet_model_name(controlnet_str)
            )
            # Use provided control_image_path if given, otherwise use default
            self.control_img_url = control_image_path if control_image_path else default_control_img_url

            # Load vae encoder model
            start_mem = common.measure_mem()
            self.vae_encoder = common.load_model_with_session(
                MODEL_PATH=abs_common_model_path,
                model_type="vae_encoder",
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                providers=["DmlExecutionProvider"],
                width=width,
                t5_sequence_len=t5_sequence_len,
                is_dynamic=is_dynamic,
            )
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["vae_encoder"] = mem_change
            Logger.debug(f"VAE Encoder Mem: {mem_change}MB")

            # Load controlnet model
            start_mem = common.measure_mem()
            self.controlnet = common.load_model_with_session(
                MODEL_PATH=abs_sub_model_path,
                model_type=controlnet_model_name,
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                providers=["DmlExecutionProvider"],
                width=width,
                t5_sequence_len=t5_sequence_len,
                is_dynamic=is_dynamic,
            )

            mem_change = common.measure_mem() - start_mem
            self.mem_dict["controlnet"] = mem_change
            Logger.debug(f"controlnet Mem: {mem_change}MB")
            self.control_image = load_image(self.control_img_url)
            # Load control image
            Logger.debug(f"Load control image from : {self.control_img_url}")
            model_type_mmdit = "transformer"
            model_type_decoder = "vae_decoder"
        else:
            # The vae_encoder and controlnet are not used and set to None
            Logger.info("Running in text2image mode without controlnet")
            self.vae_encoder = None
            self.mem_dict["vae_encoder"] = 0
            Logger.debug(f"VAE Encoder Mem: {mem_change}MB")

            self.controlnet = None
            self.mem_dict["controlnet"] = 0
            Logger.debug(f"controlnet Mem: {mem_change}MB")
            control_image = None
            # Keep the seed consistent with the previous pipeline without controlnet
            model_type_mmdit = "transformer"
            model_type_decoder = "vae_decoder"
        # Load vae decoder model
        start_mem = common.measure_mem()
        self.vae_decoder = common.load_model_with_session(
            MODEL_PATH=abs_common_model_path,
            model_type="vae_decoder",
            model_file="replaced.onnx",
            custom_op_path=custom_op_path,
            enable_dd_fusion_compile=enable_compile,
            providers=["DmlExecutionProvider"],
            width=width,
            t5_sequence_len=t5_sequence_len,
            is_dynamic=is_dynamic,
        )

        mem_change = common.measure_mem() - start_mem
        self.mem_dict["vae_decoder"] = mem_change

        # Load mmdit model
        start_mem = common.measure_mem()
        self.transformer = common.load_model_with_session(
            MODEL_PATH=abs_sub_model_path,
            model_type=model_type_mmdit,
            model_file="replaced.onnx",
            custom_op_path=custom_op_path,
            enable_dd_fusion_compile=enable_compile,
            providers=["DmlExecutionProvider"],
            width=width,
            t5_sequence_len=t5_sequence_len,
            is_dynamic=is_dynamic,
        )

        mem_change = common.measure_mem() - start_mem
        self.mem_dict["mmdit"] = mem_change
        Logger.debug(f"MMDiT Mem: {mem_change}MB")

        Logger.debug(f"VAE Decoder Mem: {mem_change}MB")
        self.t0_end = time.perf_counter()
        self.t0_npu = self.t0_end - self.t0_npu_start
        self.t0_all = self.t0_end - self.t0_start

        self.load_time_dict = {
            "all_npu_models": self.t0_npu,
            "all_models": self.t0_all,
        }
        Logger.info("All NPU models loading time = " + str(self.t0_npu))
        Logger.info("All Models loading time = " + str(self.t0_all))

        Logger.info(f"Current memory usage: {common.measure_mem()} MB")

        # record pipeline metrics
        self.pipeline_metrics = {}

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
        # ADDED: Force garbage collection to release AMD Ryzen AI accelerator resources
        import gc
        gc.collect()
        Logger.debug("Models in StableDiffusion3PipelineTrigger are released")

    def run(
        self,
        height=1024,
        width=1024,
        prompt="",
        n_prompt="",
        num_inference_steps=8,
        num_images_per_prompt=1,
        controlnet_conditioning_scale=0.5,
        guidance_scale=3.0,
        seed=None,
    ):
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
            config_dict["control_img_url"] = self.control_img_url
        common.print_config(config_dict)
        if self.enable_profile:
            # Warm-up
            t_start = time.perf_counter()
            output = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                negative_prompt=n_prompt,
                control_image=self.control_image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=(
                    torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
                ),
                max_sequence_length=self.t5_sequence_len,
                guidance_scale=guidance_scale,
            )
            Logger.debug(f"Current Mem while warm up: {common.measure_mem()}MB")
            execution_time = time.perf_counter() - t_start
            perf_gpu_time_warm_up = copy.deepcopy(pipe.perf_time_gpu_model)
            perf_time_dict_warm_up = copy.deepcopy(pipe.perf_time_dict)
            pipe._clear_time_dict()
            Logger.debug("------------------------------")

            # Profiling
            total_mem = 0
            t_start = time.perf_counter()
            for round_idx in range(self.profiling_rounds):
                total_mem += common.measure_mem()
                output = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    negative_prompt=n_prompt,
                    control_image=self.control_image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=(
                        torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
                    ),
                    max_sequence_length=self.t5_sequence_len,
                    guidance_scale=guidance_scale,
                )
                Logger.debug(f"Current Mem: {common.measure_mem()}MB")
                # ADDED: Emit round-complete marker for orchestrator progress tracking
                try:
                    print(f"__ROUND_COMPLETE__ {round_idx+1}/{self.profiling_rounds}", flush=True)
                except Exception:
                    pass
            t_total = time.perf_counter() - t_start
            images = output.images

            key = "height_{}_width_{}_t5_sequence_len_{}".format(height, width, self.t5_sequence_len)
            self.pipeline_metrics[key] = {
                "model_id": self.model_id,
                "execution_time": execution_time,
                "MODEL_PATH": self.model_path,
                "mem_dict": self.mem_dict,
                "perf_time_dict_warm_up": perf_time_dict_warm_up,
                "perf_gpu_time_warm_up": perf_gpu_time_warm_up,
                "perf_time_dict": pipe.perf_time_dict,
                "perf_gpu_time": pipe.perf_time_gpu_model,
                "t_total": t_total,
                "total_mem": total_mem,
                "profiling_rounds": self.profiling_rounds,
                "load_time_dict": self.load_time_dict,
            }
            if self.is_dynamic:
                Logger.info("============================================================")
                Logger.info(f"height: {height}, width: {width}, t5_sequence_len: {self.t5_sequence_len}")
                Logger.info("============================================================")
            common.log_pipeline_metrics(self.pipeline_metrics[key])

        else:
            start = time.perf_counter()
            output = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                negative_prompt=n_prompt,
                control_image=self.control_image,
                num_images_per_prompt=num_images_per_prompt,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=(
                    torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
                ),
                max_sequence_length=self.t5_sequence_len,
                guidance_scale=guidance_scale,
            )
            execution_time = time.perf_counter() - start
            Logger.info(f"Pipeline execution time = {execution_time:.6f}s")
            images = output.images

        return images
