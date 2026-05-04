#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import time
import json
import os
import torch
import importlib
import logging as Logger
import copy

# setting path
from .pipeline_stable_diffusion_NitroE_onnx_amd import StableDiffusionNitroEONNXPipelineAMD

from .utils import common


class StableDiffusionNitroEONNXPipelineTrigger:
    def __init__(
        self,
        model_id: str = None,
        custom_op_path: str = None,
        model_path: str = None,
        enable_compile=False,
        gpu=False,
        enable_profile=False,
        profiling_rounds=4,
        revision: str = None,
    ):
        self.model_id = model_id
        self.enable_profile = enable_profile
        self.profiling_rounds = profiling_rounds
        self.pipeline_metrics = {}
        self.mem_dict = {
            "transformer": 0,
        }
        
        # Auto-download from Hugging Face if model_path not provided
        if model_path is None:
            Logger.debug("=" * 60)
            Logger.debug(f"model_path not provided, will download model from Hugging Face")
            Logger.debug(f"Model ID: {model_id}")
            if revision:
                Logger.debug(f"Revision/Branch: {revision}")
            Logger.debug("=" * 60)
            model_path = common.download_model_from_huggingface(model_id, revision=revision)
            Logger.debug("=" * 60)
            Logger.debug(f"Model ready: {model_path}")
            Logger.debug("Starting to load model components...")
            Logger.debug("=" * 60)
        
        self.model_path = model_path
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, "llamafp32_optimum")
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text_encoder = common.load_model_with_session(
            MODEL_PATH=model_path,
            model_type="llamafp32_optimum",
            model_file="model.onnx",
            # Dynamic provider selection instead of hardcoded providers
            providers=self._get_execution_providers(),
        )
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

        # VAE Decoder Offload to CPU
        self.vae_decoder = common.load_model_with_session(
            MODEL_PATH=model_path,
            model_type="vae_decoder",
            model_file="model.onnx",
            # CHANGE MODIFIED: Dynamic provider selection instead of hardcoded providers
            providers=self._get_execution_providers(),
        )

        t0 = time.perf_counter()
        self.t_npu = 0
        t0_transformer = time.perf_counter()
        start_mem = common.measure_mem()

        if not gpu and os.environ.get("DISABLE_TRANSFORMER_DD", "0") == "0":
            print("---Loading ONNX transformer for DD")
            t0_npu_start = time.perf_counter()
            self.unet = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="transformer",
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                # Dynamic provider selection instead of hardcoded providers
                providers=self._get_execution_providers(),
            )
            self.t_npu += time.perf_counter() - t0_npu_start
        else:
            print("---Loading Original ONNX transformer")
            self.transformer = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="transformer",
                model_file="model.onnx",
                # Dynamic provider selection instead of hardcoded providers
                providers=self._get_execution_providers(),
            )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["transformer"] = mem_change
        Logger.debug(f"Model transformer loading time = {time.perf_counter() - t0_transformer}s")


        for model, mem in self.mem_dict.items():
            Logger.debug(f"==> {model}: {mem:.2f}MB")
        mem_sum = sum(self.mem_dict.values())
        provider_name = "CPU" if os.environ.get("ORT_DISABLE_GPU", "0") == "1" else "NPU"
        Logger.debug(
            # f"Total NPU memory usage: {mem_sum:.2f}MB ({(mem_sum / 1024):.2f}GB)"
            f"Total {provider_name} memory usage: {mem_sum:.2f}MB ({(mem_sum / 1024):.2f}GB)"
        )
        self.load_time_dict = {
            "all_npu_models": self.t_npu,
            "all_models": time.perf_counter() - t0,
        }

        provider_name = "CPU" if os.environ.get("ORT_DISABLE_GPU", "0") == "1" else "NPU"
        Logger.info("All NPU models loading time = " + str(self.t_npu))
        Logger.info("All Models loading time = " + str(time.perf_counter() - t0))
        Logger.debug(f"Current memory usage: {common.measure_mem()} MB")

    # CHANGE ADDED: Dynamic execution provider selection for --force-cpu functionality
    # Returns CPUExecutionProvider only when --force-cpu is used (ORT_DISABLE_GPU=1)
    def _get_execution_providers(self):
        """Get execution providers based on environment variables."""
        if os.environ.get("ORT_DISABLE_GPU", "0") == "1":
            return ["CPUExecutionProvider"]
        return ["DmlExecutionProvider", "CPUExecutionProvider"]  # DML first, CPU fallback

    def run(
        self,
        height=512,
        width=512,
        prompt="",
        n_prompt="",
        num_inference_steps=20,
        num_images_per_prompt=1,
        guidance_scale=4.5,
        seed=None,
    ):
        # Build pipeline
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

        time_record = []
        if not self.enable_profile:
            start = time.perf_counter()
            images = pipe(
                prompt=prompt,
                negative_prompt=n_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                generator=(
                    torch.Generator().manual_seed(seed) if seed else torch.Generator()
                ),
            ).images
            end = time.perf_counter()

            execution_time = end - start
            time_record.append(execution_time)
            Logger.info("Pipeline execution time = " + str(execution_time))

        else:
            total_mem = 0
            t_start = time.perf_counter()
            images = pipe(
                prompt=prompt,
                negative_prompt=n_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                generator=(
                    torch.Generator().manual_seed(seed) if seed else torch.Generator()
                ),
            ).images
            Logger.debug(f"Current Mem while warm up: {common.measure_mem()}MB")
            execution_time = time.perf_counter() - t_start
            perf_gpu_time_warm_up = copy.deepcopy(pipe.perf_time_gpu_model)
            perf_time_dict_warm_up = copy.deepcopy(pipe.perf_time_dict)
            pipe._clear_time_dict()

            t_start = time.perf_counter()
            for round_idx in range(self.profiling_rounds):
                total_mem += common.measure_mem()
                _ = pipe(
                    prompt=prompt,
                    negative_prompt=n_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale=guidance_scale,
                    generator=(
                        torch.Generator().manual_seed(seed) if seed else torch.Generator()
                    ),
                ).images
                Logger.debug(f"Current Mem: {common.measure_mem()}MB")
                # ADDED: Emit round-complete marker for orchestrator progress tracking
                try:
                    print(f"__ROUND_COMPLETE__ {round_idx+1}/{self.profiling_rounds}", flush=True)
                except Exception:
                    pass
            t_total = time.perf_counter() - t_start

            key = "height_{}_width_{}".format(height, width)
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
            common.log_pipeline_metrics(self.pipeline_metrics[key])

        return images
