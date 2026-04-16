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
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from .pipeline_stable_diffusion_xl_onnx_amd import StableDiffusionXLPipelineAMD
from diffusers.utils import load_image
from .utils import common


class StableDiffusionXLONNXPipelineAMDTrigger:
    def __init__(
        self,
        model_id: str = None,
        custom_op_path: str = None,
        model_path: str = None,
        enable_compile=False,
        gpu=False,
        enable_profile=False,
        profiling_rounds=4,
        control_image_path: str = None,
        revision: str = None,
    ):
        self.model_id = model_id
        self.enable_profile = enable_profile
        self.profiling_rounds = profiling_rounds
        self.control_image_path = control_image_path
        self.pipeline_metrics = {}
        self.mem_dict = {
            "unet": 0,
            "vae_decoder": 0,
            "vae_encoder": 0,
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
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            os.path.join(model_path, "tokenizer")
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            os.path.join(model_path, "tokenizer_2")
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(model_path, "text_encoder")
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            os.path.join(model_path, "text_encoder_2")
        )
        scheduler_name = json.load(
            open(
                os.path.join(model_path, "scheduler", "scheduler_config.json"), "r"
            )
        ).get("_class_name", None)
        scheduler_cls = getattr(
            importlib.import_module("diffusers.schedulers"), scheduler_name
        )
        self.scheduler = scheduler_cls.from_pretrained(
            os.path.join(model_path, "scheduler")
        )
        t0 = time.perf_counter()
        self.t_npu = 0
        t0_unet = time.perf_counter()
        start_mem = common.measure_mem()
        if not gpu and os.environ.get("DISABLE_UNET_DD", "0") == "0":
            print("---Loading ONNX Unet for DD")
            t0_npu_start = time.perf_counter()
            self.unet = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="unet",
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                # CHANGE MODIFIED: Dynamic provider selection instead of hardcoded ["DmlExecutionProvider"]
                providers=self._get_execution_providers(),
            )
            self.t_npu += time.perf_counter() - t0_npu_start
        else:
            print("---Loading Original ONNX Unet")
            self.unet = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="unet",
                model_file="unet.onnx",
                # CHANGE MODIFIED: Dynamic provider selection instead of hardcoded ["DmlExecutionProvider"]
                providers=self._get_execution_providers(),                
            )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["unet"] = mem_change
        Logger.debug(f"Model unet loading time = {time.perf_counter() - t0_unet}s")
        t0_vae = time.perf_counter()
        start_mem = common.measure_mem()
        if not gpu and os.environ.get("DISABLE_VAE_DD", "0") == "0":
            print("---Loading ONNX VAE Decoder for DD")
            t0_npu_start = time.perf_counter()
            self.vae_decoder = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="vae_decoder",
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                # CHANGE MODIFIED: Dynamic provider selection instead of hardcoded ["DmlExecutionProvider"]
                providers=self._get_execution_providers(),
            )
            self.t_npu += time.perf_counter() - t0_npu_start
        else:
            print("---Loading Original ONNX VAE Decoder")
            self.vae_decoder = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="vae_decoder",
                model_file="vae_decoder.onnx",
                # CHANGE MODIFIED: Dynamic provider selection instead of hardcoded ["DmlExecutionProvider"]
                providers=self._get_execution_providers(),
            )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["vae_decoder"] = mem_change
        Logger.debug(f"Model VAE loading time = {time.perf_counter() - t0_vae}s")

        if control_image_path is not None:
            start_mem = common.measure_mem()
            t0_vae = time.perf_counter()
            if not gpu and os.environ.get("DISABLE_VAE_DD", "0") == "0":
                print("---Loading ONNX VAE Encoder for DD")
                t0_npu_start = time.perf_counter()
                self.vae_encoder = common.load_model_with_session(
                    MODEL_PATH=model_path,
                    model_type="vae_encoder",
                    model_file="replaced.onnx",
                    custom_op_path=custom_op_path,
                    enable_dd_fusion_compile=enable_compile,
                    providers=self._get_execution_providers(),
                )
                self.t_npu += time.perf_counter() - t0_npu_start
            else:
                print("---Loading Original ONNX VAE Encoder")
                self.vae_encoder = common.load_model_with_session(
                    MODEL_PATH=model_path,
                    model_type="vae_encoder",
                    model_file="vae_encoder.onnx",
                    # CHANGE MODIFIED: Dynamic provider selection instead of hardcoded ["DmlExecutionProvider"]
                    providers=self._get_execution_providers(),
                )    
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["vae_encoder"] = mem_change
            Logger.debug(f"Model VAE encoder loading time = {time.perf_counter() - t0_vae}s")
        else:
            # The vae_encoder is not used and set to None
            Logger.info("Running in text2image mode without vae_encoder")
            self.vae_encoder = None
            self.mem_dict["vae_encoder"] = 0
            Logger.debug(f"VAE Encoder Mem: {mem_change}MB")

        for model, mem in self.mem_dict.items():
            Logger.debug(f"==> {model}: {mem:.2f}MB")
        mem_sum = sum(self.mem_dict.values())
        Logger.debug(
            f"Total NPU memory usage: {mem_sum:.2f}MB ({(mem_sum / 1024):.2f}GB)"
        )
        self.load_time_dict = {
            "all_npu_models": self.t_npu,
            "all_models": time.perf_counter() - t0,
        }

        if "scaling_factor" not in self.vae_decoder.config:
            self.vae_decoder.config["scaling_factor"] = (
                0.18215  # default value in AutoencoderKL
            )
        Logger.info("All NPU models loading time = " + str(self.t_npu))
        Logger.info("All Models loading time = " + str(time.perf_counter() - t0))
        Logger.debug(f"Current memory usage: {common.measure_mem()} MB")

    # CHANGE ADDED: Dynamic execution provider selection for --force-cpu functionality
    # Returns CPUExecutionProvider only when --force-cpu is used (ORT_DISABLE_GPU=1)
    def _get_execution_providers(self):
        """Get execution providers based on environment variables."""
        if os.environ.get("ORT_DISABLE_GPU") == "1":
            return ["CPUExecutionProvider"]
        else:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    def run(
        self,
        height=512,
        width=512,
        prompt="",
        n_prompt="",
        num_inference_steps=20,
        num_images_per_prompt=1,
        guidance_scale=7.5,
        seed=None,
        control_image_path = None,
        strength = 0.3,
    ):
        pipe = StableDiffusionXLPipelineAMD(
            scheduler=self.scheduler,
            vae_decoder=self.vae_decoder,
            vae_encoder=self.vae_encoder,
            unet=self.unet,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
        )
        if control_image_path is not None:
            control_image = load_image(control_image_path)
            # strength controls how control image influences result by timesteps and num_inference_steps
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            num_inference_steps_real = num_inference_steps - t_start
        else :
            control_image = None
            num_inference_steps_real = num_inference_steps
        common.print_config(
            {
                "height": height,
                "width": width,
                "prompt": prompt,
                "n_prompt": n_prompt,
                "num_inference_steps": num_inference_steps_real,
                "num_images_per_prompt": num_images_per_prompt,
                "guidance_scale": guidance_scale,
                "control_image_path": control_image_path,
                "strength": strength,
                "seed": seed,
            }
        )

        time_record = []

        if not self.enable_profile:
            start = time.perf_counter()
            images = pipe(
                prompt=prompt,
                n_prompt=n_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                control_image=control_image,
                strength = strength,
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
                n_prompt=n_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                control_image=control_image,
                strength = strength,
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
                    n_prompt=n_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale=guidance_scale,
                    control_image=control_image,
                    strength = strength,
                    generator=(
                        torch.Generator().manual_seed(seed)
                        if seed
                        else torch.Generator()
                    ),
                ).images

                Logger.debug(f"Current Mem: {common.measure_mem()}MB")
                # Emit per-round completion marker for orchestrator progress parsing
                print(f"__ROUND_COMPLETE__ {round_idx+1}/{self.profiling_rounds}", flush=True)
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
