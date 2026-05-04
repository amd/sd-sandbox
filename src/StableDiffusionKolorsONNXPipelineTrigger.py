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
import copy

# Add parent directory to path for Kolors imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kolors.modeling_chatglm import ChatGLMModel
from src.kolors.tokenization_chatglm import ChatGLMTokenizer

from .pipeline_stable_diffusion_Kolors_onnx_amd import StableDiffusionKolorsPipelineAMD
from diffusers.utils import load_image
from .utils import common


class StableDiffusionKolorsONNXPipelineAMDTrigger:
    def __init__(
        self,
        model_id: str = None,
        custom_op_path: str = None,
        model_path: str = None,
        enable_compile=False,
        gpu=False,
        enable_profile=False,
        profiling_rounds=4,
        controlnet_str: str = None,
        is_dynamic: bool = False,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.enable_profile = enable_profile
        self.profiling_rounds = profiling_rounds
        self.controlnet_str = controlnet_str
        self.is_dynamic = is_dynamic
        self.pipeline_metrics = {}
        self.mem_dict = {
            "text_encoder": 0,
            "unet": 0,
            "vae_decoder": 0,
            "vae_encoder": 0,
            "controlnet": 0,
        }
        
        # Load ChatGLM tokenizer and text encoder for Kolors
        tokenizer_path = os.path.join(model_path, "tokenizer")
        text_encoder_path = os.path.join(model_path, "text_encoder")
        
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer path not found: {tokenizer_path}")
        if not os.path.exists(text_encoder_path):
            raise ValueError(f"Text encoder path not found: {text_encoder_path}")
        
        try:
            self.tokenizer = ChatGLMTokenizer.from_pretrained(tokenizer_path)
        except:
            self.tokenizer = ChatGLMTokenizer.from_pretrained(
                self.model_id, subfolder="tokenizer"
            )
        
        # self.text_encoder = common.LoadModel(
        #     model_path,
        #     "text_encoder",
        #     "text_encoder",
        #     providers=self._get_execution_providers(),
        # )
        
        try:
            self.text_encoder = ChatGLMModel.from_pretrained(
                text_encoder_path, dtype=torch.float16
            )
        except:
            self.text_encoder = ChatGLMModel.from_pretrained(
                self.model_id, subfolder="text_encoder", dtype=torch.float16
            )
        
        # Load scheduler
        try:
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
        except:
            raise ValueError("scheduler not found")
        
        t0 = time.perf_counter()
        self.t_npu = 0

        # Load VAE Encoder
        if controlnet_str is not None and controlnet_str.lower() != "none":
            vae_encoder_start_mem = common.measure_mem()
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
                    is_dynamic=self.is_dynamic,
                )
                self.t_npu += time.perf_counter() - t0_npu_start
            else:
                print("---Loading Original ONNX VAE Encoder")
                self.vae_encoder = common.load_model_with_session(
                    MODEL_PATH=model_path,
                    model_type="vae_encoder",
                    model_file="vae_encoder.onnx",
                    providers=self._get_execution_providers(),
                )
            mem_change = common.measure_mem() - vae_encoder_start_mem
            self.mem_dict["vae_encoder"] = mem_change
            Logger.debug(f"VAE Encoder Mem: {mem_change}MB")
        else:
            self.vae_encoder = None
            self.mem_dict["vae_encoder"] = 0
            Logger.debug(f"VAE Encoder Mem: 0MB")


        # Load ControlNet
        if controlnet_str is not None and controlnet_str.lower() not in ("none", "inpainting"):
            t0_controlnet_start = time.perf_counter()
            controlnet_start_mem = common.measure_mem()
            if not gpu and os.environ.get("DISABLE_CONTROLNET_DD", "0") == "0":
                print(f"---Loading ControlNet for DD")
                self.controlnet = common.load_model_with_session(
                    MODEL_PATH=model_path,
                    model_type=f"controlnet-{controlnet_str}",
                    model_file="replaced.onnx",
                    custom_op_path=custom_op_path,
                    enable_dd_fusion_compile=enable_compile,
                    providers=self._get_execution_providers(),
                    is_dynamic=self.is_dynamic,
                )
                self.t_npu += time.perf_counter() - t0_controlnet_start
            else:
                print(f"---Loading Original ControlNet")
                self.controlnet = common.load_model_with_session(
                    MODEL_PATH=model_path,
                    model_type=f"controlnet-{controlnet_str}",
                    model_file="optimized.onnx",
                    providers=self._get_execution_providers(),
                )
            mem_change = common.measure_mem() - controlnet_start_mem
            self.mem_dict["controlnet"] = mem_change
            Logger.debug(f"ControlNet Mem: {mem_change}MB")
            Logger.debug(f"ControlNet loading time = {time.perf_counter() - t0_controlnet_start}s")
        else:
            self.controlnet = None
            self.mem_dict["controlnet"] = 0
            Logger.debug(f"ControlNet Mem: 0MB")
            Logger.debug(f"ControlNet loading time = 0s")
      

        # Load UNet
        t0_unet = time.perf_counter()
        start_mem = common.measure_mem()
        if not gpu and os.environ.get("DISABLE_UNET_DD", "0") == "0":
            model_folder = "unet/t2i"
            if controlnet_str is not None and controlnet_str.lower() != "none":
                model_folder = "unet/i2i" if controlnet_str.lower() != "inpainting" else "unet/inpainting"
            print("---Loading ONNX Unet for DD")
            t0_npu_start = time.perf_counter()
            self.unet = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type=model_folder,
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                providers=self._get_execution_providers(),
                is_dynamic=self.is_dynamic,
            )
            self.t_npu += time.perf_counter() - t0_npu_start
        else:
            print("---Loading Original ONNX Unet")
            self.unet = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="unet",
                model_file="optimized.onnx",
                providers=self._get_execution_providers(),
            )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["unet"] = mem_change
        Logger.debug(f"Model unet loading time = {time.perf_counter() - t0_unet}s")

        # Load VAE Decoder
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
                providers=self._get_execution_providers(),
                is_dynamic=self.is_dynamic,
            )
            self.t_npu += time.perf_counter() - t0_npu_start
        else:
            print("---Loading Original ONNX VAE Decoder")
            self.vae_decoder = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="vae_decoder",
                model_file="vae_decoder.onnx",
                providers=self._get_execution_providers(),
            )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["vae_decoder"] = mem_change
        Logger.debug(f"Model VAE loading time = {time.perf_counter() - t0_vae}s")

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

    def _get_execution_providers(self):
        """Get execution providers based on environment variables."""
        if os.environ.get("ORT_DISABLE_GPU") == "1":
            return ["CPUExecutionProvider"]
        else:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    def run(
        self,
        height=1024,
        width=1024,
        prompt="",
        n_prompt="",
        num_inference_steps=50,
        num_images_per_prompt=1,
        guidance_scale=5.0,
        seed=None,
        strength=0.8,
        # Inpainting parameters
        control_mask_path=None,
        # ControlNet parameters
        control_image_path=None,
        controlnet_conditioning_scale=1.0,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
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

        time_record = []

        if not self.enable_profile:
            start = time.perf_counter()
            
            # Prepare pipe call arguments
            pipe_kwargs = {
                "prompt": prompt,
                "negative_prompt": n_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "num_images_per_prompt": num_images_per_prompt,
                "guidance_scale": guidance_scale,
                "generator": torch.Generator().manual_seed(seed) if seed else torch.Generator(),
            }
            
            # control_image: used for both ControlNet conditioning and img2img init (merged with former image param)
            if control_image is not None:
                pipe_kwargs["control_image"] = control_image
                # pipe_kwargs["image"] = control_image
                pipe_kwargs["strength"] = strength
            
            if mask_image is not None:
                pipe_kwargs["mask_image"] = mask_image
            
            if hasattr(self, 'controlnet') and self.controlnet is not None:
                pipe_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
                pipe_kwargs["control_guidance_start"] = control_guidance_start
                pipe_kwargs["control_guidance_end"] = control_guidance_end
            
            images = pipe(**pipe_kwargs).images
            end = time.perf_counter()

            execution_time = end - start
            time_record.append(execution_time)
            Logger.info("Pipeline execution time = " + str(execution_time))

        else:
            # Prepare pipe call arguments
            pipe_kwargs = {
                "prompt": prompt,
                "negative_prompt": n_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "num_images_per_prompt": num_images_per_prompt,
                "guidance_scale": guidance_scale,
                "generator": torch.Generator().manual_seed(seed) if seed else torch.Generator(),
            }
            
            # control_image: used for both ControlNet conditioning and img2img init (merged with former image param)
            if control_image is not None:
                pipe_kwargs["control_image"] = control_image
                # pipe_kwargs["image"] = control_image
                pipe_kwargs["strength"] = strength
            
            if mask_image is not None:
                pipe_kwargs["mask_image"] = mask_image
            
            if hasattr(self, 'controlnet') and self.controlnet is not None:
                pipe_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
                pipe_kwargs["control_guidance_start"] = control_guidance_start
                pipe_kwargs["control_guidance_end"] = control_guidance_end
            
            total_mem = 0
            t_start = time.perf_counter()
            images = pipe(**pipe_kwargs).images
            Logger.debug(f"Current Mem while warm up: {common.measure_mem()}MB")
            execution_time = time.perf_counter() - t_start
            perf_gpu_time_warm_up = copy.deepcopy(pipe.perf_time_gpu_model)
            perf_time_dict_warm_up = copy.deepcopy(pipe.perf_time_dict)
            pipe._clear_time_dict()

            t_start = time.perf_counter()
            for round_idx in range(self.profiling_rounds):
                total_mem += common.measure_mem()
                
                # Update generator for each round
                pipe_kwargs["generator"] = torch.Generator().manual_seed(seed) if seed else torch.Generator()
                _ = pipe(**pipe_kwargs).images

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
