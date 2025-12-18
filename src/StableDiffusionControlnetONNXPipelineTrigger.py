# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

import time
import json
import os
import torch
import importlib
import logging as Logger
import copy
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
from .pipeline_controlnet_onnx_amd import StableDiffusionControlNetONNXPipelineAMD
from .utils import common


class StableDiffusionControlnetONNXPipelineTrigger:
    def __init__(
        self,
        model_id: str = "",
        custom_op_path: str = "",
        model_path: str = "",
        enable_compile=False,
        gpu=False,
        enable_profile=False,
        profiling_rounds=4,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.enable_profile = enable_profile
        self.profiling_rounds = profiling_rounds
        self.gpu = gpu
        self.mem_dict = {
            "unet": 0,
            "controlnet": 0,
            "vae_decoder": 0,
        }

        self.t0_start = time.perf_counter()
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                os.path.join(model_path, "tokenizer")
            )
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_id, subfolder="tokenizer"
            )
        try:
            self.text_encoder = CLIPTextModel.from_pretrained(
                os.path.join(model_path, "text_encoder")
            )
        except:
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.model_id, subfolder="text_encoder"
            )
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

        # Load feature extractor
        try:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                os.path.join(model_path, "feature_extractor")
            )
        except:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                self.model_id, subfolder="feature_extractor"
            )

        self.t0_npu_start = time.perf_counter()
        self.t_npu = 0

        # Load UNet model
        t0_unet = time.perf_counter()
        start_mem = common.measure_mem()
        if not gpu and os.environ.get("DISABLE_UNET_DD", "0") == "0":
            Logger.info("---Loading ONNX Unet for DD")
            t0_npu_start = time.perf_counter()
            self.unet = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="unet",
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                providers=["DmlExecutionProvider"],
            )
            self.t_npu += time.perf_counter() - t0_npu_start
        else:
            Logger.info("---Loading Original ONNX Unet")
            self.unet = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="unet",
                model_file="unet.onnx",
                providers=["DmlExecutionProvider"],
            )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["unet"] = mem_change
        Logger.debug(f"Model unet loading time = {time.perf_counter() - t0_unet}s")

        # Load ControlNet model
        t0_controlnet = time.perf_counter()
        start_mem = common.measure_mem()
        if not gpu and os.environ.get("DISABLE_CONTROLNET_DD", "0") == "0":
            Logger.info("---Loading ONNX Controlnet for DD")
            t0_npu_start = time.perf_counter()
            self.controlnet = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="controlnet",
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                providers=["DmlExecutionProvider"],
            )
            self.t_npu += time.perf_counter() - t0_npu_start
        else:
            Logger.info("---Loading Original ONNX Controlnet")
            self.controlnet = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="controlnet",
                model_file="controlnet.onnx",
                providers=["DmlExecutionProvider"],
            )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["controlnet"] = mem_change
        Logger.debug(
            f"Model Controlnet loading time = {time.perf_counter() - t0_controlnet}s"
        )

        # Load VAE Decoder model
        t0_vae = time.perf_counter()
        start_mem = common.measure_mem()
        if not gpu and os.environ.get("DISABLE_VAE_DD", "0") == "0":
            Logger.info("---Loading ONNX VAE Decoder for DD")
            t0_npu_start = time.perf_counter()
            self.vae_decoder = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="vae_decoder",
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                providers=["DmlExecutionProvider"],
            )
            self.t_npu += time.perf_counter() - t0_npu_start
        else:
            Logger.info("---Loading Original ONNX VAE Decoder")
            self.vae_decoder = common.load_model_with_session(
                MODEL_PATH=model_path,
                model_type="vae_decoder",
                model_file="vae.onnx",
                providers=["DmlExecutionProvider"],
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

        self.t0_end = time.perf_counter()
        self.t0_npu = self.t_npu
        self.t0_all = self.t0_end - self.t0_start

        self.load_time_dict = {
            "all_npu_models": self.t0_npu,
            "all_models": self.t0_all,
        }

        Logger.info("All NPU models loading time = " + str(self.t0_npu))
        Logger.info("All Models loading time = " + str(self.t0_all))
        Logger.debug(f"Current memory usage: {common.measure_mem()} MB")

        # record pipeline metrics
        self.pipeline_metrics = {}

    def __enter__(self):
        Logger.info(
            "Initializing resources for StableDiffusionControlnetONNXPipelineTrigger."
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.unet.model
        del self.controlnet.model
        del self.vae_decoder.model
        # ADDED: Force garbage collection to release AMD Ryzen AI accelerator resources
        import gc
        gc.collect()
        Logger.debug(
            "Models in StableDiffusionControlnetONNXPipelineTrigger are released"
        )

    def run(
        self,
        height=512,
        width=512,
        prompt="",
        n_prompt="",
        num_inference_steps=20,
        num_images_per_prompt=1,
        control_image=None,
        controlnet_conditioning_scale=1.0,
        guidance_scale=7.5,
        seed=None,
    ):
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

        generator = (
            torch.Generator().manual_seed(seed)
            if seed is not None
            else torch.Generator()
        )

        if self.enable_profile:
            total_mem = 0
            # Warm-up
            t_start = time.perf_counter()
            output = pipe.run(
                prompt=prompt,
                negative_prompt=n_prompt,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                image=control_image,
                height=height,
                width=width,
                guess_mode=False,
                return_dict=False,
                device=torch.device("cpu"),
            )
            Logger.debug(f"Current Mem while warm up: {common.measure_mem()}MB")
            execution_time = time.perf_counter() - t_start
            perf_gpu_time_warm_up = copy.deepcopy(pipe.perf_time_gpu_model)
            perf_time_dict_warm_up = copy.deepcopy(pipe.perf_time_dict)
            pipe._clear_time_dict()

            t_start = time.perf_counter()
            for round_idx in range(self.profiling_rounds):
                total_mem += common.measure_mem()
                output = pipe.run(
                    prompt=prompt,
                    negative_prompt=n_prompt,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    image=control_image,
                    height=height,
                    width=width,
                    guess_mode=False,
                    return_dict=False,
                    device=torch.device("cpu"),
                )
                Logger.debug(f"Current Mem: {common.measure_mem()}MB")
                # ADDED: Emit a round-complete marker so the orchestrator can track progress
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

        else:
            start = time.perf_counter()
            output = pipe.run(
                prompt=prompt,
                negative_prompt=n_prompt,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                image=control_image,
                height=height,
                width=width,
                guess_mode=False,
                return_dict=False,
                device=torch.device("cpu"),
            )
            execution_time = time.perf_counter() - start
            Logger.info(f"Pipeline execution time = {execution_time:.6f}s")

        # Extract images from output
        if isinstance(output, tuple) and len(output) > 0:
            # output is (image, has_nsfw_concept) when return_dict=False
            images = output[0] if isinstance(output[0], list) else [output[0]]
        else:
            images = [output] if output is not None else []
        return images
