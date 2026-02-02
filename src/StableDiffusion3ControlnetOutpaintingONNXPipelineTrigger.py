#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.
#

import time
import json
import os
import torch
import numpy as np
import importlib
import logging as Logger
import copy

from diffusers.utils import load_image
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
)
from skimage.morphology import binary_dilation
import PIL

from .pipeline_stable_diffusion_3_controlnet_outpainting_onnx_amd import (
    StableDiffusion3ControlNetOutpaintingPipeline,
)
from .utils import common


class StableDiffusion3ControlnetOutpaintingONNXPipelineTrigger:
    def __init__(
        self,
        model_id: str = None,
        custom_op_path: str = None,
        root_path: str = None,
        model_path: str = None,
        sub_model_path: str = None,
        common_model_path: str = None,
        control_image_path="",
        control_mask_path="",
        image_pads=[0, 0, 0, 0],
        enable_compile=False,
        gpu=False,
        enable_profile=False,
        profiling_rounds=4,
        controlnet_name=None,
        width=1024,
        t5_sequence_len=83,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.enable_profile = enable_profile
        self.profiling_rounds = profiling_rounds
        self.mem_dict = {
            "t5": 0,
            "vae_encoder": 0,
            "controlnet": 0,
            "mmdit": 0,
            "vae_decoder": 0,
        }
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
        abs_sub_model_path = os.path.join(model_path, sub_model_path)
        abs_common_model_path = os.path.join(model_path, common_model_path)

        t0_start = time.perf_counter()
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
        try:
            scheduler_name = json.load(
                open(
                    os.path.join(
                        abs_common_model_path, "scheduler", "scheduler_config.json"
                    ),
                    "r",
                )
            ).get("_class_name", None)
            scheduler_cls = getattr(
                importlib.import_module("diffusers.schedulers"), scheduler_name
            )
            self.scheduler = scheduler_cls.from_pretrained(
                os.path.join(abs_common_model_path, "scheduler")
            )
        except:
            raise ValueError("scheuler not found")

        self.t_npu = 0
        if not gpu:
            t0_npu_start = time.perf_counter()
            start_mem = common.measure_mem()
            self.text_encoder_3 = common.LoadT5NPUTorchModel(
                root_path, abs_common_model_path, "text_encoder_3_gptq_v2"
            )
            mem_change = common.measure_mem() - start_mem

            self.mem_dict["t5"] = mem_change
            Logger.debug(f"T5 Mem: {mem_change}MB")
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
            )
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["controlnet"] = mem_change
            Logger.debug(f"controlnet Mem: {mem_change}MB")
            start_mem = common.measure_mem()
            self.transformer = common.load_model_with_session(
                MODEL_PATH=abs_sub_model_path,
                model_type="transformer",
                model_file="replaced.onnx",
                custom_op_path=custom_op_path,
                enable_dd_fusion_compile=enable_compile,
                providers=["DmlExecutionProvider"],
                width=width,
                t5_sequence_len=t5_sequence_len,
            )
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["mmdit"] = mem_change
            Logger.debug(f"mmdit Mem: {mem_change}MB")
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
            )
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["vae_decoder"] = mem_change
            t0_npu_end = time.perf_counter()
            self.t_npu = t0_npu_end - t0_npu_start
        else:
            # Load text_encoder_3
            start_mem = common.measure_mem()
            self.text_encoder_3 = common.LoadModel(
                abs_common_model_path, "text_encoder_3", "text_encoder_3"
            )
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["t5"] = mem_change
            Logger.debug(f"T5 Mem: {mem_change}MB")
            # Load vae encoder model
            start_mem = common.measure_mem()
            self.vae_encoder = common.LoadModel(
                abs_common_model_path, "vae_encoder", "vae_encoder"
            )
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["vae_encoder"] = mem_change
            # Load controlnet model
            start_mem = common.measure_mem()
            self.controlnet = common.LoadModel(
                abs_sub_model_path, controlnet_model_name, controlnet_model_name
            )
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["controlnet"] = mem_change
            # Load mmdit model
            start_mem = common.measure_mem()
            self.transformer = common.LoadModel(
                abs_sub_model_path, "transformer", "transformer"
            )
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["mmdit"] = mem_change
            # Load vae decoder model
            start_mem = common.measure_mem()
            self.vae_decoder = common.LoadModel(
                abs_common_model_path, "vae_decoder", "vae_decoder"
            )
            mem_change = common.measure_mem() - start_mem
            self.mem_dict["vae_decoder"] = mem_change

        for model, mem in self.mem_dict.items():
            Logger.debug(f"==> {model}: {mem:.2f}MB")
        mem_sum = sum(self.mem_dict.values())
        Logger.info(
            f"Total NPU memory usage: {mem_sum:.2f}MB ({(mem_sum / 1024):.2f}GB)"
        )
        self.load_time_dict = {
            "all_npu_models": self.t_npu,
            "all_models": time.perf_counter() - t0_start,
        }

        if "scaling_factor" not in self.vae_decoder.config:
            self.vae_decoder.config["scaling_factor"] = (
                0.18215  # default value in AutoencoderKL
            )
        Logger.info("All NPU models loading time = " + str(self.t_npu))
        Logger.info("All Models loading time = " + str(time.perf_counter() - t0_start))
        Logger.debug(f"Current memory usage: {common.measure_mem()} MB")

        # record pipeline metrics
        self.pipeline_metrics = {}

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

    def run(
        self,
        height=1024,
        width=1024,
        prompt="",
        n_prompt="",
        num_inference_steps=8,
        num_images_per_prompt=1,
        guidance_scale=7,
        control_image_path="",
        control_mask_path="",
        seed=None,
        controlnet_conditioning_scale=0.95,
        t5_sequence_len=83,
    ):
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
                "control_image_path": control_image_path,
                "control_mask_path": control_mask_path,
                "seed": seed,
            }
        )

        time_record = []
        if not self.enable_profile:
            start = time.perf_counter()
            images = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                negative_prompt=n_prompt,
                control_name=self.control_name,
                control_image=self.control_image,
                control_mask=self.control_mask,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=(
                    torch.Generator().manual_seed(seed) if seed else torch.Generator()
                ),
                max_sequence_length=t5_sequence_len,
                guidance_scale=guidance_scale,
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
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                negative_prompt=n_prompt,
                control_name=self.control_name,
                control_image=self.control_image,
                control_mask=self.control_mask,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=(
                    torch.Generator().manual_seed(seed) if seed else torch.Generator()
                ),
                max_sequence_length=t5_sequence_len,
                guidance_scale=guidance_scale,
            ).images
            Logger.debug(f"Current Mem while warm up: {common.measure_mem()}MB")
            execution_time = time.perf_counter() - t_start
            perf_gpu_time_warm_up = copy.deepcopy(pipe.perf_time_gpu_model)
            perf_time_dict_warm_up = copy.deepcopy(pipe.perf_time_dict)
            pipe._clear_time_dict()

            t_start = time.perf_counter()
            # Renamed _  to round_idx bc it's used in a print statement
            for round_idx in range(self.profiling_rounds):
                total_mem += common.measure_mem()
                _ = pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    negative_prompt=n_prompt,
                    control_name=self.control_name,
                    control_image=self.control_image,
                    control_mask=self.control_mask,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=(
                        torch.Generator().manual_seed(seed)
                        if seed
                        else torch.Generator()
                    ),
                    max_sequence_length=t5_sequence_len,
                    guidance_scale=guidance_scale,
                ).images

                Logger.debug(f"Current Mem: {common.measure_mem()}MB")
                # CHANGED: Added progress marker for orchestrator real-time tracking (sd_ref_design integration)
                print(f"__ROUND_COMPLETE__ {round_idx+1}/{self.profiling_rounds}", flush=True)
            t_total = time.perf_counter() - t_start

            key = "height_{}_width_{}_t5_sequence_len_{}".format(height, width, t5_sequence_len)
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

        return images, self.origin_width, self.origin_height
