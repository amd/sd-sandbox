#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import time
import json
import os
import torch
import logging as Logger
import copy
from transformers import CLIPTokenizer, T5TokenizerFast
from diffusers import FlowMatchEulerDiscreteScheduler
from .pipeline_flux_onnx_amd import OnnxFluxPipelineAMD
from .utils import common


class FluxPipelineTrigger:
    """
    Pipeline trigger for Flux text-to-image generation using ONNX Runtime on AMD hardware.
    
    This class manages model loading, initialization, and execution for the Flux pipeline.
    It supports performance profiling and various optimization features.
    
    Args:
        model_id: HuggingFace model identifier
        custom_op_path: Path to custom ONNX operators
        root_path: Root path of the project
        model_path: Path to the model directory
        sub_model_path: Path to model variant subdirectory
        common_model_path: Path to shared model components
        enable_compile: Enable compilation optimizations
        enable_profile: Enable performance profiling
        profiling_rounds: Number of profiling rounds
        width: Image width (default 1024)
        height: Image height (default 1024)
        max_sequence_length: Maximum T5 sequence length (default 512)
    """
    
    def __init__(
        self,
        model_id: str = None,
        custom_op_path: str = None,
        root_path: str = None,
        model_path: str = None,
        sub_model_path: str = None,
        common_model_path: str = None,
        enable_compile: bool = False,
        enable_profile: bool = False,
        profiling_rounds: int = 4,
        width: int = 1024,
        max_sequence_length: int = 256,
        is_dynamic: bool = False,
        revision: str = None,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.enable_profile = enable_profile
        self.profiling_rounds = profiling_rounds
        self.max_sequence_length = max_sequence_length
        self.is_dynamic = is_dynamic
        self.revision = revision
        self.mem_dict = {
            "text_encoder": 0,
            "text_encoder_2": 0,
            "transformer": 0,
            "vae_decoder": 0,
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
        
        abs_sub_model_path = model_path 
        abs_common_model_path = model_path
        
        self.t0_start = time.perf_counter()
        
        # Load tokenizers
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                os.path.join(abs_common_model_path, "tokenizer")
            )
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_id, subfolder="tokenizer"
            )
        
        try:
            self.tokenizer_2 = T5TokenizerFast.from_pretrained(
                os.path.join(abs_common_model_path, "tokenizer_2")
            )
        except:
            self.tokenizer_2 = T5TokenizerFast.from_pretrained(
                self.model_id, subfolder="tokenizer_2"
            )
        
        # Load text encoders
        Logger.info("Loading CLIP text encoder...")
        start_mem = common.measure_mem()
        self.text_encoder = common.LoadModel(
            abs_common_model_path,
            "text_encoder",
            "text_encoder",
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["text_encoder"] = mem_change
        Logger.debug(f"CLIP Text Encoder Mem: {mem_change}MB")
        
        # Load scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(abs_common_model_path, "scheduler")
        )
        
        self.t0_npu_start = time.perf_counter()
        
        # Load T5 text encoder (on NPU)
        Logger.info("Loading T5 text encoder...")
        start_mem = common.measure_mem()
        self.text_encoder_2 = common.LoadT5NPUTorchModel(
            root_path, abs_common_model_path, "text_encoder_3_gptq_v2"
        )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["text_encoder_2"] = mem_change
        Logger.debug(f"T5 Text Encoder Mem: {mem_change}MB")

        # Load VAE decoder
        Logger.info("Loading VAE decoder...")
        start_mem = common.measure_mem()
        self.vae_decoder = common.load_model_with_session(
            MODEL_PATH=abs_common_model_path,
            model_type="vae_decoder",
            model_file="replaced.onnx",
            custom_op_path=custom_op_path,
            enable_dd_fusion_compile=enable_compile,
            providers=["CPUExecutionProvider"],
            width=width,
            t5_sequence_len=max_sequence_length,
            is_dynamic=is_dynamic,
        )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["vae_decoder"] = mem_change
        Logger.debug(f"VAE Decoder Mem: {mem_change}MB")
        
        # Load Flux transformer (DiT model)
        Logger.info("Loading Flux transformer...")
        start_mem = common.measure_mem()
        self.transformer = common.load_model_with_session(
            MODEL_PATH=abs_sub_model_path,
            model_type="transformer",
            model_file="replaced.onnx",
            custom_op_path=custom_op_path,
            enable_dd_fusion_compile=enable_compile,
            providers=["CPUExecutionProvider"],
            width=width,
            t5_sequence_len=max_sequence_length,
            is_dynamic=is_dynamic,
            low_memory_threshold_gb=18.0,
        )
        mem_change = common.measure_mem() - start_mem
        self.mem_dict["transformer"] = mem_change
        Logger.debug(f"Transformer Mem: {mem_change}MB")
        
        
        self.t0_end = time.perf_counter()
        self.t0_npu = self.t0_end - self.t0_npu_start
        self.t0_all = self.t0_end - self.t0_start
        
        self.load_time_dict = {
            "all_npu_models": self.t0_npu,
            "all_models": self.t0_all,
        }
        
        Logger.info(f"All NPU models loading time = {self.t0_npu:.2f}s")
        Logger.info(f"All models loading time = {self.t0_all:.2f}s")
        Logger.info(f"Current memory usage: {common.measure_mem()} MB")
        
        # Record pipeline metrics
        self.pipeline_metrics = {}
    
    def __enter__(self):
        Logger.info("Initializing resources for FluxPipelineTrigger.")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on exit."""
        del self.text_encoder.model
        del self.text_encoder_2
        del self.transformer.model
        del self.vae_decoder.model
        
        # Force garbage collection for NPU resource cleanup
        import gc
        gc.collect()
        Logger.debug("Models in FluxPipelineTrigger are released")
    
    def run(
        self,
        height: int = 1024,
        width: int = 1024,
        prompt: str = "",
        prompt_2: str = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 256,
        seed: int = None,
    ):
        """
        Run the Flux pipeline to generate images.
        
        Args:
            height: Image height (must be divisible by 8)
            width: Image width (must be divisible by 8)
            prompt: Text prompt for image generation
            prompt_2: Optional second prompt for T5 encoder (defaults to prompt)
            num_inference_steps: Number of denoising steps
            num_images_per_prompt: Number of images to generate per prompt
            guidance_scale: Guidance scale for classifier-free guidance
            max_sequence_length: Maximum sequence length for T5 encoder
            seed: Random seed for reproducibility
            
        Returns:
            List of generated PIL images
        """
        # Create pipeline
        pipe = OnnxFluxPipelineAMD(
            scheduler=self.scheduler,
            vae_decoder=self.vae_decoder,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_2,
            tokenizer_2=self.tokenizer_2,
            transformer=self.transformer,
        )
        
        # Prepare config for logging
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
        
        # Set up generator
        generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        total_mem = 0
        if self.enable_profile:
            # Warm-up run
            Logger.info("Running warm-up...")
            t_start = time.perf_counter()
            output = pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=generator,
            )
            execution_time = time.perf_counter() - t_start
            Logger.debug(f"Warm-up execution time: {execution_time:.2f}s")
            Logger.debug(f"Current Mem while warm up: {common.measure_mem()}MB")
            
            # Save warm-up metrics
            perf_gpu_time_warm_up = copy.deepcopy(pipe.perf_time_gpu_model)
            perf_time_dict_warm_up = copy.deepcopy(pipe.perf_time_dict)
            pipe._clear_time_dict()
            Logger.debug("------------------------------")
            
            # Profiling runs
            Logger.info(f"Running {self.profiling_rounds} profiling rounds...")
            total_mem = 0
            t_start = time.perf_counter()
            
            for round_idx in range(self.profiling_rounds):
                total_mem += common.measure_mem()
                output = pipe(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale=guidance_scale,
                    max_sequence_length=max_sequence_length,
                    generator=generator,
                )
                Logger.debug(f"Round {round_idx + 1} - Current Mem: {common.measure_mem()}MB")
                
                # Progress marker for orchestrator tracking
                try:
                    print(f"__ROUND_COMPLETE__ {round_idx+1}/{self.profiling_rounds}", flush=True)
                except Exception:
                    pass
            
            t_total = time.perf_counter() - t_start
            images = output.images
            
            # Store metrics
            key = f"height_{height}_width_{width}_seq_len_{max_sequence_length}"
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
            
            Logger.info("============================================================")
            Logger.info(f"height: {height}, width: {width}, max_sequence_length: {max_sequence_length}")
            Logger.info("============================================================")
            common.log_pipeline_metrics(self.pipeline_metrics[key])
        
        else:
            # Single run without profiling
            Logger.info("Running pipeline...")
            start = time.perf_counter()
            output = pipe(
                prompt=prompt,
                prompt_2=prompt_2,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=generator,
            )
            total_mem += common.measure_mem()
            execution_time = time.perf_counter() - start
            Logger.info(f"Pipeline execution time = {execution_time:.6f}s")
            images = output.images
        
        return images
