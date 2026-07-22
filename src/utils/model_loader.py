# Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
"""Unified model loading: single entry ``load_model`` + ONNX / T5 NPU loaders."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import ctypes
import onnxruntime
import torch
from diffusers.pipelines.onnx_utils import OnnxRuntimeModel
from transformers import (
    AutoTokenizer,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPImageProcessor,
    Qwen2TokenizerFast,
    T5TokenizerFast,
)

from enum import Enum

from src.utils import common
import importlib

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    ONNX_MODEL = "onnx_model"
    T5_NPU_MODEL = "t5_npu_model"
    CLIP_TEXT_MODEL = "clip_text_model"
    CLIP_TEXT_WITH_PROJECTION = "clip_text_with_projection"
    CLIP_TOKENIZER = "clip_tokenizer"
    T5_TOKENIZER = "t5_tokenizer"
    SCHEDULER = "scheduler"
    CLIP_IMAGE_PROCESSOR = "clip_image_processor"
    AUTO_TOKENIZER = "auto_tokenizer"
    QWEN2_TOKENIZER = "qwen2_tokenizer"


@dataclass
class ModelLoadConfig:
    """Configuration for model loading.
    
    Args:
        model_path: base path to the model
        model_type: type of the model 
        model_file: file name of the model
        bin_file: file name of the bin file
        disable_dd_env: environment variable to disable DD
        enable_compile: if True, enable compile
        execution_providers: list of execution providers
        custom_op_path: path to the custom op
        width: width of the model
        t5_sequence_len: sequence length of the model
        is_dynamic: if True, use dynamic model
    """

    model_path: str
    model_type: ModelType = ModelType.ONNX_MODEL
    model_file: Optional[str] = None
    bin_file: Optional[str] = None
    disable_dd_env: Optional[str] = None
    enable_compile: bool = False
    is_dynamic: bool = False
    low_memory_threshold_gb: float = -1.0
    execution_providers: List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    custom_op_path: Optional[str] = None
    width: Optional[int] = None
    t5_sequence_len: Optional[int] = None

@dataclass
class LoadMetricsCollector:
    """
    Lightweight metrics collector for model loading.

    Tracks:
    1. per (model_type, device)
    2. total per device
    3. global total
    """

    # (model, device) -> value
    time_map: Dict[Tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    mem_map: Dict[Tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))

    # device -> total
    device_time: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    device_mem: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # global total
    total_time: float = 0.0
    total_mem: float = 0.0

    def record(self, model_type: str, device: str, time_s: float, mem_mb: float):
        key = (model_type, device)

        # per model
        self.time_map[key] += time_s
        self.mem_map[key] += mem_mb

        # per device
        self.device_time[device] += time_s
        self.device_mem[device] += mem_mb

        # global
        self.total_time += time_s
        self.total_mem += mem_mb

    def summary(self) -> Dict:
        return {
            "per_model": {
                f"{m}:{d}": {
                    "time_s": self.time_map[(m, d)],
                    "mem_mb": self.mem_map[(m, d)],
                }
                for (m, d) in self.time_map
            },
            "per_device": {
                d: {
                    "time_s": self.device_time[d],
                    "mem_mb": self.device_mem[d],
                }
                for d in self.device_time
            },
            "total": {
                "time_s": self.total_time,
                "mem_mb": self.total_mem,
            },
        }

    def print_summary(self):
        """
        Example:
        ================ MODEL LOAD SUMMARY =================

        MODEL           DEVICE   TIME(s)  MEM(MB)
        --------------------------------------------------
        unet            NPU      0.45     300
        text_encoder    CPU      0.12      80

        ------------------------------------------------
        TOTAL TIME: 0.57s (NPU: 0.45s | CPU: 0.12s)
        TOTAL MEM : 380MB (NPU: 300MB | CPU: 80MB)

        ====================================================
        """
        logger.info("================ MODEL LOAD SUMMARY =================")

        # -------- per model table --------
        # Get the longest model name, adaptively calculate the length
        max_model_len = len("MODEL") # default at least len("MODEL")
        if self.time_map:
            max_model_len = max(max_model_len, max(len(str(m)) for m, _ in self.time_map.keys()))

        logger.info(f"{'MODEL':<{max_model_len}} {'DEVICE':<8} {'TIME(s)':<8} {'MEM(MB)':<8}")
        logger.info("-" * (max_model_len + 8 + 8 + 8 + 3))
 
        for (model, device), t in self.time_map.items():
            mem = self.mem_map[(model, device)]
            logger.info(f"{model:<{max_model_len}} {device:<8} {t:<8.2f} {mem:<8.0f}")

        # -------- summary --------
        npu_t = self.device_time.get("NPU", 0.0)
        cpu_t = self.device_time.get("CPU", 0.0)

        npu_m = self.device_mem.get("NPU", 0.0)
        cpu_m = self.device_mem.get("CPU", 0.0)

        t_total = self.total_time
        m_total = self.total_mem

        logger.info("-" * (max_model_len + 8 + 8 + 8 + 3))

        time_str = f"{t_total:.2f}s"
        mem_str  = f"{m_total:.0f}MB"
        val_w = max(len(time_str), len(mem_str))

        logger.info(f"TOTAL TIME: {time_str:<{val_w}} (NPU: {npu_t:.2f}s  | CPU: {cpu_t:.2f}s)")
        logger.info(f"TOTAL MEM : {mem_str:<{val_w}} (NPU: {npu_m:.0f}MB | CPU: {cpu_m:.0f}MB)")
        logger.info("====================================================")


class ModelLoader(Protocol):
    def load(self, params: Any) -> Any: ...


class CLIPTokenizerLoader:
    @staticmethod
    def load(tokenizer_path: str) -> Any:
        model_name = os.path.basename(os.path.normpath(tokenizer_path))
        logger.info("Loading CLIPTokenizer %s ... ", model_name)
        try:
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        except OSError:
            raise ValueError("Don't find the model file for %s", tokenizer_path)
        return tokenizer


class CLIPTextModelLoader:
    @staticmethod
    def load(text_model_path: str) -> Any:
        model_name = os.path.basename(os.path.normpath(text_model_path))
        logger.info("Loading CLIPTextModel %s ... ", model_name)
        try:
            text_encoder = CLIPTextModel.from_pretrained(text_model_path)
        except OSError:
            raise ValueError("Don't find the model file for %s", text_model_path)
        return text_encoder


class CLIPTextModelWithProjectionLoader:
    @staticmethod
    def load(text_model_path: str) -> Any:
        model_name = os.path.basename(os.path.normpath(text_model_path))
        logger.info("Loading CLIPTextModelWithProjection %s ... ", model_name)
        try:
            text_encoder = CLIPTextModelWithProjection.from_pretrained(text_model_path)
        except OSError:
            raise ValueError("Don't find the model file for %s", text_model_path)
        return text_encoder


class T5TokenizerFastLoader:
    @staticmethod
    def load(tokenizer_path: str) -> Any:
        model_name = os.path.basename(os.path.normpath(tokenizer_path))
        logger.info("Loading T5TokenizerFast %s ... ", model_name)
        try:
            tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
        except OSError:
            raise ValueError("Don't find the model file for %s", tokenizer_path)
        return tokenizer


class AutoTokenizerLoader:
    @staticmethod
    def load(tokenizer_path: str) -> Any:
        model_name = os.path.basename(os.path.normpath(tokenizer_path))
        logger.info("Loading AutoTokenizer %s ... ", model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except OSError:
            raise ValueError("Don't find the model file for %s", tokenizer_path)
        return tokenizer


class Qwen2TokenizerFastLoader:
    @staticmethod
    def load(tokenizer_path: str) -> Any:
        model_name = os.path.basename(os.path.normpath(tokenizer_path))
        logger.info("Loading Qwen2TokenizerFast %s ... ", model_name)
        try:
            tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path)
        except OSError:
            raise ValueError("Don't find the model file for %s", tokenizer_path)
        return tokenizer


class CLIPImageProcessorLoader:
    @staticmethod
    def load(image_processor_path: str) -> Any:
        model_name = os.path.basename(os.path.normpath(image_processor_path))
        logger.info("Loading CLIPImageProcessor %s ... ", model_name)
        try:
            image_processor = CLIPImageProcessor.from_pretrained(image_processor_path)
        except OSError:
            raise ValueError("Don't find the model file for %s", image_processor_path)
        return image_processor


class SchedulerLoader:
    @staticmethod
    def load(scheduler_path: str) -> Any:
        config_path = os.path.join(scheduler_path, "scheduler_config.json")

        try:
            with open(config_path, "r") as f:
                scheduler_name = json.load(f).get("_class_name")
        except OSError:
            raise ValueError("Don't find the config file for %s", config_path)

        scheduler_cls = getattr(
            importlib.import_module("diffusers.schedulers"),
            scheduler_name,
        )
        logger.info("Loading Scheduler ... ")
        return scheduler_cls.from_pretrained(scheduler_path)


class OnnxModelLoader:
    @staticmethod
    def load(model_file_path: str,
            config_file_path: str,
            execution_providers: Optional[List[str]] = None,
            session_options: Optional[onnxruntime.SessionOptions] = None):

        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        model_name = Path(config_file_path).parent.name

        logger.info("Loading ONNX model %s ... ", model_name)
        logger.debug("Model path: %s", model_file_path)
        logger.debug("Config path: %s", config_file_path)

        # When execution_providers is None, let session_options drive EP selection
        # (avoids the "both set" conflict when session_options already has providers).
        # When session_options is None and execution_providers is None, fall back to CPU.
        if execution_providers is None and session_options is None:
            execution_providers = ["CPUExecutionProvider"]

        session = onnxruntime.InferenceSession(
            model_file_path, sess_options=session_options, providers=execution_providers
        )
        for provider in session.get_providers():
            logger.debug("  - provider: %s", provider)
        model = OnnxRuntimeModel(session)
        try:
            with open(config_file_path, "r", encoding="utf-8") as file:
                model.config = json.load(file)
        except OSError:
            logger.warning("Don't find the config file for %s", config_file_path)
            model.config = {}

        return model


class T5NpuModelLoader:
    @staticmethod
    def load(model_path: str,
            model_file: Optional[str] = None,
            bin_file: Optional[str] = None):
        if model_file is None:
            model_file = "serialized_quantized_t5-v1_1-xxl_w4_g128_gptq.safetensors"
        if bin_file is None:
            bin_file = "serialized_quantized_t5-v1_1-xxl_w4_g128_gptq.bin"

        model_name = os.path.basename(os.path.normpath(model_path))
        logger.info("Loading NPU T5 model %s ... ", model_name)

        safetensors_path = os.path.join(model_path, model_file)
        bin_path = os.path.join(model_path, bin_file)
        common.setup_npu_runtime(bin_path)

        from modeling_t5 import T5Config, T5EncoderModel  # pyright: ignore[reportMissingImports]
        from qlinear import AIEGEMM, QLinearPerGrp  # pyright: ignore[reportMissingImports]
        from safetensors import safe_open
        from safetensors.torch import load_file

        with safe_open(safetensors_path, framework="pt") as f:
            meta = f.metadata()
        with torch.device("meta"):
            model = T5EncoderModel(T5Config.from_dict(json.loads(meta["__config__"])))
        QLinearPerGrp.prepare_model(model, meta)
        model.load_state_dict(
            load_file(safetensors_path, device="cpu"), strict=False, assign=True
        )
        model.model_name = "t5"
        AIEGEMM.load_npu(model, meta)
        return model


def detect_device(execution_providers: Optional[List[str]]) -> str:
    if execution_providers is None:
        return "CPU"
    if "RyzenAILightExecutionProvider" in execution_providers:
        return "NPU"
    elif "DmlExecutionProvider" in execution_providers:
        return "GPU"
    else:
        return "CPU"

def load_model(cfg: ModelLoadConfig, metrics_collector: LoadMetricsCollector):
    """Single public entry: dispatch to ONNX or T5 NPU loader by model type."""

    device = detect_device(cfg.execution_providers)
    model_name = Path(cfg.model_path).name

    start_time = time.perf_counter()
    start_mem = common.measure_mem()

    return_model = None
    if cfg.model_type == ModelType.CLIP_TEXT_MODEL:
        return_model = CLIPTextModelLoader.load(cfg.model_path)
    elif cfg.model_type == ModelType.CLIP_TEXT_WITH_PROJECTION:
        return_model = CLIPTextModelWithProjectionLoader.load(cfg.model_path)
    elif cfg.model_type == ModelType.CLIP_TOKENIZER:
        return_model = CLIPTokenizerLoader.load(cfg.model_path)
    elif cfg.model_type == ModelType.T5_TOKENIZER:
        return_model = T5TokenizerFastLoader.load(cfg.model_path)
    elif cfg.model_type == ModelType.AUTO_TOKENIZER:
        return_model = AutoTokenizerLoader.load(cfg.model_path)
    elif cfg.model_type == ModelType.QWEN2_TOKENIZER:
        return_model = Qwen2TokenizerFastLoader.load(cfg.model_path)
    elif cfg.model_type == ModelType.SCHEDULER:
        return_model = SchedulerLoader.load(cfg.model_path)
    elif cfg.model_type == ModelType.CLIP_IMAGE_PROCESSOR:
        return_model = CLIPImageProcessorLoader.load(cfg.model_path)
    elif cfg.model_type == ModelType.T5_NPU_MODEL and device == "NPU":
        return_model = T5NpuModelLoader.load(cfg.model_path, cfg.model_file, cfg.bin_file)
    elif cfg.model_type == ModelType.ONNX_MODEL:
        device = detect_device(cfg.execution_providers)
        config_file_path = os.path.join(cfg.model_path, "config.json")

        model_path = common.resolve_dd_model_path(
            model_path=Path(cfg.model_path).resolve(),
            device=device,
            width=cfg.width,
            t5_sequence_len=cfg.t5_sequence_len,
            is_dynamic=cfg.is_dynamic,
        )
        model_file = cfg.model_file
        if model_file is None:
            model_file = "replaced.onnx" if device == "NPU" else "model.onnx"
        model_file_path = os.path.join(model_path, model_file)

        session_options = None
        eps = cfg.execution_providers
        if device == "NPU":
            session_options = common.config_session_options(
                custom_op_path=cfg.custom_op_path,
                dd_model_path=model_path,
                enable_dd_fusion_compile=cfg.enable_compile,
                low_memory_threshold_gb=cfg.low_memory_threshold_gb,
                execution_providers=cfg.execution_providers,
            )
            eps = None
        return_model = OnnxModelLoader.load(model_file_path, config_file_path, eps, session_options)
    else:
        raise ValueError(f"Model type {cfg.model_type} is not supported for device {device}")

    metrics_collector.record(model_name, device, time.perf_counter() - start_time, common.measure_mem() - start_mem)
    return return_model

