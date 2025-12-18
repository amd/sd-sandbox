#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.
#

import torch
import onnxruntime
import os
import json
import sys
import time
import logging as Logger
import re
import psutil
import ctypes
from pathlib import Path
import pandas as pd

process = psutil.Process()

# Fix diffusers ONNX detection issue
try:
    import diffusers.utils.import_utils as import_utils
    # Force enable ONNX detection if onnx and onnxruntime are available
    try:
        import onnx
        import onnxruntime
        import_utils._onnx_available = True
    except ImportError:
        pass
except ImportError:
    pass

from diffusers.pipelines.onnx_utils import OnnxRuntimeModel

# CHANGE ADDED: Dynamic provider label helper for --force-cpu functionality
# This function returns "CPU" when --force-cpu is used (ORT_DISABLE_GPU=1), "NPU" otherwise
def _get_provider_label():
    """Get the appropriate provider label based on environment variables."""
    if os.environ.get("ORT_DISABLE_GPU") == "1":
        return "CPU"
    else:
        return "NPU"
        
def setup_npu_runntime(root_path, bin_path):
    sys.path.append("../src/t5")
    os.environ["mha_npu"] = "1"
    os.environ["bo_path"] = bin_path


def LoadT5NPUTorchModel(
    root_path,
    model_path,
    folder,
    model_name="serialized_quantized_t5-v1_1-xxl_w4_g128_gptq.pth",
    bin_name="serialized_quantized_t5-v1_1-xxl_w4_g128_gptq.bin",
):
    Logger.debug("------------------------------")
    Logger.info(f"Load NPU model {model_path}\\{folder}")

    # setup NPU envs
    serialized_ckpt = os.path.join(model_path, folder, model_name)
    bin_path = os.path.join(model_path, folder, bin_name)
    setup_npu_runntime(root_path, bin_path)

    # load model
    t0 = time.perf_counter()
    model = torch.load(serialized_ckpt, weights_only=False)
    Logger.debug(f"Model {folder} loading time = {time.perf_counter() - t0}s")

    return model


def LoadModel(
    model_path,
    config_folder,
    folder,
    session_options=None,
    filename="model.onnx",
    providers=["CPUExecutionProvider"],
):
    Logger.info("Load {} ... ".format(folder))
    config_abs_path = os.path.join(model_path, config_folder, "config.json")
    model_abs_path = os.path.join(model_path, folder, filename)

    # load model
    t0 = time.perf_counter()
    m = onnxruntime.InferenceSession(
        model_abs_path, sess_options=session_options, providers=providers
    )
    Logger.debug(f"Model {folder} loading time = {time.perf_counter() - t0}s")

    # Print the active providers
    Logger.debug("Active providers:")
    for provider in m.get_providers():
        Logger.debug(f"  - provider: {provider}")

    m = OnnxRuntimeModel(m)
    try:
        with open(config_abs_path, "r") as file:
            config = json.load(file)

        m.config = config
    except:
        Logger.warning("Don't find the config file for " + folder)

    return m

def config_session_options(
    custom_op_path, dd_model_path, enable_dd_fusion_compile
):
    ctypes.CDLL(custom_op_path)
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    if os.path.exists(custom_op_path):
        session_options.add_session_config_entry("dd_cache", (Path(dd_model_path).parent / ".cache").as_posix())
        # model loading optimization
        session_options.add_session_config_entry(
            "onnx_custom_ops_const_key", dd_model_path
        )
        # can be commented out to reduce compiling time if you have compiled before
        if enable_dd_fusion_compile:
            session_options.add_session_config_entry("compile_fusion_rt", "True")
        session_options.register_custom_ops_library(custom_op_path)
    return session_options


def get_sd_dd_model_dir(model_type, width=None):
    if width:
        return f"{model_type}/{str(width)}/dd"
    return f"{model_type}/dd"

def get_sd_dd_dynamic_model_dir(model_type):
    return f"{model_type}/dynamic/dd"

def get_sd3_dd_model_dir(model_type, width, t5_sequence_len=None):
    width_str = "512" if width == 512 else "1024"

    # Add t5_sequence_len to path for controlnet or transformer models
    if t5_sequence_len is not None:
        return f"{model_type}/{width_str}_{t5_sequence_len}/dd"

    return f"{model_type}/{width_str}/dd"


def load_model_with_session(
    MODEL_PATH,
    model_type,
    model_file,
    custom_op_path="",
    enable_dd_fusion_compile=True,
    providers=["CPUExecutionProvider"],
    width=None,
    t5_sequence_len=None,
    is_dynamic=False,
):
    session = None
    dd_model_dir = model_type
    if custom_op_path:
        if is_dynamic:
            dd_model_dir = get_sd_dd_dynamic_model_dir(model_type)
        elif model_type.startswith("controlnet-") or model_type.startswith("transformer"):
            dd_model_dir = get_sd3_dd_model_dir(model_type, width, t5_sequence_len)
        else:
            dd_model_dir = get_sd_dd_model_dir(model_type, width)
        session = config_session_options(
            custom_op_path,
            os.path.join(MODEL_PATH, dd_model_dir, model_file),
            enable_dd_fusion_compile,
        )
    return LoadModel(
        MODEL_PATH,
        model_type,
        dd_model_dir,
        session_options=session,
        filename=model_file,
        providers=providers,
    )


def get_folder(model_id):
    save_folders = {
        "runwayml/stable-diffusion-v1-5": "./SD1.5",
        "stabilityai/stable-diffusion-2-1-base": "./SD2.1-base",
        "stabilityai/sd-turbo": "./SD-turbo",
        "stabilityai/stable-diffusion-2-1": "./SD2.1",
    }
    try:
        save_folder = save_folders[model_id]
    except:
        save_folder = model_id.replace("/", "_")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    return save_folder


def print_config(config):
    print("config: {")
    for key, value in config.items():
        if isinstance(value, str):
            value = re.sub(r"\s+", " ", value.replace("\n", " ")).strip()
        print("    '{}': {}".format(key, value))
    print("}")


def measure_mem():
    return process.memory_info().vms / (1024 * 1024)


def log_pipeline_metrics(pipeline_metrics):
    model_id = pipeline_metrics["model_id"]
    execution_time = pipeline_metrics["execution_time"]
    MODEL_PATH = pipeline_metrics["MODEL_PATH"]
    mem_dict = pipeline_metrics["mem_dict"]
    perf_time_dict_warm_up = pipeline_metrics["perf_time_dict_warm_up"]
    perf_gpu_time_warm_up = pipeline_metrics["perf_gpu_time_warm_up"]
    perf_time_dict = pipeline_metrics["perf_time_dict"]
    perf_gpu_time = pipeline_metrics["perf_gpu_time"]
    t_total = pipeline_metrics["t_total"]
    total_mem = pipeline_metrics["total_mem"]
    profiling_rounds = pipeline_metrics["profiling_rounds"]
    load_time_dict = pipeline_metrics["load_time_dict"]

    Logger.info("------------------------------")
    Logger.info(f"Model path = {MODEL_PATH}")

    Logger.info(
        f"Pipeline execution time of 1st Gen in performance mode = {execution_time:.6f}s"
    )
    for k, v in perf_time_dict_warm_up.items():
        if len(v):
            avg_time = sum(v) / len(v)
            # CHANGE MODIFIED: Dynamic provider label instead of hardcoded "(NPU)"
            Logger.info(
                f"==> {k}({_get_provider_label()}): avg time of 1st Gen in performance mode {avg_time:.6f}s"
            )
    for k, v in perf_gpu_time_warm_up.items():
        if len(v):
            avg_time = sum(v) / len(v)
            Logger.info(
                f"==> {k}(CPU): avg time of 1st Gen in performance mode {avg_time:.6f}s"
            )
    Logger.info("------------------------------")
    Logger.info(
        f"Average pipeline execution time (excluding first iter) in performance mode =  {t_total / profiling_rounds:.6f}s"
    )
    for k, v in perf_time_dict.items():
        if len(v):
            avg_time = sum(v) / len(v)
            # CHANGE MODIFIED: Dynamic provider label instead of hardcoded "(NPU)"
            Logger.info(
                f"==> {k}({_get_provider_label()}): avg time (excluding first iter) in performance mode {avg_time:.6f}s"
            )
    for k, v in perf_gpu_time.items():
        if len(v):
            avg_time = sum(v) / len(v)
            Logger.info(
                f"==> {k}(CPU): avg time (excluding first iter) in performance mode {avg_time:.6f}s"
            )

    Logger.info("Memory usage by model:")
    Logger.info("------------------------------")
    for model, mem in mem_dict.items():
        Logger.info(f"==> {model}: {mem:.2f}MB")
    Logger.info("Profile data in performance mode:")
    Logger.info("------------------------------")
    # Detect execution provider based on environment
    provider_name = "CPU" if os.environ.get("ORT_DISABLE_GPU", "0") == "1" else "NPU"
    Logger.info(f"{', '.join(perf_time_dict.keys())} are on {provider_name}, others are on CPU")
    Logger.info(f"Load time of all {provider_name} models: {load_time_dict['all_npu_models']:.6f}s")
    Logger.info(f"Load time of all models: {load_time_dict['all_models']:.6f}s")
    Logger.info(f"Load time of all models: {load_time_dict['all_models']:.6f}s")
    Logger.info(f"Pipeline time for 1st Gen : {execution_time:.6f}s")
    Logger.info(
        f"Average pipeline time(excluding first iter) : {t_total / profiling_rounds:.6f}s"
    )
    Logger.info(
        f"Total memory usage : {total_mem / profiling_rounds:.2f}MB ({(total_mem / profiling_rounds / 1024):.2f}GB)"
    )
    mem_sum = sum(mem_dict.values())
    Logger.info(f"Total NPU memory usage: {mem_sum:.2f}MB ({(mem_sum / 1024):.2f}GB)")


def save_pipeline_metrics_to_excel(save_path, data):
    def format_float(val):
        try:
            return round(val, 2)
        except:
            return ""

    def parse_hw_from_key(key):
        try:
            parts = key.split('_')
            return int(parts[1]), int(parts[3])
        except:
            return None, None

    def average_list_values(pipeline_metrics):
        for model_info in pipeline_metrics.values():
            for subkey, items in model_info.items():
                if isinstance(items, dict) and all(isinstance(i, list) for i in items.values()):
                    for k, v in items.items():
                        items[k] = sum(v) / len(v) if len(v) else 0.0

    average_list_values(data)

    rows = []
    for key, item in data.items():
        height, width = parse_hw_from_key(key)

        perf_dict = item.get("perf_time_dict", {})
        mem_dict = item.get("mem_dict", {})
        profiling_rounds = item.get("profiling_rounds", 1)

        row = {}
        row["model_id"] = item.get("model_id", "")
        row["Height"] = height
        row["Width"] = width
        row["Load time of NPU models (s)"] = format_float(item.get("load_time_dict", {}).get("all_npu_models"))
        row["Pipeline Time (1st Gen, warm-up) (s)"] = format_float(item.get("execution_time"))
        row["Pipeline Time (Excl. 1st) (s)"] = format_float(item.get("t_total") / profiling_rounds)
        for model_name in list(perf_dict.keys()):
            row["{} (s)".format(model_name)] = format_float(perf_dict.get("{}".format(model_name)))
        row["Total NPU Memory Usage (GB)"] = format_float(sum(mem_dict.values()) / 1024)
        row["Total Memory Usage (GB)"] = format_float(item.get("total_mem", 0) / profiling_rounds / 1024)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(save_path, index=False)
    print(f"Pipeline metrics Excel file saved to: {save_path}")


def str2bool(value):
    return value.lower() in ("true", "1", "yes")


def get_controlnet_model_name(target: str):
    if target.lower() == "OutPainting".lower():
        model_name = "controlnet-outpainting"
    elif target.lower() == "Removal".lower():
        model_name = "controlnet-removal"
    elif target.lower() == "InPainting".lower():
        model_name = "controlnet-inpainting"
    else:
        raise ValueError(f"Unsupported controlnet type: {target}, only support OutPainting, Removal, InPainting")

    return model_name


def get_normal_controlnet_model_name(target: str):
    if target.lower() == "Canny".lower():
        model_name = "controlnet-canny"
        control_img_url = "ref/canny.jpg"  # "https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg"
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"
    elif target.lower() == "Tile".lower():
        model_name = "controlnet-tile"
        control_img_url = "ref/tile.jpg"  # "https://huggingface.co/InstantX/SD3-Controlnet-Tile/resolve/main/tile.jpg"
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"
    elif target.lower() == "Pose".lower():
        model_name = "controlnet-pose"
        control_img_url = "ref/pose.jpg"  # "https://huggingface.co/InstantX/SD3-Controlnet-Pose/resolve/main/pose.jpg"
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"
    elif target.lower() in ["outpainting", "removal", "inpainting"]:
        raise ValueError(f"Unsupported controlnet type: {target}, please try run_sd3_controlnet_outpainting.py instead.")
    else:
        model_name = "controlnet-canny"
        control_img_url = "https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg"
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"

        Logger.warning(
            f"Unhandled target: {target}, will use Canny by default in normal controlnet pipeline"
        )

    return model_name, control_img_url, prompt, n_prompt
