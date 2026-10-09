#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import ctypes
import inspect
import json
import logging as Logger
import os
import re
import sys
import time
from dataclasses import dataclass
import warnings
import psutil
from pathlib import Path
try:
    import winreg
    _WINREG_AVAILABLE = True
except ImportError:
    _WINREG_AVAILABLE = False
import pandas as pd
import torch
import onnxruntime
from huggingface_hub import snapshot_download
from typing import Any, Dict, List, Optional, Tuple, Union

process = psutil.Process()
from diffusers import OnnxRuntimeModel
from diffusers.pipelines.onnx_utils import OnnxRuntimeModel

_EP_REGISTERED = False
EP_NAME = "RyzenAILightExecutionProvider"
_ORT_ENABLE_PROFILING = False
_ORT_MODEL_ID = "model"


def configure_ort_runtime(enable_profile: bool = False, model_id: Optional[str] = None) -> None:
    """Configure ORT runtime: always suppress logs; enable session profiling when requested."""
    global _ORT_ENABLE_PROFILING, _ORT_MODEL_ID
    _ORT_ENABLE_PROFILING = enable_profile
    if model_id:
        _ORT_MODEL_ID = model_id
    onnxruntime.set_default_logger_severity(4)


def ort_profiling_enabled() -> bool:
    return _ORT_ENABLE_PROFILING


def _sanitize_profile_prefix_component(name: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\s]+', "_", name.strip())
    return sanitized.strip("._") or "model"


def ort_profile_file_prefix(model_name: str) -> str:
    model_id = _sanitize_profile_prefix_component(_ORT_MODEL_ID)
    component = _sanitize_profile_prefix_component(model_name)
    return f"{model_id}_{component}_"


def _apply_ort_logging(session_options: onnxruntime.SessionOptions) -> None:
    session_options.log_severity_level = 4


def _apply_ort_profiling(
    session_options: onnxruntime.SessionOptions,
    model_name: Optional[str] = None,
) -> None:
    if not _ORT_ENABLE_PROFILING:
        return
    # session_options.enable_profiling = True
    # if model_name:
    #     session_options.profile_file_prefix = ort_profile_file_prefix(model_name)


def create_ort_session_options(model_name: Optional[str] = None) -> onnxruntime.SessionOptions:
    session_options = onnxruntime.SessionOptions()
    _apply_ort_logging(session_options)
    _apply_ort_profiling(session_options, model_name)
    return session_options


def _configure_external_loggers():
    """Reduce noisy third-party download/network logs."""
    warning_loggers = (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "transformers.modeling_utils",
        "filelock",
        "urllib3",
        "urllib3.connectionpool",
        "requests",
    )
    for logger_name in warning_loggers:
        Logger.getLogger(logger_name).setLevel(Logger.WARNING)


_configure_external_loggers()

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

from datetime import datetime

# CHANGE ADDED: Dynamic provider label helper for --force-cpu functionality
# This function returns "CPU" when --force-cpu is used (ORT_DISABLE_GPU=1), "NPU" otherwise
def _get_provider_label():
    """Get the appropriate provider label based on environment variables."""
    if os.environ.get("ORT_DISABLE_GPU") == "1":
        return "CPU"
    else:
        return "NPU"

def resolve_project_path(sub_path: str) -> str:
    """Resolve a path relative to the project root without requiring it to exist."""
    project_root = Path(__file__).resolve().parents[2]
    return str((project_root / sub_path).resolve())


def get_absolute_path(sub_path: str) -> str:
    absolute_path = resolve_project_path(sub_path)
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"File not found: {absolute_path}")
    return absolute_path


def download_model_from_huggingface(model_id: str, force_download: bool = False, revision: str = None) -> str:
    """
    Download model from Hugging Face to local cache directory
    
    Args:
        model_id: Hugging Face model ID (e.g., "amd/stable-diffusion-1.5_amdnpu")
        force_download: Whether to force re-download (even if local cache exists)
        revision: Git branch, tag, or commit hash (e.g., "1.7.0", "main", "v1.0.0"). Default is None (use default branch)
    
    Returns:
        str: Local path of the downloaded model
    """
    try:
        # Set cache directory (under models/ in project root by default).
        models_root = (
            os.environ.get("RYZENAI_GENAI_SD_MODELS_ROOT")
        )
        if models_root is not None:
            cache_dir = Path(models_root)
        else:
            cache_dir = Path(resolve_project_path("models"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Target path where the model will be downloaded
        model_name = model_id.replace("/", "_")
        local_model_path = cache_dir / model_name
        revision_file = local_model_path / ".revision"

        # pr-231: revision-aware caching -- if a revision is requested, check
        # whether the cached model was downloaded with that same revision.
        # If revisions don't match, re-download with the correct one.
        if local_model_path.exists() and not force_download:
            cached_revision = None
            if revision_file.exists():
                cached_revision = revision_file.read_text(encoding='utf-8').strip()

            if revision and cached_revision != revision:
                Logger.info(f"Revision mismatch: cached='{cached_revision}', requested='{revision}'. Re-downloading...")
            else:
                Logger.debug(f"Using locally cached model: {local_model_path}")
                return str(local_model_path)

        Logger.info(f"Starting download from Hugging Face: {model_id}")
        if revision:
            Logger.debug(f"Using revision/branch: {revision}")
        Logger.debug(f"Target path: {local_model_path}")
        Logger.debug("This may take a few minutes depending on network speed...")
        Logger.debug("Using snapshot_download (with built-in progress bar)...")
        Logger.debug("")

        # Use snapshot_download to download entire repository
        # local_dir: directly download to the specified directory
        # snapshot_download has built-in progress bar via tqdm
        # revision: specify branch, tag, or commit hash
        try:
            download_kwargs = {
                "repo_id": model_id,
                "local_dir": str(local_model_path),
                "force_download": force_download,
            }
            if revision:
                download_kwargs["revision"] = revision

            downloaded_path = snapshot_download(**download_kwargs)

            # pr-231: save revision info for future cache validation
            if revision:
                revision_file.write_text(revision, encoding='utf-8')
            elif revision_file.exists():
                revision_file.unlink()

            Logger.debug("")
            Logger.debug(f"Model download completed: {local_model_path}")

        except Exception as download_error:
            Logger.error(f"snapshot_download failed: {str(download_error)}")
            Logger.error("This might be due to:")
            Logger.error("  1. Network connection issues")
            Logger.error("  2. Model repository access permissions")
            Logger.error("  3. Hugging Face API rate limiting")
            Logger.error("  4. Insufficient disk space")
            raise

        return str(local_model_path)

    except Exception as e:
        Logger.error(f"Failed to download model from Hugging Face: {str(e)}")
        Logger.error(f"Please check:")
        Logger.error(f"  1. Model ID is correct: {model_id}")
        Logger.error(f"  2. Network connection is working")
        Logger.error(f"  3. Hugging Face Hub is accessible (check huggingface.co)")
        Logger.error(f"  4. You have access permission to the model")
        Logger.error(f"  5. Sufficient disk space available")
        Logger.error(f"  6. Or manually specify --model_path argument")
        raise


def generate_filename(
    model_id: str,
    width: int,
    height: int,
    num_steps: int,
    prompt_idx: Optional[int] = None,
    image_idx: Optional[int] = None,
    controlnet: Optional[str] = None,
    run_mode: Optional[str] = None,
    batch_size: Optional[int] = None,
    suffix: Optional[str] = ".png",
) -> str:
    """
    Generate standardized filename.
    
    Supports various filename patterns used across different scripts.
    
    Args:
        model_id: Model identifier (e.g., "stabilityai/sd3")
        width: Image width
        height: Image height
        num_steps: Number of inference steps
        image_idx: Image index in batch
        prompt_idx: Prompt index
        controlnet: ControlNet type (optional, e.g., "canny", "pose")
        run_mode: Run mode (optional, e.g., "profiling", "batch")
        batch_size: Batch size (optional, included in filename if provided)
        suffix: File extension (optional, default is ".png")
    Returns:
        Generated filename
        
    Examples:
        # Simple filename
        filename = generate_filename(
            "stabilityai/sd-turbo", width=512, height=512, num_steps=20, suffix=".png", image_idx=0, prompt_idx=0
        )
        # -> "sd-turbo_img0_512x512_steps20_prompt0_20250108_120000.png"
        
        # ControlNet filename  
        filename = generate_filename(   
            "stabilityai/sd3", width=1024, height=1024, num_steps=8, suffix=".png", image_idx=0, prompt_idx=0,
            controlnet="canny", run_mode="profiling"
        )
        # -> "sd3_img0_canny_profiling_1024x1024_steps8_prompt0_20250108_120000.png"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = model_id.split('/')[-1]
    parts = [model_name]

    if image_idx is not None:
        parts.append(f"img{image_idx}")

    # Add controlnet info
    if controlnet and controlnet.lower() != "none":
        parts.append(controlnet)

    # Add run mode
    if run_mode:
        parts.append(run_mode)

    # Add resolution
    parts.append(f"{width}x{height}")

    # Add steps
    parts.append(f"steps{num_steps}")

    # Add prompt index
    if prompt_idx is not None:
        parts.append(f"prompt{prompt_idx}")

    # Add batch size if provided
    if batch_size is not None:
        parts.append(f"bs{batch_size}")

    # Add timestamp
    parts.append(timestamp)

    return "_".join(parts) + suffix


def setup_npu_runtime(bin_path):
    t5_path = get_absolute_path("src/t5")
    if t5_path not in sys.path:
        sys.path.insert(0, t5_path)
    os.environ["mha_npu"] = "1"
    os.environ["NPU_WTS_CACHE"] = bin_path

    from qlinear import AIEGEMM  # pyright: ignore[reportMissingImports]
    AIEGEMM.op_version = "v2"
    AIEGEMM.preemption = False
    AIEGEMM.pickle = True
    AIEGEMM.select_op_handle()


def register_ep(ep_dll: Path) -> None:
    """Register the RyzenAI EP once per process."""
    global _EP_REGISTERED
    if _EP_REGISTERED:
        return
    ep_dll = ep_dll.resolve()  # pr-231: resolve symlinks so dll_dir is correct
    dll_dir = ep_dll.parent
    # pr-231: os.add_dll_directory is required on Windows so ORT can resolve dependency DLLs
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))
    prev_cwd = os.getcwd()
    try:
        os.chdir(dll_dir)
        onnxruntime.register_execution_provider_library(EP_NAME, str(ep_dll))
    finally:
        os.chdir(prev_cwd)
    _EP_REGISTERED = True


def get_total_physical_memory_gb() -> float:
    """Get the total physical memory of the system (GiB) using the Windows GlobalMemoryStatusEx API."""
    kernel32 = ctypes.windll.kernel32
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullTotalPhys / (1024 ** 3)


def get_shared_memory_limit_gb() -> float:
    total_mem = get_total_physical_memory_gb()
    if not _WINREG_AVAILABLE:
        return total_mem
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager",
        )
        value, _ = winreg.QueryValueEx(key, "SystemPartitionCommitLimitPercentage")
        winreg.CloseKey(key)
    except FileNotFoundError:
        value = 50
    return total_mem * value / 100


def apply_low_memory_optimization(
    session_options: onnxruntime.SessionOptions,
    threshold_gb: float,
) -> bool:
    if threshold_gb < 0.0:
        return False
    shared_gb = get_shared_memory_limit_gb()
    Logger.info(
        f"Shared memory limit: {shared_gb:.2f} GiB, threshold: {threshold_gb:.2f} GiB"
    )
    if shared_gb < threshold_gb:
        session_options.add_session_config_entry("fusion_opt_enable_shared_const_buffers", "1")
        Logger.info(
            f"fusion_opt_enable_shared_const_buffers "
            f"(shared {shared_gb:.2f} GiB < threshold {threshold_gb:.2f} GiB)."
        )
        total_gb = get_total_physical_memory_gb()
        minimum_shared_threshold = threshold_gb / total_gb
        if total_gb < threshold_gb:
            pass
        else:
            Logger.warning(f"If the shared memory is less than the minimum shared threshold, the performance will be degraded.")
            Logger.warning(f"You can set the shared memory percentage by setting the value of HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers\\MemoryManager\\SystemPartitionCommitLimitPercentage at least {minimum_shared_threshold * 100}% to achieve better performance.")
        return True
    return False


def config_session_options(
    custom_op_path,
    dd_model_path,
    enable_dd_fusion_compile,
    low_memory_threshold_gb: float = -1.0,
    execution_providers: List[str] = ["CPUExecutionProvider"],
    model_name: Optional[str] = None,
):
    onnxruntime.set_default_logger_severity(4)
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    _apply_ort_logging(session_options)
    _apply_ort_profiling(session_options, model_name)
    low_memory_optimized = apply_low_memory_optimization(session_options, threshold_gb=low_memory_threshold_gb)
    if custom_op_path and os.path.exists(custom_op_path):
        # CHANGED: Check if custom_op_path exists before loading to prevent errors when path is None/invalid
        ctypes.CDLL(custom_op_path)
        register_ep(Path(custom_op_path))
        # DML must be registered before RyzenAI for fallback (e.g. NHWCConv on controlnet).
        # RyzenAI -> DML -> CPU can crash at runtime; DML -> RyzenAI -> CPU is stable.
        dd_path = Path(dd_model_path) if dd_model_path else None
        # const_key must point at the dd/ folder so the custom op can locate the
        # .fconst const-weight files; it is independent of low-memory optimization.
        const_key = dd_path.as_posix() if dd_path else ""
        compile_fusion_rt = "1" if enable_dd_fusion_compile else "0"
        dd_cache = (dd_path / "cache").as_posix() if dd_path else ""
        available_execution_providers = onnxruntime.get_available_providers()
        for ep in execution_providers:
            if ep == "DmlExecutionProvider":
                if "DmlExecutionProvider" not in available_execution_providers:
                    Logger.warning(f"DmlExecutionProvider is not available, skipping...")
                    continue
                if os.environ.get("ORT_DISABLE_GPU", "0") == "1":
                    Logger.warning(f"DmlExecutionProvider is disabled by ORT_DISABLE_GPU, skipping...")
                    continue
                session_options.add_provider("DmlExecutionProvider", {})
                Logger.debug(f"Added DmlExecutionProvider")
            elif ep == EP_NAME:
                session_options.add_provider_for_devices(
                    [d for d in onnxruntime.get_ep_devices() if d.ep_name == EP_NAME], {
                        "dd_cache": dd_cache,
                        "onnx_custom_ops_const_key": const_key,
                        "compile_fusion_rt": compile_fusion_rt
                    }
                )
                Logger.debug(f"Added {EP_NAME}")
            elif ep in available_execution_providers:
                session_options.add_provider(ep, {})
                Logger.debug(f"Added {ep}")
            else:
                Logger.warning(f"Execution provider {ep} is not available, skipping...")
        session_options.register_custom_ops_library(custom_op_path)
    return session_options


def resolve_root_model_path(
    model_id: Optional[str],
    root_path: Optional[str],
    revision: Optional[str] = None,
) -> str:
    """
    Return ``root_path`` if non-empty; otherwise download via ``model_id`` (and optional ``revision``).
    """
    if root_path is not None and str(root_path).strip() != "":
        return root_path
    Logger.debug("=" * 60)
    Logger.debug("model_path not provided, will download model from Hugging Face")
    Logger.debug(f"Model ID: {model_id}")
    if revision:
        Logger.debug(f"Revision/Branch: {revision}")
    Logger.debug("=" * 60)
    resolved = download_model_from_huggingface(model_id, revision=revision)
    Logger.debug("=" * 60)
    Logger.debug(f"Model ready: {resolved}")
    Logger.debug("Starting to load model components...")
    Logger.debug("=" * 60)
    return resolved


def resolve_dd_model_path(
    model_path: Path,
    device: str = "CPU",
    width: Optional[int] = None,
    t5_sequence_len: Optional[int] = None,
    is_dynamic: bool = False
) -> Path:
    """
    Resolve the directory of DD model for SD/SD3 pipelines.
    Args:
        model_path: path to the model
        device: device to use
        width: optional width (512, 1024, etc.)
        t5_sequence_len: optional sequence length for SD3
        is_dynamic: if True, return the dd/ path directly (tries dd/ then dynamic/dd/)
    Returns:
        Path to the model directory
    """
    if device != "NPU":
        return model_path

    # Two layouts exist on disk:
    #   SDXL/SD-style:      <component>/dd/
    #   FLUX/SD3.5-style:   <component>/dynamic/dd/
    dd_path = model_path / "dd"
    dynamic_dd_path = model_path / "dynamic" / "dd"

    if is_dynamic:
        if dd_path.exists():
            return dd_path
        if dynamic_dd_path.exists():
            return dynamic_dd_path
        raise FileNotFoundError(
            f"Dynamic model directory not found: tried {dd_path} and {dynamic_dd_path}"
        )

    # Build candidate path (width-specific, e.g. unet/512/dd)
    parts = []
    if width:
        parts.append(str(width))
    if t5_sequence_len:
        parts[-1] += f"_{t5_sequence_len}"
    parts.append("dd")

    candidate_path = model_path.joinpath(*parts)
    if candidate_path.exists():
        return candidate_path

    # Fallback: try flat dd/ then dynamic/dd/
    if dd_path.exists():
        return dd_path
    if dynamic_dd_path.exists():
        return dynamic_dd_path

    raise FileNotFoundError(f"Model path not found: {candidate_path}")


def print_config(config):
    print("config: {")
    for key, value in config.items():
        if isinstance(value, str):
            value = re.sub(r"\s+", " ", value.replace("\n", " ")).strip()
        print("    '{}': {}".format(key, value))
    print("}")


def measure_mem():
    return process.memory_info().vms / (1024 * 1024)


def str2bool(value):
    return value.lower() in ("true", "1", "yes")


def get_controlnet_model_name(target: str):
    supported_controlnet_types = ["outpainting", "removal", "inpainting", "canny", "tile", "pose", "depth", "union"]
    if target.lower() not in supported_controlnet_types:
        raise ValueError(f"Unsupported controlnet type: {target}, only support {', '.join(supported_controlnet_types)}")
    return f"controlnet-{target.lower()}"



def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Call scheduler.set_timesteps and return (timesteps, num_inference_steps).
    Supports custom timestep or sigma schedules passed by the caller.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. "
            "Please choose one to set custom values."
        )
    if timesteps is not None:
        if "timesteps" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(
                f"{scheduler.__class__}'s `set_timesteps` does not support custom "
                "timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if "sigmas" not in inspect.signature(scheduler.set_timesteps).parameters:
            raise ValueError(
                f"{scheduler.__class__}'s `set_timesteps` does not support custom "
                "sigma schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps