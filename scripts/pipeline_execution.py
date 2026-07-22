# pr-231: rewritten -- uses subprocess with unified run.py entry point
"""
Pipeline execution utilities -- subprocess runner invocation.

Builds arg lists from YAML config and launches test/run.py in a subprocess,
providing process isolation so NPU driver crashes don't kill the orchestrator.
"""

import os
import sys
import subprocess
import time
import argparse
import traceback as tb_module
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple

# Project root so we can resolve test/run.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline_helpers import (
    extract_arg_value, get_dimension_value, create_result_dict
)

# arg-validation: validate pipeline config args before subprocess launch
from config_validator import validate_extra_args


# ============================================================================
# Execution Provider Setup
# ============================================================================

def setup_execution_provider(args: argparse.Namespace, config: Dict[str, Any]):
    """Set ORT_DISABLE_GPU based on CLI --force-cpu or config defaults.execution.force_cpu."""
    execution_config = config.get('defaults', {}).get('execution', {})
    force_cpu = args.force_cpu or execution_config.get('force_cpu', False)

    if force_cpu:
        os.environ["ORT_DISABLE_GPU"] = "1"
        print("[CPU] CPU-only execution enabled")
    else:
        os.environ.pop("ORT_DISABLE_GPU", None)
        print("[AUTO] Automatic provider selection (NPU > DML > CPU)")


def setup_pipeline_execution_provider(config_item: Dict[str, Any],
                                      args: argparse.Namespace,
                                      config: Dict[str, Any]):
    """Per-pipeline EP override.  Priority: pipeline config > CLI > global."""
    pipeline_execution = config_item.get('execution', {})
    global_execution = config.get('defaults', {}).get('execution', {})

    force_cpu = (
        pipeline_execution.get('force_cpu')
        or args.force_cpu
        or global_execution.get('force_cpu', False)
    )

    if force_cpu:
        os.environ["ORT_DISABLE_GPU"] = "1"
        print(f"[CPU] {config_item['name']} -- CPU-only execution")
    else:
        os.environ.pop("ORT_DISABLE_GPU", None)


# ============================================================================
# Argument Helpers
# ============================================================================

def resolve_relative_path_args(extra_args: List[str], arg_names: List[str],
                               base_path: Path) -> List[str]:
    """Resolve relative path args to absolute paths."""
    result = []
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg in arg_names and i + 1 < len(extra_args):
            result.append(arg)
            result.append(str((base_path / extra_args[i + 1]).resolve()))
            i += 2
        else:
            result.append(arg)
            i += 1
    return result


def setup_pipeline_args(config_item: Dict[str, Any], benchmark_mode: bool,
                        config_defaults: Optional[Dict[str, Any]],
                        control_images_path: Optional[Path] = None,
                        config_files_path: Optional[Path] = None
                        ) -> Tuple[int, List[str], int, str]:
    """Prepare profiling_rounds, extra_args, num_inference_steps, model_path."""
    profiling_rounds = 10 if benchmark_mode else 1
    extra_args = config_item["extra_args"].copy()

    if config_defaults and "default_args" in config_defaults:
        extra_args = config_defaults["default_args"] + extra_args

    if control_images_path:
        extra_args = resolve_relative_path_args(
            extra_args, ['--control_image_path', '--control_mask_path'],
            control_images_path)
    if config_files_path:
        extra_args = resolve_relative_path_args(
            extra_args, ['--dynamic_shape_file_path'], config_files_path)

    num_inference_steps = extract_arg_value(
        extra_args, ["--num_inference_steps", "-n"], 0, as_int=True)
    model_path = extract_arg_value(
        extra_args, ["--model_path"], None, as_int=False)

    return profiling_rounds, extra_args, num_inference_steps, model_path


def get_pipeline_resolution(extra_args: List[str], config_item: Dict[str, Any],
                            config_defaults: Optional[Dict[str, Any]]) -> str:
    """Return 'WxH' string for result tracking."""
    width = get_dimension_value("width", extra_args, config_item, config_defaults)
    height = get_dimension_value("height", extra_args, config_item, config_defaults)
    return f"{width}x{height}"


def has_streaming_output(pipeline_configs: List[Dict[str, Any]],
                         config_defaults: Optional[Dict[str, Any]]) -> bool:
    """Check if any pipeline has streaming output enabled (for log-save decision)."""
    for cfg in pipeline_configs:
        streaming = cfg.get('streaming_output')
        if streaming is None and config_defaults:
            streaming = config_defaults.get('streaming_output', True)
        if streaming is None or streaming:
            return True
    return False


# ============================================================================
# Runner Arg-List Builder
# ============================================================================

def _build_runner_arg_list(model_id: str, extra_args: List[str],
                           profiling_rounds: int, image_only: bool = False,
                           custom_op_path: Optional[str] = None) -> List[str]:
    # pr-231: builds args for test/run.py subprocess
    """Build a string arg list for the test/run.py subprocess command."""
    args = ["--model_id", model_id, "--verbose", "--no_excel",
            "--profiling_rounds", str(profiling_rounds)]

    if not image_only:
        args.append("--enable_profile")

    # Filter out --output_image (not a runner_args flag)
    i = 0
    while i < len(extra_args):
        if extra_args[i] == "--output_image":
            i += 2          # skip flag + value
        else:
            args.append(extra_args[i])
            i += 1

    if custom_op_path:
        args.extend(["--custom_op_path", custom_op_path])

    return args


# ============================================================================
# Main Pipeline Execution -- subprocess with process isolation
# ============================================================================

def run_pipeline(config_item: Dict[str, Any], model_id: str,
                 test_path: Path, source_path: Path,
                 benchmark_mode: bool = False, timeout: int = 1200,
                 custom_prompt: Optional[str] = None,
                 prompt_file_path: Optional[str] = None,
                 sd3_controlnet_mode: str = "controlnet",
                 config_defaults: Optional[Dict[str, Any]] = None,
                 log_buffer: Optional[List[str]] = None,
                 determine_prompt_source_func: Optional[Callable] = None,
                 image_only: bool = False,
                 traceback_enabled: bool = False,
                 custom_op_path: Optional[str] = None,
                 control_images_path: Optional[Path] = None,
                 config_files_path: Optional[Path] = None) -> Dict[str, Any]:
    # pr-231: subprocess execution via unified test/run.py entry point
    """Execute a single pipeline in a subprocess for NPU process isolation."""
    pipeline_name = config_item["name"]

    # --- Prepare args from YAML config ----------------------------------
    profiling_rounds, extra_args, num_inference_steps, model_path = setup_pipeline_args(
        config_item, benchmark_mode, config_defaults,
        control_images_path, config_files_path)

    # arg-validation: validate merged args before subprocess launch
    validation = validate_extra_args(extra_args, pipeline_name)
    validation.log_results(pipeline_name)
    if not validation.is_valid:
        error_msg = "; ".join(validation.errors)
        return create_result_dict(
            script_name="run.py", model_id=model_id,
            pipeline_name=pipeline_name, success=False,
            duration=0.0, error=f"Argument validation failed: {error_msg}",
            resolution="N/A", num_inference_steps=num_inference_steps,
            model_path=model_path)

    resolution = get_pipeline_resolution(extra_args, config_item, config_defaults)

    # Ensure output_path resolves to test/generated_images (subprocess cwd is project root)
    if "--output_path" not in extra_args:
        extra_args.extend(["--output_path", str(test_path / "generated_images")])

    # Prompt source
    if determine_prompt_source_func is None:
        raise ValueError("determine_prompt_source_func must be provided")
    prompt_args, prompt_description, _ = determine_prompt_source_func(
        config_item, custom_prompt, prompt_file_path, config_defaults)
    extra_args.extend(prompt_args)

    # SD3 ControlNet mode conversion (--controlnet Canny -> None for text2img)
    if "--controlnet" in extra_args and sd3_controlnet_mode == "text2img":
        try:
            idx = extra_args.index("--controlnet")
            cn_type = extra_args[idx + 1].lower()
            if cn_type != "none":
                print(f"  [Mode] Converting ControlNet {cn_type.title()} -> text2img")
                extra_args[idx + 1] = "None"
        except (ValueError, IndexError):
            pass

    # --- Build arg list for subprocess ----------------------------------
    arg_list = _build_runner_arg_list(
        model_id, extra_args, profiling_rounds, image_only, custom_op_path)

    if prompt_description:
        print(f"  {prompt_description}")

    # Build the subprocess command: python test/run.py <args>
    script_path = _PROJECT_ROOT / "test" / "run.py"
    cmd = [sys.executable, str(script_path)] + arg_list

    if os.environ.get('DEBUG_PIPELINE'):
        print(f"  [Debug] Command: {' '.join(cmd)}")

    # --- Run subprocess with streaming output ---------------------------
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(_PROJECT_ROOT),
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Stream output line by line
        timing_lines = []
        for line in process.stdout:
            line = line.rstrip('\n')
            print(f"  {line}")
            if log_buffer is not None:
                log_buffer.append(line)
            # Capture timing/performance lines for results
            if any(kw in line for kw in ["execution time", "Pipeline", "Memory Usage"]):
                timing_lines.append(line)

        process.wait(timeout=timeout)
        duration = time.time() - start_time

        if process.returncode == 0:
            return create_result_dict(
                script_name="run.py", model_id=model_id,
                pipeline_name=pipeline_name, success=True,
                duration=duration, timing_lines=timing_lines,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
                model_path=model_path)
        else:
            return create_result_dict(
                script_name="run.py", model_id=model_id,
                pipeline_name=pipeline_name, success=False,
                duration=duration, error=f"Process exited with code {process.returncode}",
                timing_lines=timing_lines,
                resolution=resolution,
                num_inference_steps=num_inference_steps,
                model_path=model_path)

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        duration = time.time() - start_time
        return create_result_dict(
            script_name="run.py", model_id=model_id,
            pipeline_name=pipeline_name, success=False,
            duration=duration, error=f"Timed out after {timeout}s",
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            model_path=model_path)

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        if traceback_enabled:
            error_msg += "\n\nTraceback:\n" + tb_module.format_exc()
        return create_result_dict(
            script_name="run.py", model_id=model_id,
            pipeline_name=pipeline_name, success=False,
            duration=duration, error=error_msg,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            model_path=model_path)

