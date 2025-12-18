# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

"""
Consolidated helper functions for pipeline execution and reporting.

Contains utility functions for:
- File and directory management
- Argument extraction and processing
- File and directory cleaning
- Prompt source determination
- Result formatting and reporting
- Validation and dry-run checks
"""

import json
import sys
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any


def count_prompts_in_file(prompt_file_path: str) -> int:
    """
    Count the number of prompts in a JSON prompt file.
    
    Args:
        prompt_file_path (str): Path to the JSON file containing prompts
        
    Returns:
        int: Number of prompts found in the file, or 0 if file cannot be read
        
    Note:
        Returns 0 for non-existent files, invalid JSON, or non-list content.
        Errors are printed to stderr for debugging.
    """
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            print(f"Warning: Prompt file is not a list: {prompt_file_path}", file=sys.stderr)
            return 0
        return len(prompts)
    except FileNotFoundError:
        print(f"Warning: Prompt file not found: {prompt_file_path}", file=sys.stderr)
        return 0
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in prompt file {prompt_file_path}: {e}", file=sys.stderr)
        return 0
    except PermissionError as e:
        print(f"Warning: Permission denied reading {prompt_file_path}: {e}", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"Warning: Unexpected error reading {prompt_file_path}: {e}", file=sys.stderr)
        return 0


def clean_generated_images(test_path: Path) -> int:
    """
    Remove all existing images and Excel files from the generated_images folder.
    
    Args:
        test_path (Path): Path to the test directory containing generated_images folder
        
    Returns:
        int: Number of files removed (images + Excel files)
    """
    generated_images_path = test_path / "generated_images"
    
    if not generated_images_path.exists():
        print(f"  [Dir] Generated images directory doesn't exist: {generated_images_path}")
        return 0
    
    file_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp', '*.xlsx', '*.xls']
    removed_count = 0
    
    print(f"  [Cleanup] Cleaning existing images and Excel files from: {generated_images_path}")
    
    for pattern in file_patterns:
        files = list(generated_images_path.glob(pattern))
        for file in files:
            try:
                file.unlink()
                removed_count += 1
            except PermissionError:
                print(f"  [Warning] Permission denied removing {file.name}")
            except FileNotFoundError:
                # File was removed between glob and unlink - not an error
                pass
            except OSError as e:
                print(f"  [Warning] Could not remove {file.name}: {e}")
    
    if removed_count > 0:
        print(f"  [OK] Removed {removed_count} existing file(s) (images + Excel)")
    else:
        print(f"  [Info] No existing files found to remove")
    
    return removed_count


def clean_results(results_path: Path) -> int:
    """
    Remove all existing result files from the results folder.
    
    Args:
        results_path (Path): Path to the results directory
        
    Returns:
        int: Number of result files removed
    """
    if not results_path.exists():
        print(f"  [Dir] Results directory doesn't exist: {results_path}")
        return 0
    
    result_patterns = ['pipeline_results_*.txt', '*.log']
    removed_count = 0
    
    print(f"  [Cleanup] Cleaning existing results from: {results_path}")
    
    for pattern in result_patterns:
        result_files = list(results_path.glob(pattern))
        for result_file in result_files:
            try:
                result_file.unlink()
                removed_count += 1
            except PermissionError:
                print(f"  [Warning] Permission denied removing {result_file.name}")
            except FileNotFoundError:
                # File was removed between glob and unlink - not an error
                pass
            except OSError as e:
                print(f"  [Warning] Could not remove {result_file.name}: {e}")
    
    if removed_count > 0:
        print(f"  [OK] Removed {removed_count} existing result file(s)")
    else:
        print(f"  [Info] No result files found to remove")
    
    return removed_count


def determine_prompt_source(config_item: Dict[str, Any], custom_prompt: Optional[str] = None,
                           prompt_file_path: Optional[str] = None,
                           config_defaults: Optional[Dict[str, Any]] = None) -> Tuple[List[str], str, int]:
    """
    Determine the prompt source and return appropriate command arguments.
    
    Handles prompt priority: command-line > prompt_file > pipeline config > global defaults
    
    Args:
        config_item (Dict[str, Any]): Pipeline configuration from YAML
        custom_prompt (Optional[str]): Custom prompt from command line
        prompt_file_path (Optional[str]): Path to prompt file from command line
        config_defaults (Optional[Dict[str, Any]]): Global defaults from YAML config
        
    Returns:
        Tuple[List[str], str, int]: A tuple containing:
            - extra_args: List of command line arguments for prompt handling
            - description: Human-readable description of the prompt source
            - prompt_count: Number of prompts to process (for progress estimation)
    """
    extra_args = []
    description = ""
    prompt_count = 1
    
    if custom_prompt:
        extra_args.extend(["--prompt", custom_prompt])
        description = f"Using custom prompt: {custom_prompt[:60]}..."
        prompt_count = 1
    elif prompt_file_path:
        abs_prompt_path = str(Path(prompt_file_path).resolve())
        extra_args.extend(["--prompt_file_path", abs_prompt_path])
        prompt_count = count_prompts_in_file(abs_prompt_path)
        description = f"Using prompt file: {prompt_file_path}\n  → Processing {prompt_count} prompts"
    elif "prompt_file" in config_item:
        extra_args.extend(["--prompt_file_path", config_item["prompt_file"]])
        prompt_count = count_prompts_in_file(config_item["prompt_file"])
        description = f"Using YAML prompt file: {config_item['prompt_file']}\n  → Processing {prompt_count} prompts"
    elif "prompt" in config_item:
        extra_args.extend(["--prompt", config_item["prompt"]])
        prompt_count = 1
        description = f"Using YAML prompt: {config_item['prompt'][:60]}..."
    elif config_defaults and "prompt_file" in config_defaults:
        global_prompt_file = config_defaults["prompt_file"]
        abs_global_prompt_path = str(Path(global_prompt_file).resolve())
        extra_args.extend(["--prompt_file_path", abs_global_prompt_path])
        prompt_count = count_prompts_in_file(global_prompt_file)
        description = f"Using global default prompt file: {global_prompt_file}\n  → Processing {prompt_count} prompts"
    
    return extra_args, description, prompt_count


def get_model_name(model_id: str) -> str:
    """
    Extract clean model name from Hugging Face model ID for directory/filename usage.
    
    Args:
        model_id (str): Full model ID (e.g., "stabilityai/stable-diffusion-v1-5")
        
    Returns:
        str: Clean model name (e.g., "stable-diffusion-v1-5")
    """
    return model_id.split('/')[-1] if '/' in model_id else model_id


def filter_extra_args(extra_args: List[str], unsupported_args: List[str]) -> List[str]:
    """
    Remove unsupported arguments and their values from the argument list.
    
    Args:
        extra_args (List[str]): Original list of command line arguments
        unsupported_args (List[str]): List of argument names to remove
        
    Returns:
        List[str]: Filtered argument list with unsupported arguments removed
    """
    filtered_args = []
    skip_next = False
    for arg in extra_args:
        if skip_next:
            skip_next = False
            continue
        if arg in unsupported_args:
            skip_next = True
            continue
        filtered_args.append(arg)
    return filtered_args


def extract_arg_value(extra_args: List[str], arg_names: List[str], default_value: Any = None, as_int: bool = True) -> Any:
    """
    Extract argument value from extra_args list, supporting multiple argument name variants.
    
    Args:
        extra_args: List of command line arguments
        arg_names: List of argument names to search for (e.g., ["--width", "-W"])
        default_value: Value to return if not found
        as_int: Whether to convert value to int (False for string values like paths)
        
    Returns:
        Extracted value or default_value
    """
    for arg_name in arg_names:
        if arg_name in extra_args:
            try:
                idx = extra_args.index(arg_name)
                if idx + 1 >= len(extra_args):
                    print(f"Warning: Argument {arg_name} found but no value provided", file=sys.stderr)
                    continue
                value = extra_args[idx + 1]
                if as_int:
                    try:
                        return int(value)
                    except ValueError:
                        print(f"Warning: Cannot convert {arg_name} value '{value}' to int, using default", file=sys.stderr)
                        continue
                return value
            except (ValueError, IndexError) as e:
                print(f"Warning: Error extracting {arg_name}: {e}", file=sys.stderr)
                continue
    return default_value


def get_dimension_value(dimension: str, extra_args: List[str], config_item: Dict[str, Any], 
                       config_defaults: Optional[Dict[str, Any]], default: int = 512) -> int:
    """
    Get width or height value with priority: extra_args > config_item > config_defaults > default.
    Also adds to extra_args if not already present.
    
    Args:
        dimension: Either "width" or "height"
        extra_args: List of command line arguments (modified in place)
        config_item: Pipeline configuration
        config_defaults: Global default configuration
        default: Fallback value
        
    Returns:
        Resolved dimension value
    """
    arg_name = "--" + dimension
    short_name = "-W" if dimension == "width" else "-H"
    
    if arg_name in extra_args or short_name in extra_args:
        return extract_arg_value(extra_args, [arg_name, short_name], default)
    
    if dimension in config_item:
        value = config_item[dimension]
        extra_args.extend([arg_name, str(value)])
        return value
    
    if config_defaults and dimension in config_defaults:
        value = config_defaults[dimension]
        extra_args.extend([arg_name, str(value)])
        return value
    
    return default


def create_result_dict(script_name: str, model_id: str, pipeline_name: str, 
                      success: bool = False, duration: float = 0, 
                      error: str = "", timing_lines: Optional[List[str]] = None,
                      resolution: str = "", num_inference_steps: int = 0,
                      model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a standardized result dictionary for pipeline execution results.
    
    Args:
        script_name (str): Name of the pipeline script that was executed
        model_id (str): Hugging Face model identifier used
        pipeline_name (str): Human-readable pipeline name from configuration
        success (bool): Whether the pipeline executed successfully
        duration (float): Execution time in seconds
        error (str): Error message if the pipeline failed
        timing_lines (Optional[List[str]]): Detailed timing information lines
        resolution (str): Image resolution used (e.g., "512x512", "1024x1024")
        num_inference_steps (int): Number of inference/denoising steps
        model_path (Optional[str]): Path to the model files used
        
    Returns:
        Dict[str, Any]: Standardized result dictionary
    """
    return {
        "success": success,
        "duration": duration,
        "timing_lines": timing_lines or [],
        "error": error,
        "script": script_name,
        "model_id": model_id,
        "pipeline_name": pipeline_name,
        "resolution": resolution,
        "num_inference_steps": num_inference_steps,
        "model_path": model_path
    }


def print_pipeline_info(mode: str, script_name: str, model_id: str, prompt_description: str,
                       output_dir: str) -> None:
    """
    Print formatted information about the pipeline execution.
    
    Args:
        mode (str): Execution mode ("QUICK" or "BENCHMARK")
        script_name (str): Name of the pipeline script being executed
        model_id (str): Hugging Face model identifier
        prompt_description (str): Description of the prompt source being used
        output_dir (str): Directory where images will be saved
    """
    print(f"Running [{mode}]: {script_name} with {model_id}")
    
    if prompt_description:
        print(f"  → {prompt_description}")
    
    model_name = get_model_name(model_id)
    print(f"  → Images will be saved to: {output_dir}/")


# ============================================================================
# Result Reporting Functions
# ============================================================================

def save_log_file(log_buffer: List[str], mode_str: str, workspace_root: Path) -> str:
    """
    Save the complete log buffer to a timestamped text file.
    
    Args:
        log_buffer (List[str]): List of log lines to save
        mode_str (str): Mode string (e.g., "BENCHMARK" or "QUICK TEST")
        workspace_root (Path): Path to the workspace root
        
    Returns:
        str: Path to the saved log file
    """
    results_dir = workspace_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"pipeline_log_{mode_str.lower().replace(' ', '_')}_{timestamp}.txt"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Pipeline Execution Log - {mode_str} Mode\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for line in log_buffer:
            f.write(line + "\n")
    
    return str(log_file)


def save_results(results: List[Dict[str, Any]], mode_str: str, 
                total_duration: timedelta, benchmark_mode: bool, workspace_root: Path) -> str:
    """
    Save execution results to a timestamped file in the results directory.
    
    Args:
        results (List[Dict[str, Any]]): List of pipeline execution results
        mode_str (str): Mode description string for the report header
        total_duration (timedelta): Total execution time across all pipelines
        benchmark_mode (bool): Whether results are from benchmark mode
        workspace_root (Path): Path to the workspace root
        
    Returns:
        str: Filename of the saved results file for reference
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine filename suffix based on mode
    if mode_str == "IMAGE-ONLY":
        mode_suffix = "image_only"
    elif benchmark_mode:
        mode_suffix = "benchmark"
    else:
        mode_suffix = "quick"
    
    filename = f"pipeline_results_{mode_suffix}_{timestamp}.txt"
    
    results_dir = workspace_root / "results"
    results_dir.mkdir(exist_ok=True)
    filepath = results_dir / filename
    
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"PIPELINE PROFILING RESULTS - {mode_str} MODE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total duration: {total_duration}\n")
        f.write(f"Total runs: {total}\n")
        f.write(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)\n\n")
        
        for i, result in enumerate(results, 1):
            status = "SUCCESS" if result['success'] else "FAILED"
            resolution_str = f" @ {result.get('resolution', 'Unknown')}" if result.get('resolution') else ""
            f.write(f"{i}. {result['pipeline_name']} ({result['script']}) with {result['model_id']}{resolution_str}\n")
            f.write(f"   Status: {status} - Duration: {result['duration']:.2f}s\n")
            if result.get('model_path'):
                f.write(f"   Model Path: {result['model_path']}\n")
            if result.get('num_inference_steps', 0) > 0:
                f.write(f"   Inference Steps: {result['num_inference_steps']}\n")
            
            if result['success'] and result.get('timing_lines'):
                f.write("   Timing Summary:\n")
                for timing_line in result['timing_lines']:
                    f.write(f"      {timing_line}\n")
            elif not result['success']:
                error = result['error'][:200] + "..." if len(result['error']) > 200 else result['error']
                f.write(f"   Error: {error}\n")
            f.write("\n")
    
    return str(filepath)


def print_results_summary(all_results: List[Dict[str, Any]], mode_str: str, total_duration: timedelta):
    """
    Print a formatted summary of pipeline execution results to the console.
    
    Args:
        all_results (List[Dict[str, Any]]): List of all pipeline execution results
        mode_str (str): Mode description string ("QUICK TEST" or "BENCHMARK")
        total_duration (timedelta): Total execution time across all pipelines
    """
    successful = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY - {mode_str} MODE")
    print(f"{'='*60}")
    print(f"Total runs: {total}")
    print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"Total duration: {total_duration}")


def list_pipelines(config: Dict[str, Any]):
    """
    Display all available pipelines and pipeline groups to the console.
    
    Args:
        config (Dict[str, Any]): Full YAML configuration dictionary
    """
    print("\n" + "=" * 50)
    print("Available Pipelines")
    print("=" * 50)
    for name, pipeline in config.get('pipelines', {}).items():
        print(f"  {name:<25} - {pipeline.get('name', name)}")
    
    print("\n" + "=" * 50)
    print("Available Pipeline Groups")
    print("=" * 50)
    for group_name, pipelines in config.get('pipeline_groups', {}).items():
        print(f"  {group_name:<25} - {', '.join(pipelines)}")
    
    print(f"\nDefault pipelines: {', '.join(config.get('default_run', []))}")

