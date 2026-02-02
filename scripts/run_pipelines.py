# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.
"""
Comprehensive pipeline runner script with support for:
- Quick testing (1 round) and benchmarking (10 rounds)
- Individual prompts, prompt files, and YAML-configured prompts  
- Custom output directories for organized image generation
- Per-pipeline customization via YAML configuration
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Import from helper modules
from pipeline_config import load_config, get_pipeline_configs, determine_prompt_source
from pipeline_execution import (
    setup_execution_provider, setup_pipeline_execution_provider,
    run_pipeline, has_streaming_output
)
from pipeline_helpers import (
    clean_generated_images, clean_results, save_log_file, save_results,
    print_results_summary, list_pipelines
)

# Configuration
WORKSPACE_ROOT = Path(__file__).parent.parent
CONFIG_DIR = WORKSPACE_ROOT / "config"
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"


def run_all_pipelines(pipeline_configs: List[Dict[str, Any]], paths: Dict[str, Path], 
                     args: argparse.Namespace, config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Execute all pipeline configurations and collect results.
    
    Iterates through all selected pipelines, runs each model variant, and provides
    real-time feedback on execution progress and immediate outcomes.
    
    Args:
        pipeline_configs (List[Dict[str, Any]]): List of pipeline configurations to run
        paths (Dict[str, Path]): Dictionary of important file/directory paths
        args (argparse.Namespace): Parsed command-line arguments
        config (Dict[str, Any]): Full YAML configuration dictionary
        
    Returns:
        Tuple[List[Dict[str, Any]], Optional[str]]: Tuple containing:
            - List of all pipeline execution results
            - Path to saved log file (if log saving was enabled), or None
    """
    all_results = []
    log_file_path = None
    
    # Check if log saving is enabled and any pipeline has streaming output
    config_defaults = config.get('defaults', {})
    enable_log_saving = args.save_log and has_streaming_output(pipeline_configs, config_defaults)
    
    # Create log buffer if log saving is enabled
    log_buffer = [] if enable_log_saving else None
    
    if enable_log_saving:
        print(f"📝 Log saving is enabled - full output will be saved to file")
    
    for i, config_item in enumerate(pipeline_configs, 1):
        print(f"\n--- Pipeline {i}/{len(pipeline_configs)}: {config_item['name']} ---")
        
        # Setup execution providers for this specific pipeline
        setup_pipeline_execution_provider(config_item, args, config)
        
        # Determine custom_op_path: command-line > per-pipeline config > global defaults > None
        custom_op_path = args.custom_op_path
        if not custom_op_path:
            custom_op_path = config_item.get('custom_op_path')
        if not custom_op_path:
            custom_op_path = config_defaults.get('custom_op_path')
        
        for model_id in config_item["model_ids"]:
            result = run_pipeline(config_item, model_id, paths['test_path'], paths['source_path'],
                                args.benchmark, args.timeout, args.prompt, 
                                args.prompt_file, getattr(args, 'sd3_controlnet_mode', 'text2img'), 
                                config.get('defaults', {}), log_buffer, determine_prompt_source,
                                args.image_only, args.traceback, custom_op_path)
            
            # Print immediate feedback
            if result['success']:
                output_dir = "generated_images/"
                print(f"[OK] SUCCESS in {result['duration']:.1f}s")
                print(f"   [Dir] Images saved to: {output_dir}")
            else:
                print(f"[FAILED] FAILED in {result['duration']:.1f}s - {result['error'][:100]}...")
            
            all_results.append(result)
    
    # Save log file if log buffer has content
    if log_buffer:
        mode_str = "BENCHMARK" if args.benchmark else "QUICK_TEST"
        log_file_path = save_log_file(log_buffer, mode_str, WORKSPACE_ROOT)
    
    return all_results, log_file_path


def main():
    """
    Main entry point for the pipeline runner script.
    
    Parses command-line arguments, loads configuration, executes selected pipelines,
    and provides comprehensive reporting of results. Supports both quick testing
    and benchmark modes with various filtering options.
    """
    parser = argparse.ArgumentParser(description="Run all pipeline scripts with profiling")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Enable benchmark mode (10 profiling rounds). Default is quick mode (1 round).")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout in seconds for each run (default: 600)")
    parser.add_argument("--script", type=str, nargs="*",
                       help="Run only these script names (e.g. --script run_sd.py run_sd_xl.py)")
    parser.add_argument("--model_id", type=str, nargs="*",
                       help="Run only these model IDs (e.g. --model_id runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--pipelines", type=str, nargs="*",
                       help="Run specific pipelines by name (e.g. --pipelines sd_15 sd3_base)")
    parser.add_argument("--pipeline_groups", type=str, nargs="*",
                       help="Run pipeline groups (e.g. --pipeline_groups all_sd3 quick_test)")
    parser.add_argument("--list", action="store_true",
                       help="List available pipelines and groups, then exit")
    parser.add_argument("--prompt", type=str,
                       help="Custom prompt to use for all pipelines")
    parser.add_argument("--prompt_file", type=str,
                       help="Path to JSON file containing list of prompts to run through each pipeline")
    parser.add_argument("--sd3-controlnet-mode", choices=["controlnet", "text2img"], default="controlnet",
                       help="For SD3 ControlNet with prompt files: 'controlnet' uses custom prompts with ControlNet enabled (recommended), 'text2img' converts to text-to-image mode without ControlNet guidance (default: controlnet)")
    parser.add_argument("--force-cpu", action="store_true",
                       help="Force CPU-only execution (disables DML and all accelerators). Overrides config setting.")
    parser.add_argument("--no-summary", action="store_true",
                       help="Skip printing the detailed summary report and skip saving results to disk")
    parser.add_argument("--config", type=str,
                       help="Path to custom YAML configuration file (default: config/pipeline_configs.yaml)")
    parser.add_argument("--clean-images", action="store_true",
                       help="Remove all existing images and Excel files from generated_images folder before running pipelines")
    parser.add_argument("--clean-results", action="store_true",
                       help="Remove all existing result files from results folder before running pipelines")
    parser.add_argument("--save-log", action="store_true",
                       help="Save complete log output to a text file when any pipeline has streaming output enabled")
    parser.add_argument("--image-only", action="store_true",
                       help="Generate images only without profiling (batch mode). Disables warmup and profiling rounds for fastest execution.")
    parser.add_argument("--traceback", action="store_true",
                       help="Include full traceback in error messages for detailed debugging")
    parser.add_argument("--custom-op-path", type=str,
                       help="Path to onnx_custom_ops.dll (Windows) or libonnx_custom_ops.so (Linux). If not specified, will use default locations.")
    
    args = parser.parse_args()
    
    # Validate prompt arguments
    if args.prompt and args.prompt_file:
        print("Error: Cannot specify both --prompt and --prompt_file. Please choose one.")
        return False
    
    try:
        config, paths = load_config(args.config, CONFIG_DIR, WORKSPACE_ROOT)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False
    
    # Setup execution providers based on config and command line args
    setup_execution_provider(args, config)
    
    if args.list:
        list_pipelines(config)
        return True
    
    # Get pipeline configurations
    pipeline_configs = get_pipeline_configs(config, paths, args)
    if not pipeline_configs:
        print("No pipelines match the criteria. Use --list to see available pipelines.")
        return False
    
    # Clean existing images if requested
    if args.clean_images:
        print(f"\n{'='*60}")
        print(f"Cleaning Generated Images & Excel Files")
        print(f"{'='*60}")
        removed_count = clean_generated_images(paths['test_path'])
        print(f"\n📊 Cleanup Summary: {removed_count} file(s) removed")
    
    # Clean existing results if requested
    if args.clean_results:
        print(f"\n{'='*60}")
        print(f"Cleaning Results Files")
        print(f"{'='*60}")
        results_path = WORKSPACE_ROOT / "results"
        removed_count = clean_results(results_path)
        print(f"\n[Stats] Cleanup Summary: {removed_count} result file(s) removed")
    
    # Run pipelines
    if args.image_only:
        mode_str = "IMAGE-ONLY"
    elif args.benchmark:
        mode_str = "BENCHMARK"
    else:
        mode_str = "QUICK TEST"
    print(f"\n{'='*60}")
    print(f"Running {len(pipeline_configs)} pipelines in {mode_str} mode")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    all_results, log_file_path = run_all_pipelines(pipeline_configs, paths, args, config)
    
    # Calculate and handle summary/results
    total_duration = datetime.now() - start_time
    if not args.no_summary:
        print_results_summary(all_results, mode_str, total_duration)
        results_file = save_results(all_results, mode_str, total_duration, args.benchmark, WORKSPACE_ROOT)
        print(f"\nResults saved to: {results_file}")
        if log_file_path:
            print(f"Complete log saved to: {log_file_path}")
    
    return all([r['success'] for r in all_results])


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
