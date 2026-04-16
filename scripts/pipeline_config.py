"""
Pipeline configuration and prompt handling utilities.

Handles YAML configuration loading, argument processing, pipeline selection,
and prompt management from files, command line, and configuration.
"""

import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def load_config(config_file: Optional[str] = None, config_dir: Optional[Path] = None, workspace_root: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    """
    Load the YAML configuration file and set up key file paths.
    
    Args:
        config_file (str, optional): Path to custom configuration file. 
                                   If None, uses default 'pipeline_configs.yaml'
        config_dir (Path): Path to config directory
        workspace_root (Path): Path to workspace root
    
    Returns:
        Tuple[Dict[str, Any], Dict[str, Path]]: A tuple containing:
            - config: The parsed YAML configuration dictionary
            - paths: Dictionary with paths to source, models, test scripts, etc.
    
    Raises:
        FileNotFoundError: If the configuration file is not found
    """
    if config_file:
        # Use the path exactly as provided by the user
        config_path = Path(config_file)
    else:
        if not config_dir:
            raise ValueError("config_dir must be provided")
        config_path = config_dir / "pipeline_configs.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Get project paths
    paths_config = config.get('paths', {})
    source_path = workspace_root
    
    # Determine config directory path
    if config_dir:
        resolved_config_path = config_dir
    elif workspace_root:
        resolved_config_path = workspace_root / 'config'
    else:
        resolved_config_path = Path('config')
    
    paths = {
        'source_path': source_path,
        'models_path': source_path / paths_config.get('models_path', 'models'),
        'test_path': source_path / paths_config.get('test_path', 'test'),
        'config_path': resolved_config_path,
        'control_images_path': Path(paths_config['control_images_path']) if 'control_images_path' in paths_config else None,
        'config_files_path': Path(paths_config['config_files_path']) if 'config_files_path' in paths_config else None
    }
    
    return config, paths


def resolve_model_path(arg: str, models_path: Path) -> str:
    """
    Resolve model path argument to absolute or relative path.
    
    Args:
        arg: Model path argument value
        models_path: Base models directory path
        
    Returns:
        Resolved path as string
    """
    # Don't modify paths that start with .. (already relative to test directory)
    # These paths are used as-is since scripts run from test directory
    if arg.startswith('..') or Path(arg).is_absolute():
        return arg
    # Only resolve simple names like "sd_turbo" to full model path
    return str(models_path / arg)


def resolve_prompt_shorthand(arg: str, prompts_dict: Dict[str, str]) -> str:
    """
    Resolve prompt shorthand references to full prompt text.
    
    Args:
        arg: Prompt argument (may be shorthand like 'medieval' or full text)
        prompts_dict: Dictionary of prompt shortcuts
        
    Returns:
        Resolved prompt text
    """
    return prompts_dict.get(arg, arg)


def process_extra_args(extra_args: List[str], config: Dict[str, Any], paths: Dict[str, Path]) -> List[str]:
    """
    Process command-line arguments to resolve references to config defaults.
    
    Converts shorthand references (like 'medieval' for prompts) to actual values and handles
    model path resolution for absolute vs relative paths.
    
    Args:
        extra_args (List[str]): Raw argument list from pipeline configuration
        config (Dict[str, Any]): Full configuration dictionary with defaults
        paths (Dict[str, Path]): Dictionary of important paths (source, models, etc.)
        
    Returns:
        List[str]: Processed argument list with all references resolved
    """
    processed_args = []
    defaults = config.get('defaults', {})
    
    # Map argument names to their value transformers
    arg_transformers = {
        "--model_path": lambda val: resolve_model_path(val, paths['models_path']),
        "--prompt": lambda val: resolve_prompt_shorthand(val, defaults.get('prompts', {}))
    }
    
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        
        if arg in arg_transformers and i + 1 < len(extra_args):
            processed_args.append(arg)
            processed_args.append(arg_transformers[arg](extra_args[i + 1]))
            i += 2
        else:
            processed_args.append(arg)
            i += 1
    
    return processed_args


def get_pipeline_configs(config: Dict[str, Any], paths: Dict[str, Path], 
                        args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Determine which pipeline configurations to run based on command-line arguments.
    
    Handles pipeline filtering by name, group, script, or model ID and returns
    a list of resolved pipeline configurations ready for execution.
    
    Args:
        config (Dict[str, Any]): Full YAML configuration dictionary
        paths (Dict[str, Path]): Dictionary of important file/directory paths
        args (argparse.Namespace): Parsed command-line arguments
        
    Returns:
        List[Dict[str, Any]]: List of pipeline configurations to execute
        
    Raises:
        ValueError: If no pipelines are selected or if invalid pipeline names/groups are specified
    """
    pipeline_defs = config.get('pipelines', {})
    selected_pipelines = []
    
    # Add pipelines from groups
    if args.pipeline_groups:
        groups = config.get('pipeline_groups', {})
        for group in args.pipeline_groups:
            if group in groups:
                selected_pipelines.extend(groups[group])
            else:
                print(f"Warning: Pipeline group '{group}' not found")
    
    # Add specific pipelines
    if args.pipelines:
        selected_pipelines.extend(args.pipelines)
    
    # Use defaults if nothing specified
    if not selected_pipelines:
        selected_pipelines = config.get('default_run', list(pipeline_defs.keys()))
    
    # Build configurations
    configs = []
    # FIXED: Preserve order while removing duplicates (dict.fromkeys maintains insertion order in Python 3.7+)
    # Changed from set() which lost original YAML ordering
    for name in dict.fromkeys(selected_pipelines):
        if name not in pipeline_defs:
            print(f"Warning: Pipeline '{name}' not found")
            continue
        
        pipeline = pipeline_defs[name]
        # Copy pipeline config and add processed extra_args
        pipeline_config = pipeline.copy()
        pipeline_config["extra_args"] = process_extra_args(pipeline.get('extra_args', []), config, paths)
        
        configs.append(pipeline_config)
    
    # Filter by script if specified
    if args.script:
        configs = [cfg for cfg in configs if cfg["script"] in args.script]
    
    return configs
