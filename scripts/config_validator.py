#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

"""
Pipeline configuration argument validator.

Validates pipeline arguments from YAML config against the known set of valid
CLI arguments before subprocess launch, catching typos and invalid values early.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any

Logger = logging.getLogger(__name__)

# ============================================================================
# Valid argument definitions derived from runner_args.py parser
# ============================================================================

# Maps valid flag names to their expected type and constraints.
# type: "str", "int", "float", "bool" (store_true flags), "int_list", "float_list", "str_list"
VALID_ARGS: Dict[str, Dict[str, Any]] = {
    # Environment args
    "--model_path":         {"type": "str"},
    "--verbose":            {"type": "bool"},
    "-v":                   {"type": "bool", "canonical": "--verbose"},
    "--custom_op_path":     {"type": "str"},
    "--enable_compile":     {"type": "bool"},
    "-c":                   {"type": "bool", "canonical": "--enable_compile"},
    "--optimize_o1":        {"type": "bool"},
    "-O1":                  {"type": "bool", "canonical": "--optimize_o1"},

    # Model args
    "--model_id":           {"type": "str"},
    "--revision":           {"type": "str"},
    "--controlnet":         {"type": "str", "choices": [
        "canny", "tile", "pose", "depth", "union", "none",
        "outpainting", "removal", "inpainting",
    ]},
    "-C":                   {"type": "str", "canonical": "--controlnet"},

    # Runner args
    "--num_inference_steps": {"type": "int", "min": 1},
    "-n":                    {"type": "int", "min": 1, "canonical": "--num_inference_steps"},
    "--width":              {"type": "int", "min": 1},
    "-W":                   {"type": "int", "min": 1, "canonical": "--width"},
    "--height":             {"type": "int", "min": 1},
    "-H":                   {"type": "int", "min": 1, "canonical": "--height"},
    "--prompt":             {"type": "str"},
    "--n_prompt":           {"type": "str"},
    "--prompt_file_path":   {"type": "str"},
    "--prompt_2":           {"type": "str"},
    "--num_images_per_prompt": {"type": "int", "min": 1},
    "--guidance_scale":     {"type": "float", "min": 0.0},
    "--seed":               {"type": "int"},

    # ControlNet args
    "--controlnet_conditioning_scale": {"type": "float", "min": 0.0},
    "--control_image_path": {"type": "str"},
    "--control_mask_path":  {"type": "str"},
    "--edit_image_path":    {"type": "str_list"},
    "--image_pads":         {"type": "int_list"},

    # Profiling args
    "--enable_profile":     {"type": "bool"},
    "-p":                   {"type": "bool", "canonical": "--enable_profile"},
    "--no_excel":           {"type": "bool"},
    "--no_images":          {"type": "bool"},
    "--profiling_rounds":   {"type": "int", "min": 1},
    "-pr":                  {"type": "int", "min": 1, "canonical": "--profiling_rounds"},

    # Output args
    "--output_path":        {"type": "str"},

    # Dynamic shape args
    "--dynamic_shape":      {"type": "bool"},
    "-ds":                  {"type": "bool", "canonical": "--dynamic_shape"},
    "--dynamic_shape_file_path": {"type": "str"},

    # Advanced args
    "--strength":           {"type": "float", "min": 0.0, "max": 1.0},
    "--t5_sequence_len":    {"type": "int", "choices": [77, 83, 256]},
    "--tea_caching":        {"type": "bool"},
    "--tea_caching_delta":  {"type": "float"},
    "--tea_caching_coef":   {"type": "float_list"},
}

# Valid top-level keys in a pipeline definition (within pipeline_configs.yaml)
VALID_PIPELINE_KEYS: Set[str] = {
    "name", "model_ids", "args", "extra_args", "script",
    "prompt", "width", "height",
    "execution", "streaming_output", "image_only",
}

# Valid top-level keys in the YAML config root
VALID_CONFIG_KEYS: Set[str] = {
    "paths", "defaults", "pipelines", "pipeline_groups", "default_run",
}


# ============================================================================
# Validation Results
# ============================================================================

class ValidationResult:
    """Collects validation errors and warnings."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def log_results(self, pipeline_name: Optional[str] = None) -> None:
        """Log all warnings and errors."""
        prefix = f"[{pipeline_name}] " if pipeline_name else ""
        for w in self.warnings:
            Logger.warning(f"{prefix}{w}")
        for e in self.errors:
            Logger.error(f"{prefix}{e}")

    def __repr__(self) -> str:
        return f"ValidationResult(errors={len(self.errors)}, warnings={len(self.warnings)})"


# ============================================================================
# YAML Structure Validation
# ============================================================================

def validate_config_structure(config: Dict[str, Any]) -> ValidationResult:
    """Validate the top-level structure of pipeline_configs.yaml.

    Checks:
      - Required keys are present
      - No unknown top-level keys
      - 'pipelines' is a non-empty dict
      - Each pipeline has required fields
    """
    result = ValidationResult()

    # Check for unknown top-level keys
    for key in config:
        if key not in VALID_CONFIG_KEYS:
            result.add_warning(f"Unknown top-level config key: '{key}'")

    # Check required sections
    if "pipelines" not in config:
        result.add_error("Config missing required 'pipelines' section")
        return result

    if not isinstance(config["pipelines"], dict) or not config["pipelines"]:
        result.add_error("'pipelines' must be a non-empty dictionary")
        return result

    # Validate each pipeline definition
    for name, pipeline in config["pipelines"].items():
        if not isinstance(pipeline, dict):
            result.add_error(f"Pipeline '{name}': must be a dictionary")
            continue

        # Check for unknown pipeline keys
        for key in pipeline:
            if key not in VALID_PIPELINE_KEYS:
                result.add_warning(f"Pipeline '{name}': unknown key '{key}'")

        # Check required pipeline fields
        if "model_ids" not in pipeline:
            result.add_error(f"Pipeline '{name}': missing required 'model_ids'")
        elif not isinstance(pipeline["model_ids"], list) or not pipeline["model_ids"]:
            result.add_error(f"Pipeline '{name}': 'model_ids' must be a non-empty list")

        # Validate 'args' dict keys if present
        if "args" in pipeline and isinstance(pipeline["args"], dict):
            _validate_args_dict(pipeline["args"], name, result)

    # Validate pipeline_groups reference valid pipeline names
    if "pipeline_groups" in config and "pipelines" in config:
        valid_names = set(config["pipelines"].keys())
        for group_name, members in config.get("pipeline_groups", {}).items():
            if not isinstance(members, list):
                result.add_error(f"Pipeline group '{group_name}': must be a list")
                continue
            for member in members:
                if member not in valid_names:
                    result.add_warning(
                        f"Pipeline group '{group_name}': references unknown pipeline '{member}'"
                    )

    return result


def _validate_args_dict(args_dict: Dict[str, Any], pipeline_name: str,
                        result: ValidationResult) -> None:
    """Validate a pipeline's 'args' dictionary keys and value types."""
    for key, value in args_dict.items():
        flag = f"--{key}"
        if flag not in VALID_ARGS:
            result.add_warning(
                f"Pipeline '{pipeline_name}': unknown arg '{key}' "
                f"(will become '{flag}' -- not a recognized runner argument)"
            )
            continue

        spec = VALID_ARGS[flag]
        _validate_value(value, spec, key, pipeline_name, result)


# ============================================================================
# Arg-List Validation (extra_args after flattening)
# ============================================================================

def validate_extra_args(extra_args: List[str],
                        pipeline_name: Optional[str] = None) -> ValidationResult:
    """Validate a flattened extra_args list (e.g. ['--width', '512', '--verbose']).

    Checks:
      - All flags are recognized
      - Values have correct types and are within constraints
      - No duplicate flags
    """
    result = ValidationResult()
    seen_flags: Dict[str, int] = {}
    i = 0

    def _is_flag(token: str) -> bool:
        """Check if a token is a CLI flag (not a negative number)."""
        if not token.startswith("-"):
            return False
        # Negative numbers: -1, -1.0, -.5 etc. are values, not flags
        stripped = token.lstrip("-")
        if stripped and (stripped[0].isdigit() or stripped[0] == "."):
            return False
        return True

    while i < len(extra_args):
        token = extra_args[i]

        if not _is_flag(token):
            # Orphaned value without a preceding flag
            result.add_warning(
                f"Unexpected positional value '{token}' at position {i}"
            )
            i += 1
            continue

        # Check if this is a known flag
        if token not in VALID_ARGS:
            result.add_warning(f"Unknown argument: '{token}'")
            # Skip the next token if it looks like a value (not a flag)
            if i + 1 < len(extra_args) and not _is_flag(extra_args[i + 1]):
                i += 2
            else:
                i += 1
            continue

        spec = VALID_ARGS[token]
        canonical = spec.get("canonical", token)

        # Check duplicates
        if canonical in seen_flags:
            result.add_warning(f"Duplicate argument: '{token}' (already set)")
        seen_flags[canonical] = i

        # Boolean flags consume no value
        if spec["type"] == "bool":
            i += 1
            continue

        # Non-boolean flags require at least one value
        if i + 1 >= len(extra_args) or _is_flag(extra_args[i + 1]):
            result.add_error(f"Argument '{token}' requires a value")
            i += 1
            continue

        # List-valued flags (argparse nargs="+") consume every consecutive
        # value token, e.g. "--edit_image_path img1 img2".
        if str(spec["type"]).endswith("_list"):
            i += 1
            while i < len(extra_args) and not _is_flag(extra_args[i]):
                _validate_str_value(extra_args[i], spec, token, result)
                i += 1
        else:
            value_str = extra_args[i + 1]
            _validate_str_value(value_str, spec, token, result)
            i += 2

    return result


# ============================================================================
# Value Validation Helpers
# ============================================================================

def _validate_value(value: Any, spec: Dict[str, Any], key: str,
                    pipeline_name: str, result: ValidationResult) -> None:
    """Validate a typed value from YAML args dict."""
    expected_type = spec["type"]
    prefix = f"Pipeline '{pipeline_name}', arg '{key}'"

    if expected_type == "bool":
        if not isinstance(value, bool):
            result.add_error(f"{prefix}: expected boolean, got {type(value).__name__}")
        return

    if expected_type == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            result.add_error(f"{prefix}: expected integer, got {type(value).__name__} ({value!r})")
            return
        _check_numeric_constraints(value, spec, prefix, result)
        return

    if expected_type == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            result.add_error(f"{prefix}: expected number, got {type(value).__name__} ({value!r})")
            return
        _check_numeric_constraints(float(value), spec, prefix, result)
        return

    if expected_type == "str":
        if not isinstance(value, str):
            # YAML may parse unquoted values as other types
            result.add_warning(f"{prefix}: expected string, got {type(value).__name__} ({value!r})")
        # Check enum choices (case-insensitive)
        if "choices" in spec and isinstance(value, str):
            if value.lower() not in [str(c).lower() for c in spec["choices"]]:
                result.add_error(
                    f"{prefix}: invalid value '{value}'. "
                    f"Must be one of: {spec['choices']}"
                )
        return

    if expected_type in ("int_list", "float_list", "str_list"):
        if not isinstance(value, list):
            result.add_warning(f"{prefix}: expected list, got {type(value).__name__}")
        return


def _validate_str_value(value_str: str, spec: Dict[str, Any], flag: str,
                        result: ValidationResult) -> None:
    """Validate a string-encoded value from the extra_args list."""
    expected_type = spec["type"]

    if expected_type == "int":
        try:
            v = int(value_str)
        except ValueError:
            result.add_error(f"Argument '{flag}': expected integer, got '{value_str}'")
            return
        _check_numeric_constraints(v, spec, f"Argument '{flag}'", result)

    elif expected_type == "float":
        try:
            v = float(value_str)
        except ValueError:
            result.add_error(f"Argument '{flag}': expected number, got '{value_str}'")
            return
        _check_numeric_constraints(v, spec, f"Argument '{flag}'", result)

    elif expected_type == "str":
        if "choices" in spec:
            if value_str.lower() not in [str(c).lower() for c in spec["choices"]]:
                result.add_error(
                    f"Argument '{flag}': invalid value '{value_str}'. "
                    f"Must be one of: {spec['choices']}"
                )


def _check_numeric_constraints(value: Any, spec: Dict[str, Any],
                               prefix: str, result: ValidationResult) -> None:
    """Check min/max/choices for numeric values."""
    if "choices" in spec:
        if value not in spec["choices"]:
            result.add_error(
                f"{prefix}: value {value} not in allowed values {spec['choices']}"
            )
        return

    if "min" in spec and value < spec["min"]:
        result.add_error(f"{prefix}: value {value} is below minimum {spec['min']}")
    if "max" in spec and value > spec["max"]:
        result.add_error(f"{prefix}: value {value} is above maximum {spec['max']}")
