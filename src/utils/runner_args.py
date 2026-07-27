#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import argparse
import os
import sys
import logging as Logger
import platform
from .common import resolve_project_path, resolve_root_model_path
from .model_config import InferenceConfig, detect_model_spec

# pr-231: shown when the 'lib' directory (or one of its subfolders) is missing
# from the project root -- it isn't shipped in git and must be copied in from
# the Ryzen AI installation.
_LIB_DIR_HINT = (
    "The 'lib' directory is missing from the sd-sandbox project root.\n"
    "Copy it from your Ryzen AI installation, e.g.:\n"
    r'  Copy-Item "C:\Program Files\RyzenAI\1.8.0\GenAI-SD\lib" "<sd-sandbox_dir>\lib" -Recurse' "\n"
    "If Ryzen AI was installed to a different location, copy the 'lib' directory "
    "from '<RyzenAI_install_dir>\\GenAI-SD\\lib' instead."
)

parser = argparse.ArgumentParser(description="Pipeline Config")
# Env related args
parser.add_argument("--model_path", type=str, default=None, help="Path to SD/SD3 models")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output."
)
parser.add_argument("--custom_op_path", type=str, help="Path to ONNX custom ops DLL")
parser.add_argument(
    "-c", "--enable_compile", action="store_true", help="Enable compile fusion runtime."
)
parser.add_argument(
    "-O1",
    "--optimize_o1",
    action="store_true",
    help="Enable O1 preset optimizations.",
)
# model related args
parser.add_argument(
    "--model_id",
    type=str,
    default="runwayml/stable-diffusion-v1-5",
    help="Choose SD models type, \
        e.g., 'runwayml/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-2-1-base', \
            'stabilityai/sd-turbo', 'stabilityai/stable-diffusion-2-1', 'stabilityai/sdxl-turbo', \
            'stabilityai/stable-diffusion-xl-base-1.0', 'stabilityai/stable-diffusion-3-medium-diffuser', \
            'stabilityai/stable-diffusion-3.5-medium',             'black-forest-labs/FLUX.1-schnell', \
            'black-forest-labs/FLUX.1-dev', \
            'black-forest-labs/FLUX.2-klein-4B', \
            'stabilityai/stable-diffusion-3.5-medium', \
            'playgroundai/playground-v2.5-1024px-aesthetic', \
            '/Lykon/dreamshaper-xl-lightning', 'segmind/SSD-1B'",
)
parser.add_argument(
    "--revision",
    type=str,
    default=None,
    help="Git branch, tag, or commit hash for Hugging Face model (e.g., '1.7.0', 'main', 'v1.0.0'). Default: None (use default branch)",
)
parser.add_argument(
    "-C",
    "--controlnet",
    type=str,
    help="Canny, Tile, Pose, Depth, OutPainting, Removal, InPainting, or None",
)

# runner  related args
parser.add_argument(
    "-n",
    "--num_inference_steps",
    type=int,
    help="The number of denoising steps.",
)
parser.add_argument(
    "-W",
    "--width",
    type=int,
    help="The width in pixels of the generated image.",
)
parser.add_argument(
    "-H",
    "--height",
    type=int,
    help="The height in pixels of the generated image.",
)
parser.add_argument(
    "--prompt",
    type=str,
    help="Text description for generating images.",
)
parser.add_argument(
    "--n_prompt",
    type=str,
    help="Text description of what to avoid in the generated images.",
)
parser.add_argument(
    "--prompt_file_path",
    type=str,
    nargs="?",
    const="",
    help="Path to the prompt configuration file.",
)
# controlnet related args
parser.add_argument(
    "--controlnet_conditioning_scale",
    type=float,
    help="The scale of the controlnet conditioning.",
)
parser.add_argument(
    "--control_image_path",
    type=str,
    help="Path to controlnet image",
)

parser.add_argument(
    "--control_mask_path",
    type=str,
    help="Path to controlnet mask image",
)

parser.add_argument(
    "--edit_image_path",
    type=str,
    nargs="+",
    default=None,
    help="Reference image path(s) for FLUX.2-klein image-to-image or multi-image edit mode.",
)

parser.add_argument(
    "--image_pads",
    type=int,
    nargs="+",
    default=[10, 10, 10, 10],
    help="Image padding values. Provide four integers: [top, bottom, left, right] to control padding \
        for each side of the image, only used in Outpainting.",
)
parser.add_argument(
    "--num_images_per_prompt", type=int, help="Number of images per prompt."
)
parser.add_argument("--guidance_scale", type=float, help="The value of guidance_scale.")
parser.add_argument("--seed", type=int, default=None, help="The Random Generator seed")
# profiling args
parser.add_argument(
    "-p",
    "--enable_profile",
    action="store_true",
    help="Enable profiling measurement.",
)
# ADDED: Flag to disable Excel file generation for profiling (keeps results directory clean)
parser.add_argument(
    "--no_excel",
    action="store_true",
    help="Disable Excel metrics file creation (profiling metrics still printed to stdout).",
)
# ADDED: Flag to disable image generation (useful for pure profiling/benchmarking runs)
parser.add_argument(
    "--no_images",
    action="store_true",
    help="Disable image file generation (only run pipeline for profiling/benchmarking).",
)
parser.add_argument(
    "--t5_sequence_len",
    type=int,
    choices=[77, 83, 256],
    help="T5 sequence length (77, 83, or 256 for Flux)",
)
parser.add_argument(
    "--prompt_2",
    type=str,
    help="Second prompt for dual text encoder models (e.g., Flux T5 encoder)",
)
parser.add_argument(
    "-pr",
    "--profiling_rounds",
    type=int,
    default=4,
    help="Number of profiling rounds",
)
# result args
parser.add_argument(
    "--output_path",
    type=str,
    default="generated_images",
    help="output image path",
)

# dynamic shape
parser.add_argument(
    "-ds", "--dynamic_shape", action="store_true", help="Enable dynamic shape flow."
)
# IMPORTANT: Default is None (not a JSON file path) to support two distinct use cases:
# 1. --dynamic_shape with --width/--height: Use specified resolution with dynamic shape models
# 2. --dynamic_shape with --dynamic_shape_file_path: Iterate through all resolutions in JSON file
# Previously defaulted to "config/dynamic_shape_8x.json" which always iterated through all shapes,
# preventing single-resolution testing with dynamic models. Setting default=None allows the
# run_sd3.py script to differentiate between these two modes based on whether the file path is provided.
parser.add_argument(
    "--dynamic_shape_file_path",
    type=str,
    nargs="?",
    const="",
    default=None,
    help="Path to the dynamic shape configuration file.",
)

parser.add_argument(
    "--strength",
    type=float,
    default=None,
    help="indicates how much to transform the reference image in segmind-vega image to image pipeline. \
        A value of 1.0 essentially ignores image",
)

parser.add_argument(
    "--tea_caching",
    action="store_true",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--tea_caching_delta",
    type=float,
    default=None,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--tea_caching_coef",
    type=float,
    nargs="+",
    default=None,
    help=argparse.SUPPRESS,
)

# ----------------------------------------------------------------------------------------------------------------------
def check_sd3_normal_args(args):
    controlnet_type = args.controlnet.lower()
    VALID_CONTROLNET_VALUES = [
        "canny",
        "tile",
        "pose",
        "depth",
        "union",
        "none",
        "outpainting",
        "removal",
        "inpainting",
    ]
    if controlnet_type not in VALID_CONTROLNET_VALUES:
        Logger.warning(
            f"Invalid value for --controlnet/-C: '{args.controlnet}'. "
            f"Allowed values are: {', '.join(VALID_CONTROLNET_VALUES)}. "
            "Use 'None' for text-to-image mode without ControlNet."
        )
        while True:
            try:
                choice = (
                    input(
                        "Do you want to continue by running with 'Canny' (Y) or exit (N)? [Y/N]: "
                    )
                    .strip()
                    .lower()
                )
                if choice == "y":
                    Logger.info(
                        "User chose to continue. Proceeding with ControlNet type 'Canny'."
                    )
                    args.controlnet = "Canny"  # Modify args directly
                    break
                elif choice == "n":
                    Logger.info("User chose to exit. Exiting script.")
                    sys.exit(0)
                else:
                    Logger.warning("Invalid input. Please enter 'Y' or 'N'.")
            except EOFError:  # Handle cases where input stream is closed (e.g. piping)
                Logger.error("No input received. Exiting.")
                sys.exit(1)
    else:
        Logger.info(f"ControlNet type from args: {args.controlnet}")
    return args


def _apply_spec_defaults(args, cfg: InferenceConfig) -> None:
    """Fill unset ``args`` fields from ``InferenceConfig`` defaults.

    CLI-supplied values always take precedence; a field is only filled when it is
    ``None`` (or empty string for ``prompt`` / ``n_prompt``).
    """
    def _set(attr, val):
        if val is not None and getattr(args, attr, None) is None:
            setattr(args, attr, val)

    _set("height",                       cfg.height)
    _set("width",                        cfg.width)
    _set("t5_sequence_len",              cfg.t5_sequence_len)
    _set("num_inference_steps",          cfg.num_inference_steps)
    _set("num_images_per_prompt",        cfg.num_images_per_prompt)
    _set("guidance_scale",               cfg.guidance_scale)
    _set("seed",                         cfg.seed)
    _set("strength",                     cfg.strength)
    _set("controlnet_conditioning_scale", cfg.controlnet_conditioning_scale)
    _set("control_image_path",           cfg.control_image_path)
    _set("control_mask_path",            cfg.control_mask_path)

    if not getattr(args, "prompt", None) and not getattr(args, "prompt_file_path", None):
        if cfg.prompt:
            args.prompt = cfg.prompt
        elif cfg.prompt_file_path:
            args.prompt_file_path = cfg.prompt_file_path

    if not getattr(args, "n_prompt", None):
        args.n_prompt = cfg.n_prompt or ""

    if args.optimize_o1 and cfg.tea_caching_delta is not None and cfg.tea_caching_coef is not None:
        controlnet_none = args.controlnet is None or (args.controlnet is not None and args.controlnet.lower() == "none")
        if args.num_inference_steps <= 4 and controlnet_none:
            args.tea_caching = False
            _set("tea_caching_delta", 0.1)
            _set("tea_caching_coef", [1.0, 0.0])
            Logger.info("Disable tea caching for O1 when ControlNet=None and num_inference_steps<=4.")
        else:
            args.tea_caching = True
            _set("tea_caching_delta", cfg.tea_caching_delta)
            _set("tea_caching_coef", cfg.tea_caching_coef)
            Logger.info(
                "O1 preset applied: tea_caching=%s, tea_caching_delta=%s, tea_caching_coef=%s",
                args.tea_caching,
                args.tea_caching_delta,
                args.tea_caching_coef,
            )

    if not getattr(args, "common_model_path", None):
        args.common_model_path = cfg.common_model_path
    if not getattr(args, "sub_model_path", None):
        args.sub_model_path = cfg.sub_model_path


def _validate_control_paths(args) -> None:
    """Raise ``EnvironmentError`` when a declared control-image or mask path is missing."""
    for attr, label in (
        ("control_image_path", "control image"),
        ("control_mask_path",  "control mask"),
    ):
        path = getattr(args, attr, None)
        if path and "http" not in path and not os.path.exists(path):
            raise EnvironmentError(f"Can't find {label}: {path}")

    edit_paths = getattr(args, "edit_image_path", None) or []
    for path in edit_paths:
        if path and "http" not in path and not os.path.exists(path):
            raise EnvironmentError(f"Can't find edit image: {path}")


def check_args(args):
    if args.verbose:
        Logger.basicConfig(
            level=Logger.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
        )
    else:
        Logger.basicConfig(
            level=Logger.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
        )


    if not args.model_id:
        Logger.warning("model_id is not set, using built-in defaults.")
        _apply_spec_defaults(args, InferenceConfig())
    else:
        mid = args.model_id.lower()

        try:
            spec = detect_model_spec(args.model_id, getattr(args, "controlnet", None))
            _apply_spec_defaults(args, spec.defaults)
        except ValueError:
            Logger.warning(
                "No ModelSpec matched model_id=%r; using built-in defaults.", args.model_id
            )
            _apply_spec_defaults(args, InferenceConfig())

    if not hasattr(args, "sub_model_path"):
        args.sub_model_path = "normal/"
    if not hasattr(args, "common_model_path"):
        args.common_model_path = "common/"

    _validate_control_paths(args)

    if not args.model_id:
        # Skip model-path / env validation when no model is configured.
        return args

    if args.tea_caching_coef is None:
        args.tea_caching_coef = [1.0, 0.0]

    if not args.model_path:
        # No path given at all — auto-download from Hugging Face.
        args.model_path = resolve_root_model_path(args.model_id, None, args.revision)
    else:
        args.model_path = os.path.abspath(args.model_path)
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(
                f"model_path '{args.model_path}' does not exist. "
                f"Pass the correct path with --model_path, or omit it to auto-download."
            )
    

    system = platform.system()
    share_obj_name = ""
    if system == "Windows":
        share_obj_name = "onnxruntime_providers_ryzenai.dll"
    elif system == "Linux":
        share_obj_name = "libonnxruntime_providers_ryzenai.so"

    if not args.custom_op_path:
        onnx_utils_root = os.environ.get("ONNX_UTILS_ROOT")
        ryzen_root = os.environ.get("RYZEN_AI_INSTALLATION_PATH")
        candidates = []

        # 1. ONNX_UTILS_ROOT/build/install/bin
        if onnx_utils_root:
            candidates.append(
                os.path.join(onnx_utils_root, "build", "install", "bin", share_obj_name)
            )
        # 2. RYZEN_AI_INSTALLATION_PATH/deployment
        if ryzen_root:
            candidates.append(
                os.path.join(ryzen_root, "deployment", share_obj_name)
            )
        # 3. project local lib fallback  # pr-231: uncommented -- enables lib/ EP DLL discovery
        candidates.append(
            resolve_project_path(os.path.join("lib", share_obj_name))
        )
        # pick first existing
        args.custom_op_path = next(
            (p for p in candidates if os.path.exists(p)),
            None
        )

    if not args.custom_op_path or not os.path.exists(args.custom_op_path):
        raise EnvironmentError(
            f"Can't find onnx custom op plugin: {share_obj_name}\n"
            f"Searched paths:\n  " + "\n  ".join(candidates) + "\n" + _LIB_DIR_HINT
        )

    mid = args.model_id.lower()
    is_sd3_model = (
        "stable-diffusion-3" in mid
        or "stable-diffusion-3.5" in mid
        or "stable-diffusion-3-5" in mid
    )
    # pr-231: FLUX models also need DD_PLUGINS_ROOT for NPU transactions
    is_flux_model = "flux" in mid
    if is_sd3_model or is_flux_model:
        if not os.environ.get("DD_PLUGINS_ROOT"):
            os.environ["DD_PLUGINS_ROOT"] = resolve_project_path("lib/transaction/stx")
        dd_plugins_path = os.environ["DD_PLUGINS_ROOT"]
        Logger.debug(f"DD_PLUGINS_ROOT set to: {dd_plugins_path}")
        if not os.path.exists(dd_plugins_path):
            raise EnvironmentError(
                f"invalid DD_PLUGINS_ROOT {os.environ['DD_PLUGINS_ROOT']}\n" + _LIB_DIR_HINT
            )

    if "DD_ROOT" not in os.environ:
        os.environ["DD_ROOT"] = resolve_project_path("lib")
    if not os.path.exists(os.environ["DD_ROOT"]):
        raise EnvironmentError(f"invalid DD_ROOT {os.environ['DD_ROOT']}\n" + _LIB_DIR_HINT)

    if args.enable_profile:
        if args.profiling_rounds <= 0:
            raise ValueError(
                f"profiling rounds should be > 0, now is {args.profiling_rounds}"
            )

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    return args
