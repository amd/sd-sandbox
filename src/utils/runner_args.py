#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

import argparse
import os
import sys
import logging as Logger
import platform
from src.utils.common import get_absolute_path

parser = argparse.ArgumentParser(description="Pipeline Config")
# Env related args
parser.add_argument("--root_path", type=str, default="..", help="Path to SD/SD3 models")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output."
)
parser.add_argument("--custom_op_path", type=str, help="Path to ONNX custom ops DLL")
parser.add_argument(
    "--gpu", help="Running with DynamicDispatch or GPU", action="store_true"
)
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
            'stabilityai/stable-diffusion-3.5-medium', 'black-forest-labs/FLUX.1-schnell', \
            'black-forest-labs/FLUX.1-dev', \
            'stabilityai/stable-diffusion-3.5-medium', \
            'playgroundai/playground-v2.5-1024px-aesthetic', \
            '/Lykon/dreamshaper-xl-lightning', 'segmind/SSD-1B',\
            'Kwai-Kolors/Kolors'",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to SD models",
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
    default=83,
    choices=[77, 83],
    help="T5 sequence length (77 or 83)",
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


def config_sd3_controlnet_args(args):
    # outpainting
    if args.controlnet.lower() == "outpainting":
        args.height = args.height or 1024
        args.width = args.width or 1024
        if not args.height == 1024 and not args.width == 1024:
            raise ValueError("Height and width must be 1024")
        args.sub_model_path = "normal/"
        args.prompt = args.prompt or "background, nothing, blank"
        args.n_prompt = args.n_prompt or ""
        args.seed = args.seed if args.seed is not None else 0
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.95
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 5.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/outpainting.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # removal
    elif args.controlnet.lower() == "removal":
        args.height = args.height or 1024
        args.width = args.width or 1024
        if not args.height == 1024 and not args.width == 1024:
            raise ValueError("Height and width must be 1024")
        args.sub_model_path = "normal/"
        args.prompt = args.prompt or "background, nothing, blank"
        args.n_prompt = args.n_prompt or ""
        args.seed = args.seed if args.seed is not None else 0
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.95
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 5.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/removal/origin.jpg")
        args.control_mask_path = args.control_mask_path or get_absolute_path("test/ref/removal/mask.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )
        if "http" not in args.control_mask_path and not os.path.exists(args.control_mask_path):
            raise EnvironmentError(
                f"can't find control image mask path  {args.control_mask_path}"
            )

    # inpainting
    elif args.controlnet.lower() == "inpainting":
        args.height = args.height or 1024
        args.width = args.width or 1024
        if not args.height == 1024 and not args.width == 1024:
            raise ValueError("Height and width must be 1024")
        args.sub_model_path = "inpainting/"
        args.prompt = args.prompt or "holiday"
        args.n_prompt = args.n_prompt or ""
        args.seed = args.seed if args.seed is not None else 0
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.7
        )
        # 7.0 matches official SD3 ControlNet Inpainting example; higher scale helps prompt (e.g. cat) appear in mask
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/inpainting/origin.jpg")
        args.control_mask_path = args.control_mask_path or get_absolute_path("test/ref/inpainting/mask.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )
        if "http" not in args.control_mask_path and not os.path.exists(args.control_mask_path):
            raise EnvironmentError(
                f"can't find control image mask path  {args.control_mask_path}"
            )

    # canny
    elif args.controlnet.lower() == "canny":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        # Set prompt file path if neither prompt nor prompt file is specified
        if not args.prompt and not args.prompt_file_path:
            args.prompt_file_path = get_absolute_path("config/prompts_config.json")
        # Only set default prompt if no prompt and no prompt file
        if not args.prompt and not args.prompt_file_path:
            args.prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        args.n_prompt = args.n_prompt or "NSFW, nude, naked, porn, ugly"
        args.seed = args.seed if args.seed is not None else 42
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.5
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/canny.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # tile
    elif args.controlnet.lower() == "tile":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        # Set prompt file path if neither prompt nor prompt file is specified
        if not args.prompt and not args.prompt_file_path:
            args.prompt_file_path = get_absolute_path("config/prompts_config.json")
        # Only set default prompt if no prompt and no prompt file
        if not args.prompt and not args.prompt_file_path:
            args.prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        args.n_prompt = args.n_prompt or "NSFW, nude, naked, porn, ugly"
        args.seed = args.seed if args.seed is not None else 42
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.5
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/tile.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # pose
    elif args.controlnet.lower() == "pose":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        # Set prompt file path if neither prompt nor prompt file is specified
        if not args.prompt and not args.prompt_file_path:
            args.prompt_file_path = get_absolute_path("config/prompts_config.json")
        # Only set default prompt if no prompt and no prompt file
        if not args.prompt and not args.prompt_file_path:
            args.prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        args.n_prompt = args.n_prompt or "NSFW, nude, naked, porn, ugly"
        args.seed = args.seed if args.seed is not None else 42
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.5
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/pose.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # depth
    elif args.controlnet.lower() == "depth":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        args.prompt = args.prompt or "a panda cub, captured in a close-up, in forest, is perched on a tree trunk. good composition, Photography, the cub's ears, a fluffy black, are tucked behind its head, adding a touch of whimsy to its appearance. a lush tapestry of green leaves in the background. depth of field, National Geographic"
        args.n_prompt = args.n_prompt or "bad hands, blurry, NSFW, nude, naked, porn, ugly, bad quality, worst quality"
        args.seed = args.seed if args.seed is not None else 42
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.5
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/depth.jpeg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )
    # union
    elif args.controlnet.lower() == "union":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        args.prompt = args.prompt or "a lovely dog"
        args.n_prompt = args.n_prompt or ""
        args.seed = args.seed if args.seed is not None else 0
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.7
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 3.5
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/union.jfif")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # t2i flow
    elif args.controlnet.lower() == "none":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        args.prompt = (
            args.prompt
            or "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, \
            cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, \
                extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, \
                    intricate artwork by john william turner"
        )
        if not args.prompt and not args.prompt_file_path:
            args.prompt_file_path = get_absolute_path("config/prompts_config.json")
        args.n_prompt = args.n_prompt or "NSFW, nude, naked, porn, ugly"
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 3.0
        )
        args.seed = args.seed if args.seed is not None else 0


def config_sd35_controlnet_args(args):
    # outpainting
    if args.controlnet.lower() == "outpainting":
        args.height = args.height or 1024
        args.width = args.width or 1024
        if not args.height == 1024 and not args.width == 1024:
            raise ValueError("Height and width must be 1024")
        args.sub_model_path = "normal/"
        args.prompt = args.prompt or "background, nothing, blank"
        args.n_prompt = args.n_prompt or ""
        args.seed = args.seed or 0
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.95
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 5.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/outpainting.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # removal
    elif args.controlnet.lower() == "removal":
        args.height = args.height or 1024
        args.width = args.width or 1024
        if not args.height == 1024 and not args.width == 1024:
            raise ValueError("Height and width must be 1024")
        args.sub_model_path = "normal/"
        args.prompt = args.prompt or "background, nothing, blank"
        args.n_prompt = args.n_prompt or ""
        args.seed = args.seed if args.seed is not None else 0
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.95
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 5.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/removal/origin.jpg")
        args.control_mask_path = args.control_mask_path or get_absolute_path("test/ref/removal/mask.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )
        if "http" not in args.control_mask_path and not os.path.exists(args.control_mask_path):
            raise EnvironmentError(
                f"can't find control image mask path  {args.control_mask_path}"
            )

    # inpainting
    elif args.controlnet.lower() == "inpainting":
        args.height = args.height or 1024
        args.width = args.width or 1024
        if not args.height == 1024 and not args.width == 1024:
            raise ValueError("Height and width must be 1024")
        args.sub_model_path = "inpainting/"
        args.prompt = args.prompt or "holiday"
        args.n_prompt = args.n_prompt or ""
        args.seed = args.seed if args.seed is not None else 0
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.7
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 5.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/inpainting/origin.jpg")
        args.control_mask_path = args.control_mask_path or get_absolute_path("test/ref/inpainting/mask.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )
        if "http" not in args.control_mask_path and not os.path.exists(args.control_mask_path):
            raise EnvironmentError(
                f"can't find control image mask path  {args.control_mask_path}"
            )
    # canny
    elif args.controlnet.lower() == "canny":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        # Set prompt file path if neither prompt nor prompt file is specified
        if not args.prompt and not args.prompt_file_path:
            args.prompt_file_path = get_absolute_path("config/prompts_config.json")
        # Only set default prompt if no prompt and no prompt file
        if not args.prompt and not args.prompt_file_path:
            args.prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        args.n_prompt = args.n_prompt or "NSFW, nude, naked, porn, ugly"
        args.seed = args.seed if args.seed is not None else 42
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.7
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 5.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/canny.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # tile
    elif args.controlnet.lower() == "tile":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        # Set prompt file path if neither prompt nor prompt file is specified
        if not args.prompt and not args.prompt_file_path:
            args.prompt_file_path = get_absolute_path("config/prompts_config.json")
        # Only set default prompt if no prompt and no prompt file
        if not args.prompt and not args.prompt_file_path:
            args.prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        args.n_prompt = args.n_prompt or "NSFW, nude, naked, porn, ugly"
        args.seed = args.seed if args.seed is not None else 42
        args.controlnet_conditioning_scale = (
            args.controlnet_conditioning_scale
            if args.controlnet_conditioning_scale is not None
            else 0.7
        )
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 5.0
        )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/tile.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # pose
    elif args.controlnet.lower() == "pose":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        # Set prompt file path if neither prompt nor prompt file is specified
        if not args.prompt and not args.prompt_file_path:
            args.prompt_file_path = get_absolute_path("config/prompts_config.json")
        # Only set default prompt if no prompt and no prompt file
        if not args.prompt and not args.prompt_file_path:
            args.prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        args.n_prompt = args.n_prompt or "NSFW, nude, naked, porn, ugly"
        args.seed = args.seed if args.seed is not None else 42
        if args.height == 512:
            args.controlnet_conditioning_scale = (
                args.controlnet_conditioning_scale
                if args.controlnet_conditioning_scale is not None
                else 0.8
            )
            args.guidance_scale = (
                args.guidance_scale if args.guidance_scale is not None else 5.0
            )
        else:
            args.controlnet_conditioning_scale = (
                args.controlnet_conditioning_scale
                if args.controlnet_conditioning_scale is not None
                else 0.7
            )
            args.guidance_scale = (
                args.guidance_scale if args.guidance_scale is not None else 5.0
            )
        args.control_image_path = args.control_image_path or get_absolute_path("test/ref/pose.jpg")
        if "http" not in args.control_image_path and not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # t2i flow
    elif args.controlnet.lower() == "none":
        args.height = args.height or 512
        args.width = args.width or 512
        args.sub_model_path = "normal/"
        args.common_model_path = "common/"
        args.prompt = args.prompt or "A capybara holding a sign that reads Hello World"
        if not args.prompt and not args.prompt_file_path:
            args.prompt_file_path = get_absolute_path("config/prompts_config.json")
        args.n_prompt = ""
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 4.5
        )
        args.seed = args.seed if args.seed is not None else 45


def check_args(args):
    if args.verbose:
        Logger.basicConfig(
            level=Logger.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
        )
    else:
        Logger.basicConfig(
            level=Logger.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
        )

    if not os.path.exists(args.root_path):
        raise EnvironmentError(f"root_path {args.root_path} is not invalid")

    # Set defaults for sub_model_path and common_model_path if not already set
    if not hasattr(args, 'sub_model_path'):
        args.sub_model_path = "normal/"
    if not hasattr(args, 'common_model_path'):
        args.common_model_path = "common/"

    if not args.model_id:
        Logger.warning("model_id is not set.")
        args.height = args.height or 512
        args.width = args.width or 512
        args.num_inference_steps = args.num_inference_steps or 20
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = (
            args.prompt
            or "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, \
                        battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, \
                        unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner."
        )
        args.n_prompt = args.n_prompt or ""
    elif "stable-diffusion-v1-5" in args.model_id or "stable-diffusion-1.5" in args.model_id or "sd15-controlnet" in args.model_id:
        args.height = args.height or 512
        args.width = args.width or 512
        args.num_inference_steps = args.num_inference_steps or 20
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.n_prompt = args.n_prompt or ""
        args.controlnet = args.controlnet or "none"
        if args.controlnet.lower() == "canny":
            args.controlnet_conditioning_scale = (
                args.controlnet_conditioning_scale
                if args.controlnet_conditioning_scale is not None
                else 1.5
            )
            args.prompt = args.prompt or "a blue paraidse bird in the jungle"
            args.control_image_path = args.control_image_path or get_absolute_path("test/ref/control.png")
        elif args.controlnet.lower() == "none":
            args.prompt = (
                args.prompt
                or "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, \
                            battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, \
                            unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner."
            )

    elif "sd-turbo" in args.model_id:
        args.height = args.height or 512
        args.width = args.width or 512
        args.num_inference_steps = args.num_inference_steps or 1
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 0.0
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = (
            args.prompt
            or "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
        )
        args.n_prompt = args.n_prompt or ""

    elif "stable-diffusion-2-1-base" in args.model_id:
        args.height = args.height or 512
        args.width = args.width or 512
        args.num_inference_steps = args.num_inference_steps or 50
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = args.prompt or "A photo of an astronaut riding a horse on mars."

    elif "stable-diffusion-2-1" in args.model_id:
        args.height = args.height or 768
        args.width = args.width or 768
        args.num_inference_steps = args.num_inference_steps or 50
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = args.prompt or "A photo of an astronaut riding a horse on mars."

    elif "stable-diffusion-3-medium" in args.model_id:
        args.num_inference_steps = args.num_inference_steps or 8
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.common_model_path = "common/"
        args.sub_model_path = "normal/"
        args.controlnet = args.controlnet or "Canny"
        check_sd3_normal_args(args)
        # default prompt, n_prompt, seed, guidance_scale, controlnet_conditioning_scale are set in check_sd3_normal_args
        config_sd3_controlnet_args(args)
        if args.dynamic_shape_file_path is None and args.height is None and args.width is None:
            args.height = 1024
            args.width = 1024

    elif (
        "stable-diffusion-3.5-medium" in args.model_id
        or "stable-diffusion-3-5" in args.model_id
        or "stable-diffusion-3.5" in args.model_id
    ):
        args.num_inference_steps = args.num_inference_steps or 8
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.sub_model_path = "normal/"
        args.common_model_path = "common/"
        args.controlnet = args.controlnet or "Canny"
        check_sd3_normal_args(args)
        # default prompt, n_prompt, seed, guidance_scale, controlnet_conditioning_scale are set in check_sd3_normal_args
        config_sd35_controlnet_args(args)
        if args.dynamic_shape_file_path is None and args.height is None and args.width is None:
            args.height = 1024
            args.width = 1024

    elif "sdxl-turbo" in args.model_id:
        args.height = args.height or 512
        args.width = args.width or 512
        args.num_inference_steps = args.num_inference_steps or 1
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 0.0
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = (
            args.prompt
            or "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
        )
        args.n_prompt = args.n_prompt or ""

    elif (
        "playground-v2.5" in args.model_id.lower()
        or "playground2.5" in args.model_id.lower()
    ):
        args.height = args.height or 1024
        args.width = args.width or 1024
        args.num_inference_steps = args.num_inference_steps or 50
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 3.0
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = args.prompt or "A highly detailed cinematic scene, rich colors, high contrast."
        args.n_prompt = args.n_prompt or ""
    
    elif "dreamshaper-xl-lightning" in args.model_id.lower():
        args.height = args.height or 1024
        args.width = args.width or 1024
        args.num_inference_steps = args.num_inference_steps or 50
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = args.prompt or "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"
        args.n_prompt = args.n_prompt or ""

    elif "SSD-1B".lower() in args.model_id.lower():
        args.height = args.height or 1024
        args.width = args.width or 1024
        args.num_inference_steps = args.num_inference_steps or 50
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = args.prompt or "An astronaut riding a green horse"
        args.n_prompt = args.n_prompt or "ugly, blurry, poor quality"

    elif "stable-diffusion-xl-base-1.0" in args.model_id or "sdxl-base" in args.model_id:
        args.height = args.height or 1024
        args.width = args.width or 1024
        args.num_inference_steps = args.num_inference_steps or 50
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = args.prompt or "An astronaut riding a green horse"
        args.n_prompt = args.n_prompt or ""

    elif "kolors" in args.model_id.lower():
        args.height = args.height or 1024
        args.width = args.width or 1024
        args.num_images_per_prompt = args.num_images_per_prompt or 1
       
        if args.controlnet is not None and args.controlnet.lower() != "none":
            if args.controlnet.lower() == "canny":
                args.num_inference_steps = args.num_inference_steps or 50
                # Set prompt file path if neither prompt nor prompt file is specified
                # if not args.prompt and not args.prompt_file_path:
                #     args.prompt_file_path = get_absolute_path("config/prompts_config.json")
                # Only set default prompt if no prompt and no prompt file
                if not args.prompt and not args.prompt_file_path:
                    args.prompt = "全景，一只可爱的白色小狗坐在杯子里，看向镜头，动漫风格，3d渲染，辛烷值渲染。"
                args.n_prompt = args.n_prompt or "nsfw，脸部阴影，低分辨率，jpeg伪影、模糊、糟糕，黑脸，霓虹灯"
                args.controlnet_conditioning_scale = (
                    args.controlnet_conditioning_scale
                    if args.controlnet_conditioning_scale is not None
                    else 0.7
                )
                args.guidance_scale = (
                    args.guidance_scale if args.guidance_scale is not None else 6.0
                )
                args.control_image_path = args.control_image_path or get_absolute_path("test/ref/Canny_dog_condition.jpg")
                args.seed = args.seed if args.seed is not None else 66
                args.strength = args.strength or 1.0
            elif args.controlnet.lower() == "inpainting":
                args.prompt = args.prompt or "穿着钢铁侠的衣服，高科技盔甲，主要颜色为红色和金色，并且有一些银色装饰。胸前有一个亮起的圆形反应堆装置，充满了未来科技感。超清晰，高质量，超逼真，高分辨率，最好的质量，超级细节，景深"
                args.n_prompt = args.n_prompt or "残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量"
                args.guidance_scale = (
                    args.guidance_scale if args.guidance_scale is not None else 6.0
                )
                args.num_inference_steps = args.num_inference_steps or 25
                args.strength = args.strength or 0.999
                args.seed = args.seed if args.seed is not None else 603
                args.control_mask_path = args.control_mask_path or get_absolute_path("test/ref/inpainting/kolors_mask.png")
                args.control_image_path = args.control_image_path or get_absolute_path("test/ref/inpainting/kolors_origin.png")
        else:
            args.num_inference_steps = args.num_inference_steps or 50
            args.guidance_scale = (
                args.guidance_scale if args.guidance_scale is not None else 5.0
            )
            # args.seed = args.seed if args.seed is not None else 66
            # args.prompt = args.prompt or "A close-up cinematic photo of a ladybug, macro shot, zoomed, high-quality, holding a sign that says '可图'"
            args.prompt = args.prompt or "一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着“可图”"

    elif "Nitro-E".lower() in args.model_id.lower():
        args.height = args.height or 512
        args.width = args.width or 512
        args.num_inference_steps = args.num_inference_steps or 20
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 4.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = args.prompt or "A hot air balloon in the shape of a heart grand canyon."
        args.n_prompt = args.n_prompt or ""
        args.seed = args.seed if args.seed is not None else 0
        
    elif "Segmind-Vega".lower() in args.model_id.lower():
        args.height = args.height or 1024
        args.width = args.width or 1024
        args.num_inference_steps = args.num_inference_steps or 50
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        if args.control_image_path is not None:
            if args.strength is None or args.strength < 0.0 or args.strength > 1.0:
                Logger.warning(
                    f"Invalid value for --strength: '{args.strength}'. "
                    f"Allowed values should be in [0.0, 1.0]. "
                    f"Set --strength to its default value 0.3 . "
                )
                args.strength = 0.3 
            actual_iters =  args.strength * args.num_inference_steps
            if actual_iters < 1.0:
                args.strength = round(1.01 / args.num_inference_steps, 3)
                Logger.warning(
                    f"There should be at least 1 denoising iteration. "
                    f"Actual value ({actual_iters}) assigned is not applicable. "
                    f"Set --strength to {args.strength} according to "
                    f"--num_inference_steps(which is {args.num_inference_steps}). "
                )
        args.prompt = args.prompt or "An astronaut riding a green horse"
        args.n_prompt = args.n_prompt or ""

    elif "flux.1-schnell" in args.model_id.lower():
        if args.dynamic_shape_file_path is None and args.height is None and args.width is None:
            args.height = 512
            args.width = 512
        args.num_inference_steps = args.num_inference_steps or 4
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 0.0
        )
        args.seed = args.seed or 0
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = args.prompt or "A beautiful anime girl with long silver hair and blue eyes, wearing a flowing white dress.Standing in a field of flowers under golden sunset.Soft warm lighting, gentle breeze, petals floating in the air.Highly detailed, delicate face, clean line art, vibrant colors, dreamy atmosphere."
        args.n_prompt = args.n_prompt or ""
        args.t5_sequence_len = args.t5_sequence_len or 256
        args.sub_model_path = args.sub_model_path or ""
        args.common_model_path = args.common_model_path or ""
    # O1 preset for SD3/SD3.5 tea caching.
    tea_caching_coef_explicit = args.tea_caching_coef is not None
    if args.optimize_o1:
        model_id_lower = args.model_id.lower() if args.model_id else ""
        is_sd35_model = (
            "stable-diffusion-3.5" in model_id_lower
            or "stable-diffusion-3-5" in model_id_lower
        )
        is_sd3_model = (
            "stable-diffusion-3" in model_id_lower
            and not is_sd35_model
        )
        is_controlnet_none = args.controlnet and args.controlnet.lower() == "none"
        is_controlnet_inpainting = args.controlnet and args.controlnet.lower() == "inpainting"

        if is_sd3_model or is_sd35_model:
            args.tea_caching = True

            poly10_coef = [1.0, 0.0]
            flux_coef = [498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861]

            if is_sd35_model:
                if is_controlnet_none:
                    args.tea_caching_delta = args.tea_caching_delta or 0.25
                    if not tea_caching_coef_explicit:
                        args.tea_caching_coef = flux_coef
                elif is_controlnet_inpainting:
                    args.tea_caching_delta = args.tea_caching_delta or 0.25
                    if not tea_caching_coef_explicit:
                        args.tea_caching_coef = flux_coef
                else:
                    args.tea_caching_delta = args.tea_caching_delta or 0.15
                    if not tea_caching_coef_explicit:
                        args.tea_caching_coef = poly10_coef
            else:
                if is_controlnet_none:
                    args.tea_caching_delta = args.tea_caching_delta or 0.1
                    if not tea_caching_coef_explicit:
                        args.tea_caching_coef = poly10_coef
                elif is_controlnet_inpainting:
                    args.tea_caching_delta = args.tea_caching_delta or 0.15
                    if not tea_caching_coef_explicit:
                        args.tea_caching_coef = poly10_coef
                else:
                    args.tea_caching_delta = args.tea_caching_delta or 0.15
                    if not tea_caching_coef_explicit:
                        args.tea_caching_coef = poly10_coef

            if is_controlnet_none and args.num_inference_steps <= 4:
                Logger.info(
                    "Disable tea caching for O1 when ControlNet=None and num_inference_steps<=4."
                )
                args.tea_caching = False

            # Logger.info(
            #     "O1 preset applied: tea_caching=%s, tea_caching_delta=%s, tea_caching_coef=%s",
            #     args.tea_caching,
            #     args.tea_caching_delta,
            #     args.tea_caching_coef,
            # )

    if args.tea_caching_coef is None:
        args.tea_caching_coef = [1.0, 0.0]

    if not args.model_path:
        Logger.debug(f"Will auto-download model from Hugging Face: {args.model_id}")
        args.model_path = None
    else:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"model_path {args.model_path} not exist")

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

        # 3. project local lib fallback
        candidates.append(
            get_absolute_path(os.path.join("lib", share_obj_name))
        )

        # pick first existing
        args.custom_op_path = next(
            (p for p in candidates if os.path.exists(p)),
            None
        )

    if not args.custom_op_path or not os.path.exists(args.custom_op_path):
        raise EnvironmentError(
            f"Can't find onnx custom op plugin: {share_obj_name}\n"
            f"Searched paths:\n  " + "\n  ".join(candidates)
        )

    # Set DD_PLUGINS_ROOT for all SD3 related models
    is_sd3_model = (
        "stable-diffusion-3" in args.model_id.lower()
        or "stable-diffusion-3.5" in args.model_id.lower()
        or "stable-diffusion-3-5" in args.model_id.lower()
    )
    
    if is_sd3_model:
        # Set DD_PLUGINS_ROOT if not set or empty
        if not os.environ.get("DD_PLUGINS_ROOT"):
            os.environ["DD_PLUGINS_ROOT"] = get_absolute_path("lib/transaction/stx")
        
        dd_plugins_path = os.environ["DD_PLUGINS_ROOT"]
        Logger.debug(f"DD_PLUGINS_ROOT set to: {dd_plugins_path}")
        
        if not os.path.exists(dd_plugins_path):
            raise EnvironmentError(
                f"invalid DD_PLUGINS_ROOT {os.environ['DD_PLUGINS_ROOT']}"
            )

    if "DD_ROOT" not in os.environ:
        os.environ["DD_ROOT"] = get_absolute_path("lib")
    if not os.path.exists(os.environ["DD_ROOT"]):
        raise EnvironmentError(f"invalid DD_ROOT {os.environ['DD_ROOT']}")

    # dynamic shape feature support does not require width and height to be equal
    # # Check width and height are equal and valid values
    # if args.width != args.height:
    #     # TODO, sd3 support h!=w
    #     raise ValueError("Width and height must be equal")

    # if args.width not in [512, 768, 1024]:
    #     raise ValueError("Width/height must be either 512, 768 or 1024")

    # DD_ENV_INSTR_STACK_SIZE_MB are disabled for now, the default values are:
    # SD3 without preemption: 30MB
    # SD3 with preemption: 60MB
    # others: 8MB
    # [deprecated]
    # # Set DD_ENV_INSTR_STACK_SIZE_MB environment variable based on width
    # if "DD_ENV_INSTR_STACK_SIZE_MB" not in os.environ:
    #     if args.width in (1024, 768):
    #         os.environ["DD_ENV_INSTR_STACK_SIZE_MB"] = "30"
    #     # else:
    #     #     os.environ["DD_ENV_INSTR_STACK_SIZE_MB"] = ""

    # # Check if both controlnet and prompt are specified
    # if args.controlnet and args.prompt:
    #     Logger.warning(
    #         "Both controlnet and prompt parameters specified. Note that the prompt will be overridden by GetControlnetInfo based on the controlnet type."
    #     )

    # profiling mode
    if args.enable_profile:
        if args.profiling_rounds <= 0:
            raise ValueError(
                f"profiling rounds should be > 0, now is {args.profiling_rounds}"
            )

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    return args
