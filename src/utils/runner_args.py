# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

import argparse
import os
import sys
import logging as Logger

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
# model related args
parser.add_argument(
    "--model_id",
    type=str,
    default="runwayml/stable-diffusion-v1-5",
    help="Choose SD models type, \
        e.g., 'runwayml/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-2-1-base', \
            'stabilityai/sd-turbo', 'stabilityai/stable-diffusion-2-1', 'stabilityai/sdxl-turbo', \
            'stabilityai/stable-diffusion-xl-base-1.0', 'stabilityai/stable-diffusion-3-medium-diffuser', \
            stabilityai/stable-diffusion-3.5-medium",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to SD models",
)
parser.add_argument(
    "-C",
    "--controlnet",
    type=str,
    help="Canny, Tile, Pose, OutPainting, Removal, InPainting, or None",
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
parser.add_argument(
    "--dynamic_shape_file_path",
    type=str,
    nargs="?",
    const="",
    default=None,  # MODIFIED: Changed from "../config/sd3_dynamic_shape.json" to None
    help="Path to the dynamic shape configuration file. If not provided, uses single resolution from width/height args.",
)


# ----------------------------------------------------------------------------------------------------------------------
def check_sd3_normal_args(args):
    controlnet_type = args.controlnet.lower()
    VALID_CONTROLNET_VALUES = [
        "canny",
        "tile",
        "pose",
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
        args.sub_model_path = "outpainting_removal/"
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
        args.control_image_path = args.control_image_path or "ref/outpainting.jpg"
        if not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # removal
    elif args.controlnet.lower() == "removal":
        args.height = args.height or 1024
        args.width = args.width or 1024
        if not args.height == 1024 and not args.width == 1024:
            raise ValueError("Height and width must be 1024")
        args.sub_model_path = "outpainting_removal/"
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
        args.control_image_path = args.control_image_path or "ref/removal/origin.jpg"
        args.control_mask_path = args.control_mask_path or "ref/removal/mask.jpg"
        if not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )
        if not os.path.exists(args.control_mask_path):
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
            args.guidance_scale if args.guidance_scale is not None else 2.5
        )
        args.control_image_path = args.control_image_path or "ref/inpainting/origin.jpg"
        args.control_mask_path = args.control_mask_path or "ref/inpainting/mask.jpg"
        if not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )
        if not os.path.exists(args.control_mask_path):
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
            args.prompt_file_path = "../config/prompts_config.json"
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
        args.control_image_path = args.control_image_path or "ref/canny.jpg"
        if not os.path.exists(args.control_image_path):
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
            args.prompt_file_path = "../config/prompts_config.json"
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
        args.control_image_path = args.control_image_path or "ref/tile.jpg"
        if not os.path.exists(args.control_image_path):
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
            args.prompt_file_path = "../config/prompts_config.json"
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
        args.control_image_path = args.control_image_path or "ref/pose.jpg"
        if not os.path.exists(args.control_image_path):
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
            args.prompt_file_path = "../config/prompts_config.json"
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
        args.sub_model_path = "outpainting_removal/"
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
        args.control_image_path = args.control_image_path or "ref/outpainting.jpg"
        if not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )

    # removal
    elif args.controlnet.lower() == "removal":
        args.height = args.height or 1024
        args.width = args.width or 1024
        if not args.height == 1024 and not args.width == 1024:
            raise ValueError("Height and width must be 1024")
        args.sub_model_path = "outpainting_removal/"
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
        args.control_image_path = args.control_image_path or "ref/removal/origin.jpg"
        args.control_mask_path = args.control_mask_path or "ref/removal/mask.jpg"
        if not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )
        if not os.path.exists(args.control_mask_path):
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
        args.control_image_path = args.control_image_path or "ref/inpainting/origin.jpg"
        args.control_mask_path = args.control_mask_path or "ref/inpainting/mask.jpg"
        if not os.path.exists(args.control_image_path):
            raise EnvironmentError(
                f"can't find control image path  {args.control_image_path}"
            )
        if not os.path.exists(args.control_mask_path):
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
            args.prompt_file_path = "../config/prompts_config.json"
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
        args.control_image_path = args.control_image_path or "ref/canny.jpg"
        if not os.path.exists(args.control_image_path):
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
            args.prompt_file_path = "../config/prompts_config.json"
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
        args.control_image_path = args.control_image_path or "ref/tile.jpg"
        if not os.path.exists(args.control_image_path):
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
            args.prompt_file_path = "../config/prompts_config.json"
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
        args.control_image_path = args.control_image_path or "ref/pose.jpg"
        if not os.path.exists(args.control_image_path):
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
            args.prompt_file_path = "../config/prompts_config.json"
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
    elif "stable-diffusion-v1-5" in args.model_id:
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
            args.model_path = (
                args.model_path or args.root_path + "/models/sd15_controlnet/"
            )
            args.controlnet_conditioning_scale = (
                args.controlnet_conditioning_scale
                if args.controlnet_conditioning_scale is not None
                else 1.5
            )
            args.prompt = args.prompt or "a blue paraidse bird in the jungle"
            args.control_image_path = args.control_image_path or "ref/control.png"
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
        args.num_images_per_prompt = args.num_images_per_prompt or 2
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

    elif "stable-diffusion-3-medium-diffusers" in args.model_id:
        args.num_inference_steps = args.num_inference_steps or 8
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.common_model_path = "common/"
        args.sub_model_path = "normal/"
        args.controlnet = args.controlnet or "Canny"
        check_sd3_normal_args(args)
        # default prompt, n_prompt, seed, guidance_scale, controlnet_conditioning_scale are set in check_sd3_normal_args
        config_sd3_controlnet_args(args)

    elif (
        "stable-diffusion-3.5-medium" in args.model_id
        or "stable-diffusion-3-5" in args.model_id
        or "stable-diffusion-3.5" in args.model_id
    ):
        args.num_inference_steps = args.num_inference_steps or 8
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.model_path = args.model_path or args.root_path + "/models/sd3.5/"
        args.sub_model_path = "normal/"
        args.common_model_path = "common/"
        args.controlnet = args.controlnet or "Canny"
        check_sd3_normal_args(args)
        # default prompt, n_prompt, seed, guidance_scale, controlnet_conditioning_scale are set in check_sd3_normal_args
        config_sd35_controlnet_args(args)

    elif "sdxl-turbo" in args.model_id:
        args.height = args.height or 512
        args.width = args.width or 512
        args.num_inference_steps = args.num_inference_steps or 1
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 0.0
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 2
        args.prompt = (
            args.prompt
            or "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
        )
        args.n_prompt = args.n_prompt or ""
        args.model_path = args.model_path or args.root_path + "/models/sdxl_turbo/"

    elif "stable-diffusion-xl-base-1.0" in args.model_id:
        args.height = args.height or 1024
        args.width = args.width or 1024
        args.num_inference_steps = args.num_inference_steps or 50
        args.guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else 7.5
        )
        args.num_images_per_prompt = args.num_images_per_prompt or 1
        args.prompt = args.prompt or "An astronaut riding a green horse"
        args.n_prompt = args.n_prompt or ""
        args.model_path = args.model_path or args.root_path + "/models/sdxl-base-1.0/"

    if not args.model_path:
        args.model_path = args.root_path + "/models/sd3/"
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"model_path {args.model_path} not exist")

    if not args.custom_op_path:
        onnx_utils_root = os.environ.get("ONNX_UTILS_ROOT", "")
        if onnx_utils_root:
            args.custom_op_path = os.path.join(
                onnx_utils_root, "build", "install", "bin", "onnx_custom_ops.dll"
            )
        else:
            # Use absolute path to the lib directory in the current project
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            args.custom_op_path = os.path.join(project_root, "lib", "onnx_custom_ops.dll")
    if not os.path.exists(args.custom_op_path):
        raise EnvironmentError(f"can't find onnx_custom_ops.dll {args.custom_op_path}")

    if args.dynamic_shape:
        if "DD_PLUGINS_ROOT" not in os.environ:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            os.environ["DD_PLUGINS_ROOT"] = os.path.join(project_root, "lib", "transaction", "stx")
        if not os.path.exists(os.environ["DD_PLUGINS_ROOT"]):
            raise EnvironmentError(
                f"invalid DD_PLUGINS_ROOT {os.environ['DD_PLUGINS_ROOT']}"
            )

    if "DD_ROOT" not in os.environ:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        os.environ["DD_ROOT"] = os.path.join(project_root, "lib") + "/"
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
