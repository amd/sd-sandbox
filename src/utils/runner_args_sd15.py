# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

import argparse
import os
import sys
import logging as Logger

parser = argparse.ArgumentParser(description="SD1.5 ControlNet Pipeline Config")
# Env related args
parser.add_argument("--root_path", type=str, default="..", help="Path to SD models")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output."
)
parser.add_argument("--custom_op_path", type=str, help="Path to ONNX custom ops DLL")
parser.add_argument(
    "--DOD_ROOT",
    type=str,
    default="../lib",
    help="Path to Dynamic Dispatch root directory",
)
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
    help="SD1.5 model ID",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to SD1.5 models",
)

# runner related args
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
    help="Negative text description for generating images.",
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

# profiling args
parser.add_argument(
    "-p",
    "--enable_profile",
    action="store_true",
    help="Enable profiling measurement.",
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

    # Set default values for SD1.5
    args.height = args.height or 512
    args.width = args.width or 512
    args.num_inference_steps = args.num_inference_steps or 20
    args.guidance_scale = args.guidance_scale or 7.5
    args.num_images_per_prompt = args.num_images_per_prompt or 1
    args.prompt = (
        args.prompt
        or "a blue paradise bird in the jungle"
    )
    args.n_prompt = args.n_prompt or "NSFW, nude, naked, porn, ugly"
    args.controlnet_conditioning_scale = args.controlnet_conditioning_scale or 1.0

    if not args.model_path:
        args.model_path = args.root_path + "/models/sd15_controlnet/"
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"model_path {args.model_path} not exist")

    if not args.custom_op_path:
        onnx_utils_root = os.environ.get("ONNX_UTILS_ROOT")
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

    # profiling mode
    if args.enable_profile:
        if args.profiling_rounds <= 0:
            raise ValueError(
                f"profiling rounds should >= 0, now is {args.profiling_rounds}"
            )

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args 