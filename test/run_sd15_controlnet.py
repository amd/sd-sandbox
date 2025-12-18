# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

import json
from pathlib import Path
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
import logging as Logger
from datetime import datetime  # ADDED: For timestamp-based unique filenames

import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.StableDiffusionControlnetONNXPipelineTrigger import (
    StableDiffusionControlnetONNXPipelineTrigger,
)
from src.utils import runner_args
from src.utils import common


def load_canny_image(image_path, low_threshold=100, high_threshold=200):
    """Load and process control image for Canny edge detection"""
    try:
        image = load_image(image_path)
        image = np.array(image)

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)

        Logger.debug("Control image loaded and processed successfully")
        return control_image
    except Exception as e:
        Logger.warning(f"Failed to load control image: {e}")
        return None


if __name__ == "__main__":
    args = runner_args.parser.parse_args()
    args.controlnet = "canny"
    runner_args.check_args(args)

    with StableDiffusionControlnetONNXPipelineTrigger(
        model_id=args.model_id,
        custom_op_path=args.custom_op_path,
        model_path=args.model_path,
        enable_profile=args.enable_profile,
        enable_compile=args.enable_compile,
        profiling_rounds=args.profiling_rounds,
        gpu=args.gpu,
    ) as pipe_trigger:
        prompt_list = [args.prompt]
        run_mode = "profiling" if args.enable_profile else "batch"
        for idx, prompt in enumerate([args.prompt]):
            control_image = load_canny_image(args.control_image_path)
            images = pipe_trigger.run(
                height=args.height,
                width=args.width,
                prompt=prompt,
                n_prompt=args.n_prompt,
                num_inference_steps=args.num_inference_steps,
                control_image=control_image,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )
            output_dir = Path(args.output_path)
            if not args.no_images:  # ADDED: Check --no_images flag
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # ADDED: Unique timestamp per run
                for i in range(len(images)):
                    image = images[i]
                    image.save(
                        f"{output_dir}/{args.model_id.split('/')[-1]}_{i}_controlnet_{run_mode}_{args.width}x{args.height}_steps{args.num_inference_steps}_idx{idx}_{timestamp}.png"  # MODIFIED: Added timestamp to filename
                    )
            if args.enable_profile:
                save_path = f"{output_dir}/{args.model_id.split('/')[-1]}_controlnet_{args.width}x{args.height}_steps{args.num_inference_steps}_idx{idx}.xlsx"
                if not args.no_excel:  # MODIFIED: Added --no_excel check
                    common.save_pipeline_metrics_to_excel(save_path, pipe_trigger.pipeline_metrics)
