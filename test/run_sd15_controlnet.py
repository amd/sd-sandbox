# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

from pathlib import Path
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
import json
import logging as Logger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        
        if args.prompt_file_path:
            with open(args.prompt_file_path, "r") as prompt_file:
                prompt_list = json.load(prompt_file)
        else:
        prompt_list = [args.prompt]

        run_mode = "profiling" if args.enable_profile else "batch"
        for prompt_idx, prompt in enumerate(prompt_list):
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

            if not args.no_images:
                for image_idx in range(len(images)):
                    image = images[image_idx]
                    filename = common.generate_filename(
                        args.model_id, args.width, args.height, args.num_inference_steps, prompt_idx, image_idx, args.controlnet, run_mode, suffix=".png"
                    )
                    img_path = f"{output_dir}/{filename}"
                    image.save(img_path)
                    Logger.info(f"[Image saved] {img_path}")
            
            if args.enable_profile and not args.no_excel:
                excel_filename = common.generate_filename(
                    args.model_id, args.width, args.height, args.num_inference_steps, prompt_idx, controlnet=args.controlnet, run_mode=run_mode, suffix=".xlsx"
                    )
                save_path = f"{output_dir}/{excel_filename}"
                    common.save_pipeline_metrics_to_excel(save_path, pipe_trigger.pipeline_metrics)
                Logger.info(f"[Excel saved] {save_path}")
            