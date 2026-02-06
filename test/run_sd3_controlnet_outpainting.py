# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

from pathlib import Path
import logging as Logger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from src.StableDiffusion3ControlnetOutpaintingONNXPipelineTrigger import (
    StableDiffusion3ControlnetOutpaintingONNXPipelineTrigger,
)

from src.utils import runner_args
from src.utils import common


if __name__ == "__main__":
    args = runner_args.parser.parse_args()
    args.controlnet = args.controlnet or "OutPainting"
    runner_args.check_args(args)
    pipe_trigger = StableDiffusion3ControlnetOutpaintingONNXPipelineTrigger(
        model_id=args.model_id,
        custom_op_path=args.custom_op_path,
        root_path=args.root_path,
        model_path=args.model_path,
        sub_model_path=args.sub_model_path,
        common_model_path=args.common_model_path,
        control_image_path=args.control_image_path,
        control_mask_path=args.control_mask_path,
        image_pads=args.image_pads,
        enable_compile=args.enable_compile,
        gpu=args.gpu,
        enable_profile=args.enable_profile,
        profiling_rounds=args.profiling_rounds,
        controlnet_name=args.controlnet,
        width=args.width,
        t5_sequence_len=args.t5_sequence_len,
    )

    if args.prompt_file_path:
        with open(args.prompt_file_path, "r") as prompt_file:
            prompt_list = json.load(prompt_file)
    else:
        prompt_list = [args.prompt]

    run_mode = "profiling" if args.enable_profile else "batch"
    for prompt_idx, prompt in enumerate(prompt_list):
        images, origin_width, origin_height = pipe_trigger.run(
            height=args.height,
            width=args.width,
            prompt=args.prompt,
            n_prompt=args.n_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            control_image_path=args.control_image_path,
            control_mask_path=args.control_mask_path,
            num_images_per_prompt=args.num_images_per_prompt,
            seed=args.seed,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        )

        output_dir = Path(args.output_path)
        if not args.no_images:  # ADDED: Check --no_images flag
            for image_idx in range(len(images)):
                image = images[image_idx]
                image = image.resize((origin_width, origin_height))
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
