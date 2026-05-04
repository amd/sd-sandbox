
#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

from pathlib import Path
import json
import logging as Logger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.StableDiffusionXLONNXPipelineTrigger import StableDiffusionXLONNXPipelineAMDTrigger

from src.utils import runner_args
from src.utils import common


if __name__ == "__main__":
    args = runner_args.parser.parse_args()
    runner_args.check_args(args)
    pipe_trigger = StableDiffusionXLONNXPipelineAMDTrigger(
        model_id=args.model_id,
        custom_op_path=args.custom_op_path,
        model_path=args.model_path,
        enable_compile=args.enable_compile,
        gpu=args.gpu,
        enable_profile=args.enable_profile,
        profiling_rounds=args.profiling_rounds,
        control_image_path = args.control_image_path,
        revision=args.revision,
    )

    if args.prompt_file_path:
        with open(args.prompt_file_path, "r") as prompt_file:
            prompt_list = json.load(prompt_file)
    else:
        prompt_list = [args.prompt]

    if args.dynamic_shape and args.dynamic_shape_file_path:
        # Case 2: Dynamic shape with JSON file - read all shapes from file
        with open(args.dynamic_shape_file_path, 'r', encoding='utf-8') as f:
            support_shapes = json.load(f)
        if not isinstance(support_shapes, list) or not all(
            isinstance(support_shape, dict) for support_shape in support_shapes
        ):
            raise TypeError(
                "invalid dynamic_shape_list attr, should be list(dict)"
            )
        Logger.info(f"Using resolutions from {args.dynamic_shape_file_path}: {support_shapes}")
    else:
        # Case 1: Use provided height/width from args (works with or without --dynamic_shape flag)
        support_shapes = [{"height": args.height, "width": args.width}]
        Logger.info(f"Using resolution from args: {args.width}x{args.height}")

    if args.control_image_path is not None:
        args.controlnet = "latent_strength{}".format(args.strength)
        # strength controls how control image influences result by timesteps and num_inference_steps
        init_timestep = min(int(args.num_inference_steps * args.strength), args.num_inference_steps)
        t_start = max(args.num_inference_steps - init_timestep, 0)
        num_inference_steps_real = args.num_inference_steps - t_start
    else :
        num_inference_steps_real = args.num_inference_steps

    run_mode = "profiling" if args.enable_profile else "batch"
    for prompt_idx, prompt in enumerate(prompt_list):
        for i in range(len(support_shapes)):
            height = support_shapes[i]["height"]
            width = support_shapes[i]["width"]
            images = pipe_trigger.run(
                height=height,
                width=width,
                prompt=prompt,
                n_prompt=args.n_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                seed=args.seed,
                control_image_path = args.control_image_path,
                strength = args.strength,
            )
            output_dir = Path(args.output_path)
            if not args.no_images:  # ADDED: Check --no_images flag
                for image_idx in range(len(images)):
                    image = images[image_idx]
                    filename = common.generate_filename(
                        args.model_id, width, height, num_inference_steps_real, prompt_idx, image_idx, args.controlnet, run_mode, suffix=".png"
                    )
                    img_path = f"{output_dir}/{filename}"
                    image.save(img_path)
                    Logger.info(f"[Image saved] {img_path}")
            if args.enable_profile and not args.no_excel:
                excel_filename = common.generate_filename(
                    args.model_id, width, height, num_inference_steps_real, prompt_idx, controlnet=args.controlnet, run_mode=run_mode, suffix=".xlsx"
                )
                excel_path = f"{output_dir}/{excel_filename}"
                common.save_pipeline_metrics_to_excel(excel_path, pipe_trigger.pipeline_metrics)
                Logger.info(f"[Excel saved] {excel_path}")
