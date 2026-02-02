# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

from pathlib import Path
import logging as Logger
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.StableDiffusion3PipelineTrigger import StableDiffusion3PipelineTrigger

from src.utils import runner_args
from src.utils import common


if __name__ == "__main__":
    args = runner_args.parser.parse_args()
    runner_args.check_args(args)

    pipe_trigger = StableDiffusion3PipelineTrigger(
        model_id=args.model_id,
        custom_op_path=args.custom_op_path,
        root_path=args.root_path,
        model_path=args.model_path,
        sub_model_path=args.sub_model_path,
        common_model_path=args.common_model_path,
        controlnet_str=args.controlnet,
        enable_compile=args.enable_compile,
        enable_profile=args.enable_profile,
        profiling_rounds=args.profiling_rounds,
        width=args.width,
        t5_sequence_len=args.t5_sequence_len,
        is_dynamic=args.dynamic_shape,
    )
        if args.prompt_file_path:
            with open(args.prompt_file_path, "r") as prompt_file:
                prompt_list = json.load(prompt_file)
        else:
            prompt_list = [args.prompt]

    # MODIFIED: Dynamic shape handling now supports two distinct use cases:
    # Case 1: User provides --dynamic_shape with explicit --width/--height args
    #         ? Use the single provided resolution with dynamic shape models
    # Case 2: User provides --dynamic_shape with --dynamic_shape_file_path JSON
    #         ? Load and iterate through all resolutions defined in the JSON file
    # 
    # This change allows dynamic models to be tested at a single resolution without
    # requiring a separate JSON file, while still supporting multi-resolution testing
    # when a JSON file is provided. Previously, --dynamic_shape always required a JSON file.
    if args.dynamic_shape and args.dynamic_shape_file_path:
        # Case 2: Dynamic shape with JSON file - read all shapes from file
        with open(args.dynamic_shape_file_path, 'r', encoding='utf-8') as f:
            support_shapes = json.load(f)
        if not isinstance(support_shapes, list) or not all(
            isinstance(support_shape, dict) for support_shape in support_shapes
        ):
            raise TypeError(
                "invalid dynamic_shape_list attr, shoudl be list(dict)"
            )
        Logger.info(f"Using resolutions from {args.dynamic_shape_file_path}: {support_shapes}")
    else:
        # Case 1: Use provided height/width from args (works with or without --dynamic_shape flag)
        support_shapes = [{"height": args.height, "width": args.width, "t5_sequence_len": args.t5_sequence_len}]
        Logger.info(f"Using resolution from args: {args.width}x{args.height}, t5_seq_len={args.t5_sequence_len}")

        run_mode = "profiling" if args.enable_profile else "batch"
    output_dir = Path(args.output_path)

    for prompt_idx, prompt in enumerate(prompt_list):
        for i in range(len(support_shapes)):
            height = support_shapes[i]["height"]
            width = support_shapes[i]["width"]
            t5_sequence_len = support_shapes[i]["t5_sequence_len"]
            pipe_trigger.t5_sequence_len = t5_sequence_len
            images = pipe_trigger.run(
                height=height,
                width=width,
                prompt=prompt,
                n_prompt=args.n_prompt,
                num_inference_steps=args.num_inference_steps,
                control_image_path=args.control_image_path,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )
            if not args.no_images:
                for image_idx in range(len(images)):
                    image = images[image_idx]
                    img_filename = common.generate_filename(
                        args.model_id, width, height, args.num_inference_steps, prompt_idx, image_idx, args.controlnet, run_mode, suffix=".png"
                        )
                    img_path = f"{output_dir}/{img_filename}"
                    image.save(img_path)
                    Logger.info(f"[Image saved] {img_path}")
            if args.enable_profile and not args.no_excel:
                excel_filename = common.generate_filename(
                    args.model_id, width, height, args.num_inference_steps, prompt_idx, controlnet=args.controlnet, run_mode=run_mode, suffix=".xlsx"
                        )
                save_path = f"{output_dir}/{excel_filename}"
                    common.save_pipeline_metrics_to_excel(save_path, pipe_trigger.pipeline_metrics)
