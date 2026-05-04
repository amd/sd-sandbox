
#
# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved.
#

from pathlib import Path
import json
import logging as Logger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusers.utils import load_image
from src.StableDiffusionKolorsONNXPipelineTrigger import StableDiffusionKolorsONNXPipelineAMDTrigger

from src.utils import runner_args
from src.utils import common


if __name__ == "__main__":
    args = runner_args.parser.parse_args()
    runner_args.check_args(args)
    pipe_trigger = StableDiffusionKolorsONNXPipelineAMDTrigger(
        model_id=args.model_id,
        custom_op_path=args.custom_op_path,
        model_path=args.model_path,
        enable_compile=args.enable_compile,
        gpu=args.gpu,
        enable_profile=args.enable_profile,
        profiling_rounds=args.profiling_rounds,
        controlnet_str=args.controlnet,
        is_dynamic=args.dynamic_shape,
    )

    if args.prompt_file_path:
        with open(args.prompt_file_path, "r") as prompt_file:
            prompt_list = json.load(prompt_file)
    else:
        prompt_list = [args.prompt]
    
    # Set controlnet identifier for filename generation
    if args.control_image_path is not None and args.controlnet_conditioning_scale is not None:
        args.controlnet = "controlnet_scale{}".format(args.controlnet_conditioning_scale)
    else:
        args.controlnet = None

    # Dynamic shape handling: two use cases
    # Case 1: --dynamic_shape with --dynamic_shape_file_path JSON file
    # Case 2: --dynamic_shape with CLI --width/--height args
    if args.dynamic_shape and args.dynamic_shape_file_path:
        # Load shapes from JSON file
        with open(args.dynamic_shape_file_path, 'r', encoding='utf-8') as f:
            support_shapes = json.load(f)
        if not isinstance(support_shapes, list) or not all(
            isinstance(support_shape, dict) for support_shape in support_shapes
        ):
            raise TypeError("Invalid dynamic_shape_list, should be list(dict)")
        Logger.info(f"Using resolutions from {args.dynamic_shape_file_path}: {support_shapes}")
    else:
        # Use CLI args (single resolution)
        support_shapes = [{"height": args.height, "width": args.width, "seq_len": 512}]
        Logger.info(f"Using resolution from args: {args.width}x{args.height}")

    run_mode = "profiling" if args.enable_profile else "batch"
    output_dir = Path(args.output_path)
    inpainting_origin_size = None
    if args.control_mask_path is not None and args.control_image_path is not None:
        control_image_ref = load_image(args.control_image_path)
        inpainting_origin_size = control_image_ref.size

    for prompt_idx, prompt in enumerate(prompt_list):
        for i in range(len(support_shapes)):
            height = support_shapes[i]["height"]
            width = support_shapes[i]["width"]
            seq_len = support_shapes[i].get("seq_len", 512)
            
            # Prepare run arguments
            run_kwargs = {
                "height": height,
                "width": width,
                "prompt": prompt,
                "n_prompt": args.n_prompt,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "num_images_per_prompt": args.num_images_per_prompt,
                "seed": args.seed,
            }
            
            # Add img2img parameters if available
            if hasattr(args, 'image_path') and args.image_path is not None:
                run_kwargs["image_path"] = args.image_path

            if hasattr(args, 'strength') and args.strength is not None:
                run_kwargs["strength"] = args.strength
            
            # Add inpainting parameters if available
            if hasattr(args, 'control_mask_path') and args.control_mask_path is not None:
                run_kwargs["control_mask_path"] = args.control_mask_path
            
            # Add ControlNet parameters if available
            if hasattr(args, 'control_image_path') and args.control_image_path is not None:
                run_kwargs["control_image_path"] = args.control_image_path

            if hasattr(args, 'controlnet_conditioning_scale') and args.controlnet_conditioning_scale is not None:
                run_kwargs["controlnet_conditioning_scale"] = args.controlnet_conditioning_scale

            images = pipe_trigger.run(**run_kwargs)

            if not args.no_images:
                for image_idx in range(len(images)):
                    image = images[image_idx]
                    if inpainting_origin_size is not None:
                        image = image.resize(inpainting_origin_size)
                    filename = common.generate_filename(
                        args.model_id, width, height, args.num_inference_steps, prompt_idx, image_idx, args.controlnet, run_mode, suffix=".png"
                    )
                    img_path = f"{output_dir}/{filename}"
                    image.save(img_path)
                    Logger.info(f"[Image saved] {img_path}")
            if args.enable_profile and not args.no_excel:
                excel_filename = common.generate_filename(
                    args.model_id, width, height, args.num_inference_steps, prompt_idx, controlnet=args.controlnet, run_mode=run_mode, suffix=".xlsx"
                )
                excel_path = f"{output_dir}/{excel_filename}"
                common.save_pipeline_metrics_to_excel(excel_path, pipe_trigger.pipeline_metrics)
                Logger.info(f"[Excel saved] {excel_path}")
