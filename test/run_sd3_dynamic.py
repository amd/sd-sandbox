# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

import sys
from pathlib import Path
import logging as Logger
from datetime import datetime  # ADDED: For timestamp-based unique filenames

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import json
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
        control_image_path=args.control_image_path,
        enable_compile=args.enable_compile,
        enable_profile=args.enable_profile,
        profiling_rounds=args.profiling_rounds,
        width=args.width,
        t5_sequence_len=args.t5_sequence_len,
        is_dynamic=args.dynamic_shape,
    )
    
    # FIXED: Use prompt file in both profiling and batch mode
    # Previous logic excluded prompt_file_path when enable_profile=True
    # Now also supports prompt files for controlnet modes (not just "none")
    if args.prompt_file_path:
        with open(args.prompt_file_path, "r") as prompt_file:
            prompt_list = json.load(prompt_file)
    else:
        prompt_list = [args.prompt]

    # MODIFIED: Use command-line arguments directly (no JSON file dependency)
    # Single resolution from width/height/t5_sequence_len args
    Logger.info(f"Using resolution from args: {args.width}x{args.height}, t5_seq_len={args.t5_sequence_len}")

    run_mode = "profiling" if args.enable_profile else "batch"
    for idx, prompt in enumerate(prompt_list):
        # Use single resolution from command-line args
        height = args.height
        width = args.width
        t5_sequence_len = args.t5_sequence_len
        pipe_trigger.t5_sequence_len = t5_sequence_len
        
        images = pipe_trigger.run(
            height=height,
            width=width,
            prompt=prompt,
            n_prompt=args.n_prompt,
            num_inference_steps=args.num_inference_steps,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        output_dir = Path(args.output_path)
        if not args.no_images:  # ADDED: Check --no_images flag
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # ADDED: Unique timestamp per run
            for i in range(len(images)):
                image = images[i]
                if args.controlnet.lower() != "none":
                    filename = f"{output_dir}/{args.model_id.split('/')[-1]}_{i}_{args.controlnet}_{run_mode}_hw_{height}x{width}_steps{args.num_inference_steps}_idx{idx}_{timestamp}.png"
                    image.save(filename)
                    print(f"  [Image saved] {filename}")
                else:
                    filename = f"{output_dir}/{args.model_id.split('/')[-1]}_{i}_without_controlnet_{run_mode}_hw_{height}x{width}_steps{args.num_inference_steps}_idx{idx}_{timestamp}.png"
                    image.save(filename)
                    print(f"  [Image saved] {filename}")
        if args.enable_profile:
            if args.controlnet.lower() != "none":
                save_path = f"{output_dir}/{args.model_id.split('/')[-1]}_{args.controlnet}_{run_mode}_hw_{height}x{width}_steps{args.num_inference_steps}_idx{idx}.xlsx"
            else:
                save_path = f"{output_dir}/{args.model_id.split('/')[-1]}_without_controlnet_{run_mode}_hw_{height}x{width}_steps{args.num_inference_steps}_idx{idx}.xlsx"
            if not args.no_excel:  # MODIFIED: Added --no_excel check
                common.save_pipeline_metrics_to_excel(save_path, pipe_trigger.pipeline_metrics)
