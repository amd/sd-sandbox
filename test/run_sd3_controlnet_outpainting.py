# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

import sys
from pathlib import Path
from datetime import datetime  # ADDED: For timestamp-based unique filenames

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
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

    for idx, prompt in enumerate(prompt_list):
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # ADDED: Unique timestamp per run
            for i in range(len(images)):
                image = images[i]
                image = image.resize((origin_width, origin_height))
                image.save(
                    f"{output_dir}/{args.model_id.split('/')[-1]}_{i}_{args.controlnet}_{args.width}x{args.height}_steps{args.num_inference_steps}_idx{idx}_{timestamp}.png"  # MODIFIED: Added timestamp to filename
                )
        if args.enable_profile:
            if args.controlnet.lower() != "none":
                save_path = f"{output_dir}/{args.model_id.split('/')[-1]}_{args.controlnet}_{args.width}x{args.height}_steps{args.num_inference_steps}_idx{idx}.xlsx"
            else:
                save_path = f"{output_dir}/{args.model_id.split('/')[-1]}_without_controlnet_{args.width}x{args.height}_steps{args.num_inference_steps}_idx{idx}.xlsx"
            if not args.no_excel:  # MODIFIED: Added --no_excel check
                common.save_pipeline_metrics_to_excel(save_path, pipe_trigger.pipeline_metrics)
