# Copyright (C) 2025 Advanced Micro Devices, Inc.  All rights reserved. Portions of this file consist of AI-generated content.

from pathlib import Path
import json
import sys
from datetime import datetime  # ADDED: For timestamp-based unique filenames
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
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
    )

    if args.prompt_file_path:
        with open(args.prompt_file_path, "r") as prompt_file:
            prompt_list = json.load(prompt_file)
    else:
        prompt_list = [args.prompt]

    for idx, prompt in enumerate(prompt_list):
        images = pipe_trigger.run(
            height=args.height,
            width=args.width,
            prompt=prompt,
            n_prompt=args.n_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            seed=args.seed,
        )

        output_dir = Path(args.output_path)
        if not args.no_images:  # ADDED: Check --no_images flag
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # ADDED: Unique timestamp per run
            for i in range(len(images)):
                image = images[i]
                image.save(
                    f"{output_dir}/{args.model_id.split('/')[-1]}_{i}_{args.width}x{args.height}_steps{args.num_inference_steps}_idx{idx}_{timestamp}.png"  # MODIFIED: Added timestamp to filename
                )
        if args.enable_profile and not args.no_excel:  # MODIFIED: Added --no_excel check
            save_path = f"{output_dir}/{args.model_id.split('/')[-1]}_{args.width}x{args.height}_steps{args.num_inference_steps}_idx{idx}_bs{args.num_images_per_prompt}.xlsx"
            common.save_pipeline_metrics_to_excel(save_path, pipe_trigger.pipeline_metrics)
