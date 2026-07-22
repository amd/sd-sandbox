#
# Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
#

from __future__ import annotations

import logging

from src.trigger import StableDiffusionXLONNXPipelineAMDTrigger

from .base_runner import BaseRunner

logger = logging.getLogger(__name__)


class SDXLRunner(BaseRunner):
    trigger_cls = StableDiffusionXLONNXPipelineAMDTrigger

    # pr-231: Strength validation moved from get_run_kwargs — trigger reads args directly
    def pre_run(self, args):
        if args.control_image_path is not None:
            if "segmind-vega" in args.model_id:
                strength = args.strength
                if strength is None or not 0.0 <= strength <= 1.0:
                    logger.warning(
                        "Invalid --strength %r for segmind-vega; resetting to 0.3.",
                        strength,
                    )
                    strength = 0.3
                min_strength = 1.01 / max(args.num_inference_steps, 1)
                if strength * args.num_inference_steps < 1.0:
                    strength = round(min_strength, 3)
                    logger.warning(
                        "Too few denoising steps; adjusted --strength to %s.",
                        strength,
                    )
                args.strength = strength
            args.controlnet = f"latent_strength{args.strength}"