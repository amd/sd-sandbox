# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from src.trigger import StableDiffusion3ControlnetOutpaintingONNXPipelineTrigger

from .base_runner import BaseRunner


class SD3xOutpaintingRunner(BaseRunner):
    # pr-231: Trigger reads args directly; hooks replace the full run() override
    trigger_cls = StableDiffusion3ControlnetOutpaintingONNXPipelineTrigger

    def pre_run(self, args):
        args.controlnet = args.controlnet or "OutPainting"

    def get_save_dimensions(self, trigger, shape):
        # Outpainting saves at the original image dimensions, not the padded shape
        return trigger.origin_width, trigger.origin_height
