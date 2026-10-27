# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from diffusers.utils import load_image

from src.trigger import StableDiffusionKolorsONNXPipelineAMDTrigger

from .base_runner import BaseRunner


class KolorsRunner(BaseRunner):
    # pr-231: Trigger reads args directly; hooks replace the full run() override
    trigger_cls = StableDiffusionKolorsONNXPipelineAMDTrigger
    default_shape_extras = {"seq_len": 512}

    def get_save_dimensions(self, trigger, shape):
        # Inpainting saves at the original image dimensions, not the generated shape
        if trigger._inpainting_origin_size is not None:
            return trigger._inpainting_origin_size
        return shape.width, shape.height
