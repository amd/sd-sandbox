# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from src.trigger import StableDiffusionControlnetONNXPipelineTrigger

from .base_runner import BaseRunner


class SD15ControlnetRunner(BaseRunner):
    # pr-231: Trigger reads args directly; canny loading moved into trigger
    trigger_cls = StableDiffusionControlnetONNXPipelineTrigger

    def pre_run(self, args):
        args.controlnet = "canny"
