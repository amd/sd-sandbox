#
# Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
#

from __future__ import annotations

from src.trigger import StableDiffusion3PipelineTrigger

from .base_runner import BaseRunner


class SD3xRunner(BaseRunner):
    # pr-231: Trigger reads args directly — no getter boilerplate needed
    trigger_cls = StableDiffusion3PipelineTrigger

    def __init__(self, args):
        super().__init__(args)
        self.default_shape_extras = {"t5_sequence_len": args.t5_sequence_len}
