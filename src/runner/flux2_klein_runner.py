#
# Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
#

from __future__ import annotations

from src.trigger.StableDiffusionFlux2KleinONNXPipelineTrigger import (
    StableDiffusionFlux2KleinONNXPipelineTrigger,
)

from .base_runner import BaseRunner


class Flux2KleinRunner(BaseRunner):
    trigger_cls = StableDiffusionFlux2KleinONNXPipelineTrigger
    default_shape_extras = {"txt_seq_len": 256}
