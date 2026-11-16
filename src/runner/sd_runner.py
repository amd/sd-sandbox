# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from src.trigger import StableDiffusionONNXPipelineAMDTrigger

from .base_runner import BaseRunner


class SDRunner(BaseRunner):
    # Trigger reads args directly — no getter boilerplate needed
    trigger_cls = StableDiffusionONNXPipelineAMDTrigger
