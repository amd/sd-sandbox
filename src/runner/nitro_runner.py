# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from src.trigger import StableDiffusionNitroEONNXPipelineTrigger

from .base_runner import BaseRunner


class NitroERunner(BaseRunner):
    # Trigger reads args directly — no getter boilerplate needed
    trigger_cls = StableDiffusionNitroEONNXPipelineTrigger
