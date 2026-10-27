# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from src.trigger import FluxPipelineTrigger

from .base_runner import BaseRunner


class FluxRunner(BaseRunner):
    # pr-231: Trigger reads args directly — no getter boilerplate needed
    trigger_cls = FluxPipelineTrigger
    default_shape_extras = {"txt_seq_len": 256}
