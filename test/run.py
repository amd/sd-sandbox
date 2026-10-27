# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.runner import detect_runner_class
from src.utils import runner_args


if __name__ == "__main__":
    args = runner_args.parser.parse_args()
    runner_args.check_args(args)

    runner_cls = detect_runner_class(
        args.model_id,
        getattr(args, "controlnet", None),
        runner_name=getattr(args, "runner", None),
    )
    runner_cls(args).run()
