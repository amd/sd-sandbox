#
# Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
#

from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union

from src.utils import common
from src.trigger import BaseOnnxPipelineTrigger

logger = logging.getLogger(__name__)

RunResult = Union[List[Any], Tuple[Any, ...]]


def register_project_root() -> None:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass
class ShapeContext:
    height: int
    width: int
    extras: Dict[str, Any] = field(default_factory=dict)


def get_shapes(
    args,
    *,
    default_extras: Optional[Mapping[str, Any]] = None,
) -> List[ShapeContext]:
    extras = dict(default_extras or {})
    shape_file = getattr(args, "dynamic_shape_file_path", None)

    if getattr(args, "dynamic_shape", False) and shape_file:
        with open(shape_file, "r", encoding="utf-8") as f:
            raw_shapes = json.load(f)
        if not isinstance(raw_shapes, list) or not all(
            isinstance(item, dict) for item in raw_shapes
        ):
            raise TypeError("invalid dynamic_shape_list attr, should be list(dict)")

        shapes: List[ShapeContext] = []
        for item in raw_shapes:
            item_extras = {
                k: v for k, v in item.items() if k not in ("height", "width")
            }
            shapes.append(
                ShapeContext(
                    height=item["height"],
                    width=item["width"],
                    extras={**extras, **item_extras},
                )
            )
        logger.info("Using resolutions from %s: %s", shape_file, raw_shapes)
        return shapes

    height = args.height
    width = args.width
    extras_str = ", ".join(f"{k}: {v}" for k, v in extras.items())
    logger.info(
        "Using resolution from args: {height: %s, width: %s%s%s}",
        height,
        width,
        (", " + extras_str) if extras_str else "",
        "",
    )
    return [ShapeContext(height=height, width=width, extras=dict(extras))]



def get_prompts(args) -> List[str]:
    if getattr(args, "prompt_file_path", None):
        with open(args.prompt_file_path, "r", encoding="utf-8") as prompt_file:
            prompt_list = json.load(prompt_file)
        if not isinstance(prompt_list, list):
            raise TypeError("prompt_file_path must contain a JSON list of strings")
        return prompt_list
    return [args.prompt]


def get_run_mode(args) -> str:
    return "profiling" if getattr(args, "enable_profile", False) else "batch"


def normalize_images(result: RunResult) -> List[Any]:
    if isinstance(result, tuple):
        if not result:
            return []
        head = result[0]
        return head if isinstance(head, list) else [head]
    if isinstance(result, list):
        return result
    return [result]


def save_images(
    args,
    images: Iterable[Any],
    *,
    width: int,
    height: int,
    num_inference_steps: int,
    prompt_idx: int,
    run_mode: str,
    controlnet: Optional[str] = None,
) -> None:
    if getattr(args, "no_images", False):
        return

    output_dir = Path(args.output_path)
    for image_idx, image in enumerate(images):
        if getattr(image, "size", None) and image.size != (width, height):
            image = image.resize((width, height))
        filename = common.generate_filename(
            args.model_id,
            width,
            height,
            num_inference_steps,
            prompt_idx,
            image_idx,
            controlnet=controlnet,
            run_mode=run_mode,
            suffix=".png",
        )
        img_path = output_dir / filename
        image.save(img_path)
        logger.info("[Image saved] %s", img_path)


def save_profiling(
    trigger,
    args,
    *,
    width: int,
    height: int,
    prompt_idx: int,
    num_inference_steps: int,
    run_mode: str,
    controlnet: Optional[str] = None,
) -> None:
    if not getattr(args, "enable_profile", False) or getattr(args, "no_excel", False):
        return

    output_dir = Path(args.output_path)
    excel_filename = common.generate_filename(
        args.model_id,
        width,
        height,
        num_inference_steps,
        prompt_idx,
        controlnet=controlnet,
        run_mode=run_mode,
        suffix=".xlsx",
    )
    save_path = output_dir / excel_filename
    metrics_key = f"h{height}_w{width}"
    profiling_result = getattr(trigger, "pipeline_metrics", {}).get(metrics_key)
    if profiling_result is None:
        logger.error("Profiling result not found for key: %s", metrics_key)
        return
    profiling_result.save_to_excel(str(save_path))


class BaseRunner(ABC):
    """Shared prompt × shape orchestration — subclasses set trigger_cls and optional hooks."""

    trigger_cls: Type[BaseOnnxPipelineTrigger]
    default_shape_extras: Dict[str, Any] = {}

    def __init__(self, args):
        self.args = args

    # Hooks — override in subclasses for model-specific behaviour

    def pre_run(self, args) -> None:
        """Set defaults on args before the run loop (e.g. args.controlnet)."""

    def get_save_dimensions(self, trigger, shape) -> Tuple[int, int]:
        """Override when saved image size differs from shape (e.g. outpainting)."""
        return shape.width, shape.height

    # pr-231: Triggers accept args directly — no get_trigger_kwargs / get_run_kwargs needed
    def run(self) -> None:
        args = self.args
        self.pre_run(args)
        trigger_ctx = self.trigger_cls(args)

        prompt_list = get_prompts(args)
        shape_list = get_shapes(args, default_extras=self.default_shape_extras)
        run_mode = get_run_mode(args)
        controlnet_label = getattr(args, "controlnet", "none")
        num_inference_steps = args.num_inference_steps

        for prompt_idx, prompt in enumerate(prompt_list):
            for shape in shape_list:
                images = normalize_images(trigger_ctx.run(args, prompt, shape))
                save_w, save_h = self.get_save_dimensions(trigger_ctx, shape)
                save_images(
                    args,
                    images,
                    width=save_w,
                    height=save_h,
                    num_inference_steps=num_inference_steps,
                    prompt_idx=prompt_idx,
                    controlnet=controlnet_label,
                    run_mode=run_mode,
                )
                save_profiling(
                    trigger_ctx,
                    args,
                    width=shape.width,
                    height=shape.height,
                    prompt_idx=prompt_idx,
                    num_inference_steps=num_inference_steps,
                    controlnet=controlnet_label,
                    run_mode=run_mode,
                )
