# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

from src.utils.model_config import _cn_equals

from .base_runner import BaseRunner

logger = logging.getLogger(__name__)


@dataclass
class RunnerSpec:
    """Maps ``(model_id, controlnet)`` to a :class:`BaseRunner` subclass."""

    name: str
    runner_cls: Type[BaseRunner]
    model_ids: List[str]
    controlnet_filter: Optional[Union[str, List[str]]] = None

    def matches(self, model_id_lower: str, controlnet: Optional[str]) -> bool:
        if not any(token in model_id_lower for token in self.model_ids):
            return False
        if self.controlnet_filter is None:
            return True
        if isinstance(self.controlnet_filter, list):
            return any(_cn_equals(controlnet, item) for item in self.controlnet_filter)
        return _cn_equals(controlnet, self.controlnet_filter)


class RunnerRegistry:
    def __init__(self) -> None:
        self._specs: List[RunnerSpec] = []
        self._by_name: Dict[str, RunnerSpec] = {}

    def register(self, spec: RunnerSpec) -> None:
        self._specs.append(spec)
        self._by_name[spec.name] = spec

    def get_by_name(self, name: str) -> RunnerSpec:
        try:
            return self._by_name[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._by_name))
            raise ValueError(f"Unknown runner {name!r}. Known: {known}") from exc

    def detect(self, model_id: str, controlnet: Optional[str] = None) -> RunnerSpec:
        mid = model_id.lower()
        for spec in self._specs:
            if spec.matches(mid, controlnet):
                logger.debug(
                    "Auto-detected runner=%s for model_id=%r controlnet=%r",
                    spec.name,
                    model_id,
                    controlnet,
                )
                return spec
        known = ", ".join(sorted({s.name for s in self._specs}))
        raise ValueError(
            f"No runner matched model_id={model_id!r} controlnet={controlnet!r}. "
            f"Pass --runner explicitly. Known: {known}"
        )

    @property
    def names(self) -> List[str]:
        return list(self._by_name)


_REGISTRY = RunnerRegistry()


def register(spec: RunnerSpec) -> None:
    _REGISTRY.register(spec)


def detect_runner_class(
    model_id: str,
    controlnet: Optional[str] = None,
    *,
    runner_name: Optional[str] = None,
) -> Type[BaseRunner]:
    if runner_name:
        return _REGISTRY.get_by_name(runner_name).runner_cls
    return _REGISTRY.detect(model_id, controlnet).runner_cls


def list_runner_names() -> List[str]:
    return _REGISTRY.names


def _register_builtin_runners() -> None:
    from .flux2_klein_runner import Flux2KleinRunner
    from .flux_runner import FluxRunner
    from .kolors_runner import KolorsRunner
    from .nitro_runner import NitroERunner
    from .sd_runner import SDRunner
    from .sd15_controlnet_runner import SD15ControlnetRunner
    from .sd3x_runner import SD3xRunner
    from .sd3x_outpainting_runner import SD3xOutpaintingRunner
    from .sdxl_runner import SDXLRunner

    # Most specific rules first (same ordering idea as model_config.py).
    register(RunnerSpec(
        name="sd15_controlnet",
        runner_cls=SD15ControlnetRunner,
        model_ids=["sd15-controlnet", "sd1.5-controlnet-canny"],
    ))
    register(RunnerSpec(
        name="sd15_controlnet",
        runner_cls=SD15ControlnetRunner,
        model_ids=["stable-diffusion-v1-5", "stable-diffusion-1.5"],
        controlnet_filter="canny",
    ))
    register(RunnerSpec(
        name="sd3x_outpainting",
        runner_cls=SD3xOutpaintingRunner,
        model_ids=[
            "stable-diffusion-3-medium",
            "stable-diffusion-3.5-medium",
            "stable-diffusion-3-5",
            "stable-diffusion-3.5",
        ],
        controlnet_filter=["outpainting", "removal", "inpainting"],
    ))
    register(RunnerSpec(
        name="sd3x_inpainting",
        runner_cls=SD3xOutpaintingRunner,
        model_ids=[
            "stable-diffusion-3-medium",
            "stable-diffusion-3.5-medium",
            "stable-diffusion-3-5",
            "stable-diffusion-3.5",
        ],
        controlnet_filter="inpainting",
    ))
    register(RunnerSpec(
        name="sd3x",
        runner_cls=SD3xRunner,
        model_ids=[
            "stable-diffusion-3-medium",
            "stable-diffusion-3.5-medium",
            "stable-diffusion-3-5",
            "stable-diffusion-3.5",
        ],
    ))
    register(RunnerSpec(
        name="flux2_klein",
        runner_cls=Flux2KleinRunner,
        model_ids=["flux.2-klein"],
    ))
    register(RunnerSpec(
        name="flux",
        runner_cls=FluxRunner,
        model_ids=["flux.1-schnell", "flux.1-dev", "flux.1"],
    ))
    register(RunnerSpec(
        name="kolors",
        runner_cls=KolorsRunner,
        model_ids=["kolors"],
    ))
    register(RunnerSpec(
        name="nitro",
        runner_cls=NitroERunner,
        model_ids=["nitro-e"],
    ))
    register(RunnerSpec(
        name="sdxl",
        runner_cls=SDXLRunner,
        model_ids=[
            "sdxl-turbo",
            "stable-diffusion-xl-base-1.0",
            "sdxl-base",
            "playground-v2.5",
            "playground2.5",
            "dreamshaper-xl-lightning",
            "ssd-1b",
            "segmind-vega",
        ],
    ))
    register(RunnerSpec(
        name="sd15",
        runner_cls=SDRunner,
        model_ids=[
            "stable-diffusion-v1-5",
            "stable-diffusion-1.5",
            "sd-turbo",
            "stable-diffusion-2-1-base",
            "stable-diffusion-2-1",
        ],
    ))


_register_builtin_runners()
