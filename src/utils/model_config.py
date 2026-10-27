# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
GenAI-SD configuration: inference defaults, runner ``model_id`` profiles, SD3 / SD3.5 ControlNet CLI.

Architecture
------------
``ModelRegistry`` is the central registry that owns all ``ModelSpec`` entries.
Each ``ModelSpec`` is a **single declarative record** that bundles:

- ``model_ids``        — substrings used to recognise the model from ``model_id``
- ``defaults``         — an ``InferenceConfig`` with preset inference parameters
- ``controlnet_filter``— optional controlnet value constraint (``None`` = any)

``detect_runner_model_type`` returns the matching ``ModelSpec`` directly, so callers
access ``spec.defaults`` in one step — no separate ``RUNNER_DEFAULTS`` lookup needed.

**To add a new model variant**, call ``register()`` once — no other edits needed::

    from src.genai_sd.common.config import register, ModelSpec, InferenceConfig

    register(ModelSpec(
        model_ids=["my-model-v1"],
        defaults=InferenceConfig(height=1024, width=1024, guidance_scale=7.5, ...),
    ))

Public API
----------
- ``InferenceConfig``        — dataclass for inference hyper-parameters
- ``SD3ControlNetConfig``   — ``InferenceConfig`` subclass for SD3 ControlNet (guidance=7.0, cond=0.5)
- ``SD35ControlNetConfig``  — ``InferenceConfig`` subclass for SD3.5 ControlNet (guidance=5.0, cond=0.7)
- ``ModelSpec``              — single model variant declaration
- ``ModelRegistry``          — registry class (access via the singleton ``_REGISTRY``)
- ``register(spec)``         — register a new ``ModelSpec`` at module or call-site level
- ``detect_runner_model_type(model_id, controlnet)`` → ``ModelSpec``
- ``MODEL_SPECS``            — ordered list of all registered specs (derived view)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-relative path helper
# ---------------------------------------------------------------------------

# src/utils/config.py → parents[2] = <project_root>
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_project_path(sub_path: str) -> str:
    absolute_path = _PROJECT_ROOT / sub_path
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"File not found: {absolute_path}")
    return str(absolute_path.resolve())


# ---------------------------------------------------------------------------
# Inference config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """Inference hyper-parameters (maps directly to ``runner_args`` fields)."""

    height: int = 512
    width: int = 512
    prompt: str = (
        "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, "
        "battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, "
        "unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner."
    )
    n_prompt: str = ""
    num_inference_steps: int = 20
    num_images_per_prompt: int = 1
    guidance_scale: float = 7.5
    t5_sequence_len: Optional[int] = None
    seed: Optional[int] = None
    strength: Optional[float] = None
    controlnet: Optional[str] = None
    controlnet_conditioning_scale: Optional[float] = None
    control_image_path: Optional[str] = None
    control_mask_path: Optional[str] = None
    tea_caching_delta: Optional[float] = None
    tea_caching_coef: Optional[List[float]] = None
    common_model_path: str = "common/"
    sub_model_path: str = "normal/"
    prompt_file_path: Optional[str] = _get_project_path("config/prompts_config.json")

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {self.num_inference_steps}")
        if self.num_images_per_prompt <= 0:
            raise ValueError(f"num_images_per_prompt must be positive, got {self.num_images_per_prompt}")
        if self.guidance_scale < 0:
            raise ValueError(f"guidance_scale must be >= 0, got {self.guidance_scale}")


# Backward-compatible alias used by ``genai_sd.common.__init__``.
Text2ImageInferenceConfig = InferenceConfig


@dataclass
class SD3ControlNetConfig(InferenceConfig):
    """SD3 ControlNet defaults (stable-diffusion-3-medium).

    guidance_scale=7.0, controlnet_conditioning_scale=0.5
    """

    height: int = 512
    width: int = 512
    num_inference_steps: int = 8
    guidance_scale: float = 7.0
    num_images_per_prompt: int = 1
    prompt: str = (
        "Anime style illustration of a girl wearing a suit. A moon in sky. "
        "In the background we see a big rain approaching. text 'InstantX' on image"
    )
    n_prompt: str = "NSFW, nude, naked, porn, ugly"
    seed: int = 42
    controlnet_conditioning_scale: float = 0.5
    t5_sequence_len: int = 83
    tea_caching_delta: float = 0.15
    tea_caching_coef: List[float] = field(default_factory=lambda: [1.0, 0.0])


@dataclass
class SD35ControlNetConfig(InferenceConfig):
    """SD3.5 ControlNet defaults (stable-diffusion-3.5-*).

    Differs from SD3: guidance_scale=5.0, controlnet_conditioning_scale=0.7
    """

    height: int = 512
    width: int = 512
    num_inference_steps: int = 8
    guidance_scale: float = 5.0
    num_images_per_prompt: int = 1
    prompt: str = (
        "Anime style illustration of a girl wearing a suit. A moon in sky. "
        "In the background we see a big rain approaching. text 'InstantX' on image"
    )
    n_prompt: str = "NSFW, nude, naked, porn, ugly"
    seed: int = 42
    controlnet_conditioning_scale: float = 0.7
    t5_sequence_len: int = 83
    tea_caching_delta: float = 0.15
    tea_caching_coef: List[float] = field(default_factory=lambda: [1.0, 0.0])


# ---------------------------------------------------------------------------
# ControlNet normalisation helper
# ---------------------------------------------------------------------------


def _cn_equals(controlnet: Optional[str], want: str) -> bool:
    """True when ``controlnet`` matches the non-None ``want`` string.

    ``want == "none"`` also matches ``None``, empty string, or the literal ``"none"``,
    mirroring the CLI ``--controlnet`` default behaviour.

    Note: only call this when ``want`` is not ``None``; the ``None``-means-any-controlnet
    short-circuit lives in ``ModelSpec.matches``.
    """
    if controlnet is None:
        return want == "none"
    return str(controlnet).strip().lower() == want


# ---------------------------------------------------------------------------
# ModelSpec — declarative record for one model variant
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Complete declaration for one model variant.

    Attributes
    ----------
    model_ids:
        One or more substrings; **at least one** must appear anywhere inside the
        lower-cased ``model_id`` string for this spec to match.
    defaults:
        Default ``InferenceConfig`` used when no CLI overrides are provided.
    controlnet_filter:
        Controlnet constraint.  ``None`` (default) matches any controlnet value.
        Otherwise the normalised controlnet string must equal this value
        (``"none"`` also matches unset / empty).
    """

    model_ids: List[str]
    defaults: InferenceConfig
    controlnet_filter: Optional[str] = None

    def matches(self, model_id_lower: str, controlnet: Optional[str]) -> bool:
        """Return ``True`` when both the model-id and controlnet constraints are satisfied.

        Model-id matching uses substring search: a spec matches when **any** of its
        registered ``model_ids`` keys appears anywhere inside ``model_id_lower``.
        This lets callers pass full HuggingFace paths (e.g. ``"stabilityai/stable-diffusion-3.5-medium"``)
        without stripping the organisation prefix first.
        """
        if not any(p in model_id_lower for p in self.model_ids):
            return False
        if self.controlnet_filter is None:
            return True
        return _cn_equals(controlnet, self.controlnet_filter)

    def __repr__(self) -> str:
        cn = self.controlnet_filter if self.controlnet_filter is not None else "<any>"
        return f"ModelSpec(model_ids={self.model_ids!r}, controlnet={cn!r})"


# ---------------------------------------------------------------------------
# ModelRegistry — ordered registry with first-match semantics
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Ordered registry of :class:`ModelSpec` entries.

    Resolution uses first-match semantics: the first registered spec whose
    :meth:`ModelSpec.matches` returns ``True`` wins.

    Usage::

        # Detect and get defaults in one step:
        spec = _REGISTRY.detect("stable-diffusion-v1-5_amdnpu")
        if spec is not None:
            cfg = spec.defaults

        # Extend at runtime (e.g. from a plugin):
        _REGISTRY.register(ModelSpec(model_ids=["my-model"], defaults=InferenceConfig(...)))
    """

    def __init__(self) -> None:
        self._specs: List[ModelSpec] = []

    # ── Registration ────────────────────────────────────────────────────────

    def register(self, spec: ModelSpec) -> None:
        """Append ``spec`` to the registry (lower priority than previously registered specs)."""
        self._specs.append(spec)
        logger.debug("Registered %r", spec)

    # ── Resolution ──────────────────────────────────────────────────────────

    def detect(
        self,
        model_id: Optional[str],
        controlnet: Optional[str] = None,
    ) -> ModelSpec:
        """Return the first matching ``ModelSpec``, or ``None``.

        Returns ``None`` for falsy ``model_id`` and logs a warning when a non-empty
        ``model_id`` matches nothing.  Callers that need to distinguish between
        "empty" and "unknown" can check ``bool(model_id)`` first.
        """
        if not model_id:
            raise ValueError("model_id is required")
        mid = model_id.lower()
        for spec in self._specs:
            if spec.matches(mid, controlnet):
                return spec
        raise ValueError(
            f"No ModelSpec matched model_id={model_id} controlnet={controlnet}"
        )

    # ── Read-only views ─────────────────────────────────────────────────────

    @property
    def specs(self) -> List[ModelSpec]:
        """Ordered snapshot of all registered specs (read-only copy)."""
        return list(self._specs)

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"ModelRegistry({len(self._specs)} specs)"


# Module-level singleton — the authoritative registry for this process.
_REGISTRY = ModelRegistry()


def register(spec: ModelSpec) -> None:
    """Register a :class:`ModelSpec` in the module-level :data:`_REGISTRY`.

    Call this from any module that needs to add a new model variant without
    modifying this file::

        from src.genai_sd.common.config import register, ModelSpec, InferenceConfig

        register(ModelSpec(
            model_ids=["my-model-v1"],
            defaults=InferenceConfig(height=1024, width=1024, ...),
        ))
    """
    _REGISTRY.register(spec)


# ---------------------------------------------------------------------------
# Built-in model registrations
# ---------------------------------------------------------------------------
# ── SD 1.x ──────────────────────────────────────────────────────────────────

register(ModelSpec(
    model_ids=["sd15-controlnet", "sd1.5-controlnet-canny"],
    defaults=InferenceConfig(
        height=512, width=512,
        num_inference_steps=20, guidance_scale=7.5,
        prompt="a blue paraidse bird in the jungle",
        controlnet="canny", controlnet_conditioning_scale=1.5,
        control_image_path=_get_project_path("test/ref/control.png"),
    ),
))

register(ModelSpec(
    model_ids=["stable-diffusion-v1-5", "stable-diffusion-1.5"],
    controlnet_filter="canny",
    defaults=InferenceConfig(
        height=512, width=512,
        num_inference_steps=20, guidance_scale=7.5,
        prompt="a blue paraidse bird in the jungle",
        controlnet="canny", controlnet_conditioning_scale=1.5,
        control_image_path=_get_project_path("test/ref/control.png"),
    ),
))

register(ModelSpec(
    model_ids=["stable-diffusion-v1-5", "stable-diffusion-1.5"],
    defaults=InferenceConfig(
        height=512, width=512,
        num_inference_steps=20, guidance_scale=7.5,
        prompt=(
            "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, "
            "battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, "
            "unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner"
        ),
    ),
))

register(ModelSpec(
    model_ids=["sd-turbo"],
    defaults=InferenceConfig(
        height=512, width=512,
        num_inference_steps=1, guidance_scale=0.0,
        prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    ),
))

register(ModelSpec(
    model_ids=["stable-diffusion-2-1-base"],
    defaults=InferenceConfig(
        height=512, width=512,
        num_inference_steps=50, guidance_scale=7.5,
        prompt="A photo of an astronaut riding a horse on mars.",
    ),
))

register(ModelSpec(
    model_ids=["stable-diffusion-2-1"],
    defaults=InferenceConfig(
        height=768, width=768,
        num_inference_steps=50, guidance_scale=7.5,
        prompt="A photo of an astronaut riding a horse on mars.",
    ),
))

# ── SD3 / SD3.5 ControlNet variants (must precede the base SD3x catch-all) ──

register(ModelSpec(
    model_ids=["stable-diffusion-3-medium"],
    controlnet_filter="outpainting",
    defaults=SD3ControlNetConfig(
        height=1024, width=1024,
        prompt="background, nothing, blank", n_prompt="",
        controlnet="outpainting",
        control_image_path=_get_project_path("test/ref/outpainting.jpg"),
        seed=0, controlnet_conditioning_scale=0.95, guidance_scale=5.0,
    ),
))

# -- Shared: removal (guidance=5.0, cond=0.95) -----------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3-medium", "stable-diffusion-3.5-medium", "stable-diffusion-3-5", "stable-diffusion-3.5"],
    controlnet_filter="removal",
    defaults=SD3ControlNetConfig(
        height=1024, width=1024,
        prompt="background, nothing, blank", n_prompt="",
        controlnet="removal",
        control_image_path=_get_project_path("test/ref/removal/origin.jpg"),
        control_mask_path=_get_project_path("test/ref/removal/mask.jpg"),
        seed=0, controlnet_conditioning_scale=0.95, guidance_scale=5.0,
    ),
))

# -- Shared: depth (guidance=7.0, cond=0.5) --------------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3-medium", "stable-diffusion-3.5-medium", "stable-diffusion-3-5", "stable-diffusion-3.5"],
    controlnet_filter="depth",
    defaults=SD3ControlNetConfig(
        prompt=(
            "a panda cub, captured in a close-up, in forest, is perched on a tree trunk. "
            "good composition, Photography, the cub's ears, a fluffy black, are tucked behind its head, "
            "adding a touch of whimsy to its appearance. a lush tapestry of green leaves in the background. "
            "depth of field, National Geographic"
        ),
        n_prompt="bad hands, blurry, NSFW, nude, naked, porn, ugly, bad quality, worst quality",
        controlnet="depth",
        control_image_path=_get_project_path("test/ref/depth.jpeg"),
    ),
))

# -- Shared: union (guidance=3.5, cond=0.7) --------------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3-medium", "stable-diffusion-3.5-medium", "stable-diffusion-3-5", "stable-diffusion-3.5"],
    controlnet_filter="union",
    defaults=SD3ControlNetConfig(
        prompt="a lovely dog", n_prompt="",
        controlnet="union",
        control_image_path=_get_project_path("test/ref/union.jfif"),
        seed=0, controlnet_conditioning_scale=0.7, guidance_scale=3.5,
    ),
))

# -- SD3.5: inpainting (guidance=5.0, cond=0.7) ----------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3.5-medium", "stable-diffusion-3-5", "stable-diffusion-3.5"],
    controlnet_filter="inpainting",
    defaults=SD35ControlNetConfig(
        height=1024, width=1024,
        prompt="holiday", n_prompt="",
        controlnet="inpainting",
        sub_model_path="inpainting/",
        control_mask_path=_get_project_path("test/ref/inpainting/mask.jpg"),
        control_image_path=_get_project_path("test/ref/inpainting/origin.jpg"),
        seed=0, controlnet_conditioning_scale=0.7, guidance_scale=5.0,
        tea_caching_delta=0.25,
        tea_caching_coef=[498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861],
    ),
))

# -- SD3: inpainting (guidance=7.0, cond=0.7) ------------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3-medium"],
    controlnet_filter="inpainting",
    defaults=SD3ControlNetConfig(
        height=1024, width=1024,
        prompt="holiday", n_prompt="",
        controlnet="inpainting",
        sub_model_path="inpainting/",
        control_mask_path=_get_project_path("test/ref/inpainting/mask.jpg"),
        control_image_path=_get_project_path("test/ref/inpainting/origin.jpg"),
        seed=0, controlnet_conditioning_scale=0.7, guidance_scale=7.0,
        tea_caching_delta=0.15,
        tea_caching_coef=[1.0, 0.0],
    ),
))

# -- SD3.5: canny (guidance=5.0, cond=0.7) ---------------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3.5-medium", "stable-diffusion-3-5", "stable-diffusion-3.5"],
    controlnet_filter="canny",
    defaults=SD35ControlNetConfig(
        controlnet="canny",
        control_image_path=_get_project_path("test/ref/canny.jpg"),
    ),
))

# -- SD3: canny (guidance=7.0, cond=0.5) -----------------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3-medium"],
    controlnet_filter="canny",
    defaults=SD3ControlNetConfig(
        controlnet="canny",
        control_image_path=_get_project_path("test/ref/canny.jpg"),
    ),
))

# -- SD3.5: tile (guidance=5.0, cond=0.7) ----------------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3.5-medium", "stable-diffusion-3-5", "stable-diffusion-3.5"],
    controlnet_filter="tile",
    defaults=SD35ControlNetConfig(
        controlnet="tile",
        control_image_path=_get_project_path("test/ref/tile.jpg"),
    ),
))

# -- SD3: tile (guidance=7.0, cond=0.5) ------------------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3-medium"],
    controlnet_filter="tile",
    defaults=SD3ControlNetConfig(
        controlnet="tile",
        control_image_path=_get_project_path("test/ref/tile.jpg"),
    ),
))

# -- SD3.5: pose (guidance=5.0, cond=0.7; 0.8 at height=512 via runner_args) -

register(ModelSpec(
    model_ids=["stable-diffusion-3.5-medium", "stable-diffusion-3-5", "stable-diffusion-3.5"],
    controlnet_filter="pose",
    defaults=SD35ControlNetConfig(
        controlnet="pose",
        control_image_path=_get_project_path("test/ref/pose.jpg"),
    ),
))

# -- SD3: pose (guidance=7.0, cond=0.5) ------------------------------------

register(ModelSpec(
    model_ids=["stable-diffusion-3-medium"],
    controlnet_filter="pose",
    defaults=SD3ControlNetConfig(
        controlnet="pose",
        control_image_path=_get_project_path("test/ref/pose.jpg"),
    ),
))

# -- SD3.5: t2i catch-all (guidance=4.5, seed=45, prompt=capybara) ----------

register(ModelSpec(
    model_ids=["stable-diffusion-3.5-medium", "stable-diffusion-3-5", "stable-diffusion-3.5"],
    defaults=SD35ControlNetConfig(
        height=512, width=512,
        num_inference_steps=8, guidance_scale=4.5,
        prompt="A capybara holding a sign that reads Hello World",
        n_prompt="",
        seed=45,
        tea_caching_delta=0.25,
        tea_caching_coef=[498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861],
    ),
))

# -- SD3: t2i catch-all (guidance=3.0, seed=0, prompt=sailing ship) ---------

register(ModelSpec(
    model_ids=["stable-diffusion-3-medium"],
    defaults=InferenceConfig(
        height=512, width=512,
        num_inference_steps=8, guidance_scale=3.0,
        t5_sequence_len=83,
        prompt=(
            "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, "
            "cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, "
            "extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, "
            "intricate artwork by john william turner"
        ),
        prompt_file_path=_get_project_path("config/prompts_config.json"),
        n_prompt="NSFW, nude, naked, porn, ugly",
        seed=0,
        tea_caching_delta=0.1,
        tea_caching_coef=[1.0, 0.0],
    ),
))

# ── SDXL family ──────────────────────────────────────────────────────────────

register(ModelSpec(
    model_ids=["sdxl-turbo"],
    defaults=InferenceConfig(
        height=512, width=512,
        num_inference_steps=1, guidance_scale=0.0,
        prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    ),
))

register(ModelSpec(
    model_ids=["playground-v2.5", "playground2.5"],
    defaults=InferenceConfig(
        height=1024, width=1024,
        num_inference_steps=50, guidance_scale=3.0,
        prompt="A highly detailed cinematic scene, rich colors, high contrast.",
    ),
))

register(ModelSpec(
    model_ids=["dreamshaper-xl-lightning"],
    defaults=InferenceConfig(
        height=1024, width=1024,
        num_inference_steps=50, guidance_scale=7.5,
        prompt=(
            "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, "
            "steel metal, elegant, sharp focus, soft lighting, vibrant colors"
        ),
    ),
))

register(ModelSpec(
    model_ids=["ssd-1b"],
    defaults=InferenceConfig(
        height=1024, width=1024,
        num_inference_steps=50, guidance_scale=7.5,
        prompt="An astronaut riding a green horse",
        n_prompt="ugly, blurry, poor quality",
    ),
))

register(ModelSpec(
    model_ids=["stable-diffusion-xl-base-1.0", "sdxl-base"],
    defaults=InferenceConfig(
        height=1024, width=1024,
        num_inference_steps=50, guidance_scale=7.5,
        prompt="An astronaut riding a green horse",
    ),
))

# ── Kolors ────────────────────────────────────────────────────────────────────

register(ModelSpec(
    model_ids=["kolors"],
    controlnet_filter="canny",
    defaults=InferenceConfig(
        height=1024, width=1024,
        num_inference_steps=50, guidance_scale=6.0,
        prompt="全景，一只可爱的白色小狗坐在杯子里，看向镜头，动漫风格，3d渲染，辛烷值渲染。",
        n_prompt="nsfw，脸部阴影，低分辨率，jpeg伪影、模糊、糟糕，黑脸，霓虹灯",
        controlnet="canny", controlnet_conditioning_scale=0.7,
        control_image_path=_get_project_path("test/ref/Canny_dog_condition.jpg"),
        seed=66, strength=1.0,
    ),
))

register(ModelSpec(
    model_ids=["kolors"],
    controlnet_filter="inpainting",
    defaults=InferenceConfig(
        height=1024, width=1024,
        num_inference_steps=25, guidance_scale=6.0,
        prompt=(
            "穿着钢铁侠的衣服，高科技盔甲，主要颜色为红色和金色，并且有一些银色装饰。"
            "胸前有一个亮起的圆形反应堆装置，充满了未来科技感。超清晰，高质量，超逼真，高分辨率，最好的质量，超级细节，景深"
        ),
        n_prompt="残缺的手指，畸形的手指，畸形的手，残肢，模糊，低质量",
        controlnet="inpainting",
        control_mask_path=_get_project_path("test/ref/inpainting/kolors_mask.png"),
        control_image_path=_get_project_path("test/ref/inpainting/kolors_origin.png"),
        seed=603, strength=0.999,
    ),
))

register(ModelSpec(
    model_ids=["kolors"],
    defaults=InferenceConfig(
        height=1024, width=1024,
        num_inference_steps=50, guidance_scale=5.0,
        prompt="一张瓢虫的照片，微距，变焦，高质量，电影，拿着一个牌子，写着“可图”",
    ),
))

# ── Others ────────────────────────────────────────────────────────────────────

register(ModelSpec(
    model_ids=["nitro-e"],
    defaults=InferenceConfig(
        height=512, width=512,
        num_inference_steps=20, guidance_scale=4.5,
        prompt="A hot air balloon in the shape of a heart grand canyon.",
        seed=0,
    ),
))

register(ModelSpec(
    model_ids=["segmind-vega"],
    defaults=InferenceConfig(
        height=1024, width=1024,
        num_inference_steps=50, guidance_scale=7.5,
        prompt="An astronaut riding a green horse",
        strength=0.3,
    ),
))

# ── FLUX.2 Klein ──────────────────────────────────────────────────────────────

register(ModelSpec(
    model_ids=["flux.2-klein"],
    defaults=InferenceConfig(
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        num_images_per_prompt=1,
        seed=0,
        prompt=(
            "A beautiful anime girl with long silver hair and blue eyes, wearing a flowing white dress. "
            "Standing in a field of flowers under golden sunset. Soft warm lighting, gentle breeze, "
            "petals floating in the air. Highly detailed, delicate face, clean line art, vibrant colors, "
            "dreamy atmosphere."
        ),
        n_prompt="",
    ),
))

# ── Flux ──────────────────────────────────────────────────────────────────────

register(ModelSpec(
    model_ids=["flux.1-schnell", "flux.1-dev", "flux.1"],
    defaults=InferenceConfig(
        height=512, width=512,
        num_inference_steps=4, guidance_scale=0.0,
        num_images_per_prompt=1,
        t5_sequence_len=256,
        seed=0,
        prompt=(
            "A beautiful anime girl with long silver hair and blue eyes, wearing a flowing white dress. "
            "Standing in a field of flowers under golden sunset. Soft warm lighting, gentle breeze, "
            "petals floating in the air. Highly detailed, delicate face, clean line art, vibrant colors, "
            "dreamy atmosphere."
        ),
        n_prompt="",
    ),
))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Ordered snapshot of all registered specs.
MODEL_SPECS: List[ModelSpec] = _REGISTRY.specs


def detect_model_spec(
    model_id: Optional[str],
    controlnet: Optional[str] = None,
) -> ModelSpec:
    """Return the first matching ``ModelSpec`` for ``(model_id, controlnet)``.

    Delegates to :meth:`ModelRegistry.detect` on the module-level :data:`_REGISTRY`.

    Returns ``None`` for falsy ``model_id`` or when no spec matches (a warning is
    logged in the latter case).  Raise :exc:`UnknownModelError` upstream if strict
    behaviour is needed::

        spec = detect_runner_model_type(args.model_id, getattr(args, "controlnet", None))
    """
    return _REGISTRY.detect(model_id, controlnet)
