# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# pr-231: new file -- mixin providing properties and utility methods shared across all AMD ONNX pipelines

import functools
import inspect
from typing import List, Optional, Union

import torch
from diffusers.utils.torch_utils import randn_tensor


class AMDPipelineMixin:
    """Shared properties and helpers for AMD ONNX pipeline classes.

    Eliminates copy-pasted boilerplate that was duplicated across 5-8 pipeline files.
    Subclasses may override any method when variant-specific logic is needed.
    """

    # ---- simple property getters (were copy-pasted in 5-8 files each) ----

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    # ---- performance tracking (was identical in 6/8 files) ----

    @staticmethod
    def _detect_module_device(module) -> str:
        if module is None:
            return "CPU"
        perf_device = getattr(module, "perf_device", None)
        if perf_device is not None and isinstance(perf_device, str):
            return perf_device
        if hasattr(module, "model") and hasattr(module.model, "get_providers"):
            if "RyzenAILightExecutionProvider" in module.model.get_providers():
                return "NPU"
        return "CPU"

    @property
    def perf_time_dict(self) -> dict:
        d = self.__dict__.get("_perf_time_dict")
        if d is None:
            d = {}
            self.__dict__["_perf_time_dict"] = d
        return d

    @perf_time_dict.setter
    def perf_time_dict(self, value: dict) -> None:
        self.__dict__["_perf_time_dict"] = value

    @staticmethod
    def merge_perf_timings(func):
        """Decorate a method so repeated ``_record_timing`` calls for the same
        component during that call fold into a single slot.

        Use on ``encode_prompt``-style methods where a component runs several
        times per generation (positive + negative prompt, or multiple CLIP
        encoders sharing one key) but should be reported as one combined time.
        Calls made outside such a method (e.g. per denoise step) are unaffected
        and remain one slot each, preserving per-step statistics.
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            prev = getattr(self, "_perf_merge_seen", None)
            self._perf_merge_seen = set()
            try:
                return func(self, *args, **kwargs)
            finally:
                self._perf_merge_seen = prev
        return wrapper

    def _record_timing(self, module_ref, elapsed: float) -> None:
        if module_ref is None:
            return
        model_name = getattr(module_ref, "model_name", None) or type(module_ref).__name__
        entry = self.perf_time_dict.get(model_name)
        if entry is None:
            entry = {"time": [], "device": self._detect_module_device(module_ref)}
            self.perf_time_dict[model_name] = entry
        # Inside a ``merge_perf_timings`` scope, fold repeated calls for the same
        # component into one slot; otherwise every call is its own slot.
        seen = getattr(self, "_perf_merge_seen", None)
        if seen is not None and model_name in seen:
            entry["time"][-1] += elapsed
        else:
            entry["time"].append(elapsed)
            if seen is not None:
                seen.add(model_name)

    def _clear_time_dict(self):
        for entry in self.perf_time_dict.values():
            entry["time"] = []

    # ---- perf dict read helpers ----

    def perf_times(self, name: str) -> list:
        """Return the timing list recorded for *name* (empty list if absent)."""
        entry = self.perf_time_dict.get(name)
        return entry["time"] if entry else []

    def perf_device(self, name: str) -> str:
        """Return the runtime device recorded for *name* ('CPU' if absent)."""
        entry = self.perf_time_dict.get(name)
        return entry["device"] if entry else "CPU"

    def perf_times_by_device(self, device: str) -> dict:
        """Return ``{name: time_list}`` for components on *device*."""
        return {
            name: entry["time"]
            for name, entry in self.perf_time_dict.items()
            if entry.get("device") == device
        }

    # ---- latent preparation (was identical in 6/8 files) ----

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        # pr-231: FlowMatchEulerDiscreteScheduler (SD3.5) does not have init_noise_sigma
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    # ---- scheduler step kwargs (was identical in 3/4 files) ----

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # ---- guidance scale embedding (was identical in 3 files) ----

    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
