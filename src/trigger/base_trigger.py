# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# pr-231: new file -- shared trigger base class with profiling template and EP setup
"""Shared Trigger logic: profiling template and execution providers."""

from __future__ import annotations

import copy
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils import common
from src.utils.model_loader import LoadMetricsCollector

logger = logging.getLogger(__name__)
_trigger_logger = logging.getLogger("genai_sd.trigger")


# ---------------------------------------------------------------------------
# Structured profiling result
# ---------------------------------------------------------------------------

@dataclass
class ProfilingResult:
    """Holds all metrics captured during a profiling run."""

    model_id: str
    model_path: str
    width: int
    height: int
    warmup_time_s: float
    avg_time_s: float
    avg_memory_mb: float
    t_total: float
    total_mem: float
    profiling_rounds: int
    # Unified perf dicts: {name: {"time": [...], "device": "NPU"|"CPU"}}
    perf_time_dict_warmup: Dict[str, dict] = field(default_factory=dict)
    perf_time_dict: Dict[str, dict] = field(default_factory=dict)
    load_metrics_collector: Optional[LoadMetricsCollector] = None

    @staticmethod
    def _entry_times(perf_dict: Optional[dict], name: str) -> Optional[list]:
        """Extract the timing list for *name* from a unified perf dict.

        Tolerates the legacy shape where the value was a plain list of times.
        """
        entry = (perf_dict or {}).get(name)
        if isinstance(entry, dict):
            return entry.get("time")
        return entry

    @staticmethod
    def _entry_device(perf_dict: Optional[dict], name: str) -> Optional[str]:
        entry = (perf_dict or {}).get(name)
        if isinstance(entry, dict):
            return entry.get("device")
        return None

    def save_to_excel(self, save_path: str) -> None:
        """Save this ProfilingResult to *save_path* as an Excel file."""
        import pandas as pd

        def fmt(val):
            try:
                return round(val, 4)
            except Exception:
                return ""

        rounds = self.profiling_rounds or 1
        lmc = self.load_metrics_collector
        npu_load_s = lmc.device_time.get("NPU", 0.0) if lmc else 0.0
        npu_mem_mb = (
            sum(v for (_, dev), v in lmc.mem_map.items() if dev == "NPU")
            if lmc else 0.0
        )

        row: dict = {
            "Model ID":                              self.model_id,
            "Height":                                self.height,
            "Width":                                 self.width,
            "Load time of NPU models (s)":           fmt(npu_load_s),
            "Pipeline Time (1st Gen, warm-up) (s)":  fmt(self.warmup_time_s),
            "Pipeline Time (Excl. 1st) (s)":         fmt(self.t_total / rounds),
        }
        for model_name in (self.perf_time_dict or {}):
            times = self._entry_times(self.perf_time_dict, model_name)
            if isinstance(times, list) and len(times) > 0:
                averaged = sum(times) / len(times)
                row[f"{model_name} (s)"] = fmt(averaged)
        row["Total NPU Memory Usage (GB)"] = fmt(npu_mem_mb / 1024)
        row["Total Memory Usage (GB)"] = fmt(self.total_mem / rounds / 1024)

        pd.DataFrame([row]).to_excel(save_path, index=False)
        logger.info(f"[Excel saved] {save_path}")

    def print_summary(self):
        rounds = self.profiling_rounds or 1
        lmc = self.load_metrics_collector
        lmc.print_summary()
        npu_mem_mb = (
            sum(v for (_, dev), v in lmc.mem_map.items() if dev == "NPU")
            if lmc else 0.0
        )

        def _avg(times: list | None) -> float:
            if isinstance(times, list) and times:
                return sum(times) / len(times)
            return 0.0

        rows: List[Tuple[str, str, float, float]] = []
        all_names = set((self.perf_time_dict or {}).keys()) | set(
            (self.perf_time_dict_warmup or {}).keys()
        )
        for name in sorted(all_names):
            warmup_time = round(_avg(self._entry_times(self.perf_time_dict_warmup, name)), 6)
            avg_time = round(_avg(self._entry_times(self.perf_time_dict, name)), 6)
            device = str(
                self._entry_device(self.perf_time_dict, name)
                or self._entry_device(self.perf_time_dict_warmup, name)
                or "CPU"
            )
            if warmup_time > 0.0 or avg_time > 0.0:
                rows.append((name, device, warmup_time, avg_time))
        rows.sort(key=lambda r: ({"CPU": 0, "NPU": 1}.get(r[1], 2), r[0]))

        warm_col = "WARM_UP_TIME(s)"
        avg_col = "AVG_TIME(s)"
        max_model_len = len("MODEL")
        if rows:
            max_model_len = max(max_model_len, max(len(model) for model, _, _, _ in rows))
        header_width = max_model_len + 8 + len(warm_col) + len(avg_col) + 6
        
        title = "Pipeline Performance Summary"
        title_space = (header_width - len(title)) // 2 - 1
        logger.info(f"{"=" * (title_space)} {title} {"=" * (title_space)}")
        logger.info(
            f"{'MODEL':<{max_model_len}} {'DEVICE':<8} "
            f"{warm_col:<{len(warm_col)}} {avg_col}"
        )
        logger.info("-" * header_width)

        for model, device, warm_t, avg_t in rows:
            logger.info(
                f"{model:<{max_model_len}} {device:<8} "
                f"{warm_t:<{len(warm_col)}.6f} {avg_t:.6f}"
            )

        logger.info("-" * header_width)

        avg_total_mem = self.total_mem / rounds
        logger.info(
            f"Total NPU Memory Usage: {npu_mem_mb:.2f}MB ({(npu_mem_mb / 1024):.2f}GB)"
        )
        logger.info(
            f"Total Memory Usage    : {avg_total_mem:.2f}MB ({(avg_total_mem / 1024):.2f}GB)"
        )
        logger.info(
            f"Pipeline warm-up execution time (1st Gen)      : {self.warmup_time_s:.2f}s"
        )
        logger.info(
            f"Pipeline average execution time (excl. 1st Gen): {self.t_total / rounds:.2f}s"
        )
        logger.info("=" * header_width)
# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalize_pipe_output(result: Any) -> list:
    """Normalize diffusers pipeline output to a flat image list."""
    if hasattr(result, "images"):
        return result.images
    if isinstance(result, tuple) and result:
        img0 = result[0]
        return img0 if isinstance(img0, list) else [img0]
    return result if isinstance(result, list) else [result]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseOnnxPipelineTrigger(ABC):
    """Base class for ONNX pipeline triggers.

    Subclasses must implement:
      - ``build_pipe()`` – return ``(pipe_obj, run_fn)`` where ``pipe_obj``
        is the diffusers pipeline (used for per-op timing) and ``run_fn``
        is a zero-argument closure that executes one inference call.
      - ``run(...)`` – store per-call params as ``self._*``, then delegate
        via ``return super().run(height, width, enable_profile=...)``.

    ``model_id`` must be assigned before any profiling call.
    """

    profiling_rounds: int = 1

    # Common attributes set from args — eliminates 6-line copy-paste in every trigger
    def __init__(self, args=None) -> None:
        self.pipeline_metrics: Dict[str, ProfilingResult] = {}
        self.load_metrics_collector = LoadMetricsCollector()
        self.execution_providers: List[str] = self._get_execution_providers()
        if args is not None:
            self.model_id = args.model_id
            self.enable_profile = args.enable_profile
            self.profiling_rounds = args.profiling_rounds

    # ------------------------------------------------------------------
    # Template method (invoked by subclass via super().run())
    # ------------------------------------------------------------------

    def run(self, pipe_obj: Any, run_fn: Callable[[], Any], height: int, width: int, enable_profile: bool = False) -> list:  # type: ignore[override]
        if enable_profile:
            return self._run_with_profiling(pipe_obj, run_fn, height, width)
        return self._run_once(run_fn)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _run_once(self, run_fn: Callable[[], Any]) -> list:
        """Execute run_fn() once and log elapsed wall time."""
        t0 = time.perf_counter()
        images = _normalize_pipe_output(run_fn())
        self.load_metrics_collector.print_summary()
        logger.info("Pipeline execution time = %.6fs", time.perf_counter() - t0)
        return images

    def _run_with_profiling(
        self,
        pipe_obj: Any,
        run_fn: Callable[[], Any],
        height: int,
        width: int,
    ) -> list:
        log = self._make_log_adapter()

        log.info("Running warm-up round...")
        t_warmup = time.perf_counter()
        images = _normalize_pipe_output(run_fn())
        warmup_time = time.perf_counter() - t_warmup
        log.debug("Warm-up mem: %.2fMB", common.measure_mem())

        perf_time_dict_warmup = copy.deepcopy(getattr(pipe_obj, "perf_time_dict", {}))
        if hasattr(pipe_obj, "_clear_time_dict"):
            pipe_obj._clear_time_dict()

        log.info("Running %d profiling rounds...", self.profiling_rounds)
        total_mem = 0.0
        t_start = time.perf_counter()
        for idx in range(self.profiling_rounds):
            total_mem += common.measure_mem()
            _normalize_pipe_output(run_fn())
            log.info("Round %d mem: %.2fMB", idx + 1, common.measure_mem())
            log.info("__ROUND_COMPLETE__ %d/%d", idx + 1, self.profiling_rounds)
        t_total = time.perf_counter() - t_start

        avg_time = t_total / self.profiling_rounds
        avg_mem = total_mem / self.profiling_rounds

        metrics_key = f"h{height}_w{width}"
        self.pipeline_metrics[metrics_key] = ProfilingResult(
            model_id=getattr(self, "model_id", "unknown"),
            model_path=getattr(
                self, "root_model_path", getattr(self, "model_path", "unknown")
            ),
            width=width,
            height=height,
            warmup_time_s=warmup_time,
            avg_time_s=avg_time,
            avg_memory_mb=avg_mem,
            t_total=t_total,
            total_mem=total_mem,
            profiling_rounds=self.profiling_rounds,
            perf_time_dict_warmup=perf_time_dict_warmup,
            perf_time_dict=copy.deepcopy(getattr(pipe_obj, "perf_time_dict", {})),
            load_metrics_collector=self.load_metrics_collector,
        )
        self.pipeline_metrics[metrics_key].print_summary()

        return images

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_execution_providers() -> List[str]:
        if os.environ.get("ORT_DISABLE_GPU", "0") == "1":
            return ["CPUExecutionProvider"]
        else:
            return ["RyzenAILightExecutionProvider", "CPUExecutionProvider"]

    def _make_log_adapter(self) -> logging.LoggerAdapter:
        extra = {
            k: getattr(self, k)
            for k in ("model_id", "model_path")
            if getattr(self, k, None) is not None
        }
        return logging.LoggerAdapter(_trigger_logger, extra)
