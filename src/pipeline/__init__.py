# Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
"""Model-specific ONNX pipeline implementations."""

from .pipeline_flux_onnx_amd import OnnxFluxPipelineAMD
from .pipeline_controlnet_onnx_amd import StableDiffusionControlNetONNXPipelineAMD
from .pipeline_stable_diffusion_3_controlnet_onnx_amd import (
    OnnxStableDiffusion3ControlNetPipelineAMD,
)
from .pipeline_stable_diffusion_3_controlnet_outpainting_onnx_amd import (
    StableDiffusion3ControlNetOutpaintingPipeline,
)
from .pipeline_stable_diffusion_xl_onnx_amd import StableDiffusionXLPipelineAMD
from .pipeline_stable_diffusion_onnx_amd import StableDiffusionONNXPipelineAMD
from .pipeline_stable_diffusion_Kolors_onnx_amd import StableDiffusionKolorsPipelineAMD
from .pipeline_stable_diffusion_NitroE_onnx_amd import StableDiffusionNitroEONNXPipelineAMD
from .pipeline_stable_diffusion_flux2_klein_onnx_amd import (
    StableDiffusionFlux2KleinPipelineAMD,
)

__all__ = [
    "OnnxFluxPipelineAMD",
    "StableDiffusionControlNetONNXPipelineAMD",
    "OnnxStableDiffusion3ControlNetPipelineAMD",
    "StableDiffusion3ControlNetOutpaintingPipeline",
    "StableDiffusionXLPipelineAMD",
    "StableDiffusionONNXPipelineAMD",
    "StableDiffusionKolorsPipelineAMD",
    "StableDiffusionNitroEONNXPipelineAMD",
    "StableDiffusionFlux2KleinPipelineAMD",
]
