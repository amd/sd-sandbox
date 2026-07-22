# Glossary

Definitions for key terms used throughout the GenAI-SD documentation.

---

## A

**ATOM**
The AMD NPU runtime Python package (`amd_atom`). It provides the Python bindings and scheduling layer
for running compute workloads on the AMD Ryzen AI NPU. Included in the Ryzen AI Software suite.

---

## B

**BFP16 (Block Floating Point 16)**
A quantization format where a block of numbers shares a single exponent while each number has its
own 16-bit mantissa. BFP16 is the primary quantization format used by the AMD NPU for running
DD model subgraphs. It is more efficient than standard FP16 or FP32 while preserving most of
the model's accuracy.

---

## C

**Cache directory (`cache/`)**
The directory produced by `onnx_utils partition` alongside `replaced.onnx`. It contains compiled
NPU kernel binaries (`.xclbin` files) and subgraph metadata (`.json` files) used by the DD runtime
at inference time. You must keep this directory alongside `replaced.onnx` for inference to work.

**ControlNet**
A neural network architecture that adds spatial control to diffusion models by conditioning the
denoising process on a reference image (edge map, pose skeleton, depth map, etc.). GenAI-SD
supports ControlNet for SD1.5 (Canny) and SD3/SD3.5 (Canny, Tile, Pose, Depth, Union).

---

## D

**DD (Dynamic Dispatch)**
AMD's technology for partitioning ONNX model graphs and running subgraphs on the NPU. The
`onnx_utils partition` tool replaces NPU-eligible subgraphs with a single `DynamicDispatch`
custom operator. See [docs/04-model-generate-guide.md](04-model-generate-guide.md) for the full workflow.

**DiT (Diffusion Transformer)**
A transformer-based diffusion model architecture (as opposed to the older U-Net architecture).
SD3/SD3.5 use MM-DiT (Multimodal DiT) and FLUX.1-schnell uses a bidirectional attention DiT.

**DML (DirectML)**
Microsoft's DirectX 12 API for running GPU compute workloads. GenAI-SD uses DML for iGPU
inference of operators that are not offloaded to the NPU (e.g., the CLIP text encoder in FLUX).

**DD_PLUGINS_ROOT**
An environment variable required for dynamic shape inference. It must point to the directory
containing the compiled DD plugin libraries. Windows: set via `set DD_PLUGINS_ROOT=.\lib\transaction\stx`.
Linux: see [docs/05-linux-guide.md](05-linux-guide.md).

**Dynamic shape**
The ability of a single partitioned model to accept inputs of varying spatial dimensions at
inference time without reloading or recompiling. Dynamic shape support requires partitioning with
a `dynamic_shape_list` and the DD plugin set via `DD_PLUGINS_ROOT`.

---

## F

**FLUX.1-schnell**
A fast, distilled diffusion transformer from Black Forest Labs. Uses a bidirectional attention
architecture with two text encoders (CLIP + T5). The T5 encoder runs on the NPU in GenAI-SD
while the CLIP encoder runs on the iGPU.

**FLUX.2 Klein**
A compact (~4B parameter) diffusion transformer model. Uses a Qwen3 text encoder and supports
both text-to-image and image editing via `--edit_image_path`. The tokenization uses the `Qwen2TokenizerFast` class (Qwen3 models reuse the Qwen2 tokenizer
implementation in Transformers). The distilled variant (`amd/FLUX.2-klein-4B-amdnpu`) generates
1024×1024 images in 4 steps. See [docs/03-model-reference-guide.md](03-model-reference-guide.md) for usage.

---

## G

**GPTQ**
A post-training quantization technique for transformer models that reduces weight precision to
4-bit integers while maintaining accuracy. GenAI-SD uses GPTQ-quantized T5 text encoders
(`text_encoder_3_gptq_v2/`) in SD3/SD3.5 pipelines to reduce memory usage.

---

## I

**iGPU (Integrated GPU)**
The GPU integrated into the CPU die. In Ryzen AI devices, the iGPU is used for operations
that are not offloaded to the NPU, typically via DirectML (DML). The iGPU handles text encoders
(CLIP) and some preprocessing tasks.

**InPainting**
A technique for replacing a masked region of an existing image with new content guided by a
text prompt. The model sees both the unmasked image context and the prompt when filling the
masked region. GenAI-SD supports InPainting for SD3 and SD3.5 via a dedicated model variant.

---

## K

**Krackan / Strix**
Internal AMD code names for supported AMD Ryzen AI processor generations. These determine which
NPU driver and firmware to install. Windows: [docs/01-quick-start.md](01-quick-start.md).
Linux: [docs/05-linux-guide.md](05-linux-guide.md).

---

## L

**latent_dim**
In FLUX.1-schnell, the DiT operates on a 1-D sequence of patchified latent tokens. `latent_dim`
is the total number of tokens: `latent_dim = (H/16) × (W/16)` for an H×W image. For example,
a 1024×1024 image produces `latent_dim = 4096`. This value appears in the partition `dynamic_shape_list`.

**LFS (Git Large File Storage)**
A git extension for storing large binary files (like ONNX model weights) efficiently. Models
downloaded from Hugging Face via `git clone` require `git lfs install` and `git lfs pull` to
download the actual weight files instead of LFS pointer files.

---

## M

**MM-DiT (Multimodal Diffusion Transformer)**
The transformer architecture used in SD3 and SD3.5. It processes image patches and text tokens
in a joint attention layer, allowing rich cross-modal conditioning. The MM-DiT is the primary
compute component of SD3/SD3.5 and is offloaded to the NPU in GenAI-SD.

---

## N

**NPU (Neural Processing Unit)**
The dedicated AI accelerator built into AMD Ryzen AI processors. The NPU is optimized for matrix
multiply and convolution operations in BFP16 format. In GenAI-SD, the NPU runs the fused DD
subgraphs (transformer, UNet, VAE, ControlNet) via the Dynamic Dispatch runtime.

---

## O

**O1 / optimize_o1**
A preset optimization flag (`--optimize_o1` / `-O1`) that enables a curated set of compile-time
and runtime optimizations for the ONNX Runtime fusion engine. Improves throughput on supported
models. See [docs/03-model-reference-guide.md](03-model-reference-guide.md) for models that support it.

**ONNX (Open Neural Network Exchange)**
An open format for representing machine learning models. GenAI-SD uses ONNX models downloaded
from Hugging Face and processed by `onnx_utils` for NPU deployment.

**ONNX Runtime (ORT)**
The inference engine used to run ONNX models. GenAI-SD uses ONNX Runtime with the Ryzen AI
execution provider (`onnxruntime_providers_ryzenai.dll`) which exposes the `DynamicDispatch`
custom operator.

**onnx_utils**
AMD's command-line tool (part of the `ryzenai-onnx-utils` package) for preprocessing and
partitioning ONNX models for NPU deployment. Key commands:
- `onnx_utils preprocess` — graph optimization
- `onnx_utils partition` — NPU subgraph splitting
- `onnx_utils report dd_offload` — verify offload ratio

**OutPainting**
A technique for extending the borders of an existing image outward. The model generates new
content beyond the original image edges. GenAI-SD supports OutPainting for SD3 and SD3.5 via
the ControlNet outpainting pipeline (`python test/run.py --controlnet OutPainting`).
The OutPainting ControlNet weights are not included in the current public Hugging Face release
and must be supplied locally; use InPainting for masked-region editing.

---

## P

**Pipeline Orchestrator**
The batch pipeline runner at `scripts/run_pipelines.py`. It reads pipeline definitions from
`config/pipeline_configs.yaml`, spawns `test/run.py` for each model, captures output, and
reports a pass/fail summary. Supports `--pipelines`, `--pipeline_groups`, `--benchmark`,
`--image-only`, and `--list` options. See [docs/02-user-guide.md](02-user-guide.md).

**ProfilingResult**
A dataclass (`src/trigger/base_trigger.py`) that holds all metrics captured during a profiling
run: warm-up time, per-round timing (`perf_time_dict` for NPU, `perf_time_gpu_model` for CPU),
memory usage, and load metrics. Stored in `trigger.pipeline_metrics["h{H}_w{W}"]` and saved as
an `.xlsx` file by `save_profiling()` in `base_runner.py`.

**Partition strategy YAML**
A configuration file that specifies AMD pass chains and hardware targets for the `onnx_utils partition`
step. Strategy files are bundled inside the `ryzenai-onnx-utils` package at
`<conda_env>/Lib/site-packages/ryzenai_onnx_utils/data/partition_strategies/`.
Pass only the filename to `onnx_utils partition` — see [docs/04-model-generate-guide.md](04-model-generate-guide.md).

---

## R

**Runner**
The outermost layer of the GenAI-SD pipeline architecture (`src/runner/`). A Runner reads CLI
args, creates a Trigger, and orchestrates the prompt × shape loop. All Runners extend
`BaseRunner` and set `trigger_cls`. Simple runners (e.g. `SDRunner`) need only one line;
complex ones (e.g. `SD3xOutpaintingRunner`) override `pre_run()` or `get_save_dimensions()`.
See [docs/07-developer-guide.md](07-developer-guide.md).

**RYZENAI_GENAI_SD_MODELS_ROOT**
An environment variable that overrides the default model download/cache directory
(`<project_root>/models/`). When set, all `--model_id` auto-downloads and cache lookups use
this path. Example:

```powershell
$env:RYZENAI_GENAI_SD_MODELS_ROOT = "D:\my_models"
```

**replaced.onnx**
The output of `onnx_utils partition`. It is an ONNX model identical in interface to `optimized.onnx`
but with NPU-eligible subgraphs replaced by a single `DynamicDispatch` custom node. ONNX Runtime
loads this file at inference time. It must be kept alongside the `cache/` directory.

**Ryzen AI Software**
AMD's software platform for deploying AI models on Ryzen AI processors. The platform includes
the NPU driver, XRT runtime, `onnx_utils`, `ryzenai-dynamic-dispatch`, and ONNX Runtime
execution provider. Installation instructions at https://ryzenai.docs.amd.com/en/latest/inst.html

---

## S

**SD (Stable Diffusion)**
A family of latent diffusion models for image generation. GenAI-SD supports SD1.5, SD-Turbo,
SD2.1, SDXL (base/turbo/Playground 2.5/SSD-1B/DreamShaper XL/Segmind-Vega), SD3 Medium,
SD3.5 Medium, FLUX.1-schnell, and FLUX.2 Klein. All AMD NPU-optimized models
are hosted on Hugging Face under the `amd/` namespace.

**SDXL (Stable Diffusion XL)**
The 1024×1024 native resolution version of Stable Diffusion from Stability AI. Uses a larger
U-Net architecture with dual text encoders. GenAI-SD supports SDXL-base, SDXL-turbo, and
Segmind-Vega (an SDXL distillate).

**ShapeContext**
A dataclass (`src/runner/base_runner.py`) that packages a single inference resolution:
`height`, `width`, and an `extras` dict for model-specific parameters (e.g.
`{"txt_seq_len": 256}` for Flux, `{"t5_sequence_len": 83}` for SD3). The Runner passes one
`ShapeContext` per iteration to `trigger.run(args, prompt, shape)`.

---

## T

**T5**
A large language model text encoder used in SD3, SD3.5, and FLUX. T5 produces high-quality
text representations with a long context window. In GenAI-SD, the T5 encoder runs on the NPU
(using a GPTQ-quantized version for memory efficiency).

**Trigger**
The middle layer of the GenAI-SD pipeline architecture (`src/trigger/`). A Trigger loads model
weights in `__init__` and exposes a `run(args, prompt, shape)` method that constructs the
pipeline object, defines a `run_fn` closure, and delegates to
`BaseOnnxPipelineTrigger.run(pipe_obj, run_fn, height, width, enable_profile=...)` for
profiling or single-run execution. See [docs/07-developer-guide.md](07-developer-guide.md).

---

## U

**UNet (U-Net)**
The convolutional neural network architecture used in SD1.5, SD2.1, and SDXL for the denoising
step. It has an encoder-decoder structure with skip connections. The UNet is the primary
compute component in SD1.5/SDXL and is fully offloaded to the NPU in GenAI-SD.

---

## V

**VAE (Variational Autoencoder)**
The encoder/decoder pair that maps between pixel space and latent space. The VAE encoder
compresses an image into a latent representation; the VAE decoder reconstructs pixels from
a latent. In GenAI-SD, the VAE decoder is the final step in the pipeline and is offloaded
to the NPU.

---

## X

**XRT (Xilinx Runtime)**
The hardware abstraction layer for AMD/Xilinx NPU hardware. Provides `xrt-smi` for device
management. To configure the NPU for peak performance:
```bash
xrt-smi configure --pmode performance
```
See https://ryzenai.docs.amd.com/en/latest/xrt_smi.html for full reference.
