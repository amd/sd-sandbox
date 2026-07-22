# Dynamic Dispatch (DD) Model Guide (Windows)

This guide explains how to create NPU-optimized DD (Dynamic Dispatch) models from standard ONNX
models using `onnx_utils` on **Windows 11**.

> **Platform:** Windows only. `onnx_utils preprocess` and `onnx_utils partition` are not supported
> on Linux. For Linux inference with pre-partitioned models, see [docs/05-linux-guide.md](05-linux-guide.md).
>
> **Prerequisite:** Complete the environment setup in [docs/01-quick-start.md](01-quick-start.md)
> before following any steps here. The `onnx_utils` tool is included in the Ryzen AI Software
> installation.

---

## Table of Contents

- [What is Dynamic Dispatch?](#what-is-dynamic-dispatch)
- [Overview of the DD Workflow](#overview-of-the-dd-workflow)
- [Prerequisites](#prerequisites)
- [Step 1: Download Base Models](#step-1-download-base-models)
- [Step 2: Export PyTorch Model to ONNX](#step-2-export-pytorch-model-to-onnx)
- [Step 3: Preprocess — model.onnx → optimized.onnx](#step-3-preprocess--modelonnx--optimizedonnx)
- [Step 4: Partition Strategy YAML Files](#step-4-partition-strategy-yaml-files)
- [Step 5: Partition — optimized.onnx → replaced.onnx](#step-5-partition--optimizedonnx--replacedonnx)
- [**Alternative: One-Shot with `onnx_utils optimize sd`**](#alternative-one-shot-with-onnx_utils-optimize-sd)
- [Step 6: Dynamic Shape Configuration](#step-6-dynamic-shape-configuration)
- [Step 7: Verify the Output](#step-7-verify-the-output)
- [Model-Specific Workflows](#model-specific-workflows)
  - [SD 1.5 / SD Turbo / SD 2.1](#sd-15--sd-turbo--sd-21)
  - [SD 1.5 ControlNet](#sd-15-controlnet)
  - [SDXL / SDXL-turbo / Segmind-Vega](#sdxl--sdxl-turbo--segmind-vega)
  - [SD3 / SD3.5 (Fixed Shape)](#sd3--sd35-fixed-shape)
  - [SD3 / SD3.5 ControlNet (Fixed Shape)](#sd3--sd35-controlnet-fixed-shape)
  - [SD3 / SD3.5 OutPainting / Removal / InPainting](#sd3--sd35-outpainting--removal--inpainting)
  - [SD3 / SD3.5 Dynamic Shape](#sd3--sd35-dynamic-shape)
  - [FLUX.1-schnell](#flux1-schnell)
- [Troubleshooting](#troubleshooting)

---

## What is Dynamic Dispatch?

**Dynamic Dispatch (DD)** is AMD's technology for running ONNX model subgraphs on the NPU
(Neural Processing Unit) inside AMD Ryzen AI chips. Instead of running the entire model on the CPU
or GPU, DD partitions the ONNX graph into:

- **NPU subgraphs** — fused kernels compiled for the NPU, stored as a `DynamicDispatch` node
- **CPU/iGPU fallback** — operations that are numerically sensitive or unsupported on NPU

The result is a `replaced.onnx` file where large compute-intensive subgraphs are replaced with
a single `DynamicDispatch` custom ONNX operator. At runtime, ONNX Runtime loads this custom
operator (`onnxruntime_providers_ryzenai.dll` on Windows; `libonnxruntime_providers_ryzenai.so` on
Linux) and dispatches the
fused subgraph to the NPU.

### Key files produced by the DD workflow

```
model_component/
├─ dd/
│  ├─ replaced.onnx       ← the partitioned model (ONNX Runtime loads this)
│  ├─ cache/              ← compiled DD kernel metadata and binaries
│  │  └─ *.json           ← subgraph metadata
│  └─ config.json         ← model config (copied from original model directory)
└─ optimized.onnx         ← intermediate (after preprocess, input to partition)
```

### NPU offload ratios (reference)

| Component | Offload ratio |
|-----------|--------------|
| SD1.5 UNet | ~94.5% |
| SD1.5 VAE decoder | ~96.7% |
| SD1.5 ControlNet | ~81.5% |
| SD3/SD3.5 MM-DiT | ~90%+ |
| FLUX transformer | ~90%+ |

---

## Overview of the DD Workflow

```
Hugging Face model (safetensors)
         │
         ▼  Step 2: optimum-cli export onnx  (or sd15_export_local.py)
    model.onnx
         │
         ├─── Manual path (Steps 3–5) ──────────────────────────────┐
         │    ▼  Step 3: onnx_utils preprocess                       │
         │  optimized.onnx                                           │
         │    ▼  Step 4: select partition_strategy.yaml              │
         │    ▼  Step 5: onnx_utils partition                        │
         │                                                           │
         └─── One-shot path (Alternative) ──────────────────────────┘
              ▼  onnx_utils optimize sd  (combines Steps 3–5)
         │
         ▼
    dd/replaced.onnx  +  dd/cache/
         │
         ▼  ONNX Runtime + Ryzen AI EP
      NPU inference
```

---

## Prerequisites

1. Completed [docs/01-quick-start.md](01-quick-start.md) (conda environment active, Ryzen AI Software installed).
2. Verify `onnx_utils` is available:

```bash
onnx_utils --version
```

3. For dynamic shape models, set the DD plugin environment variable:

```bash
set DD_PLUGINS_ROOT=.\lib\transaction\stx
```

---

## Step 1: Download Base Models

### Model Source Reference

| Family | Model | Hugging Face Repo |
|--------|-------|-----------------|
| SD 1.5 | SD 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| SD 1.5 | SD Turbo | `stabilityai/sd-turbo` |
| SD 1.5 | SD 1.5 ControlNet (Canny) | `lllyasviel/control_v11p_sd15_canny` |
| SDXL | SDXL-base | `stabilityai/stable-diffusion-xl-base-1.0` |
| SDXL | SDXL-turbo | `stabilityai/sdxl-turbo` |
| SDXL | Playground 2.5 | `playgroundai/playground-v2.5-1024px-aesthetic` |
| SDXL | SSD-1B | `segmind/SSD-1B` |
| SDXL | DreamShaper XL | `Lykon/dreamshaper-xl-lightning` |
| SDXL | Segmind-Vega | `segmind/Segmind-Vega` |
| SD3 | SD3 Medium | `stabilityai/stable-diffusion-3-medium-diffusers` |
| SD3 | SD3.5 Medium 🔒 | `stabilityai/stable-diffusion-3.5-medium` |
| SD3 | SD3 ControlNet Canny | `InstantX/SD3-Controlnet-Canny` |
| SD3 | SD3 ControlNet Tile | `InstantX/SD3-Controlnet-Tile` |
| SD3 | SD3 ControlNet Pose | `InstantX/SD3-Controlnet-Pose` |
| FLUX | FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` |
| FLUX | FLUX.2 Klein 🔒 | `black-forest-labs/FLUX.2-klein-4B` |

> **🔒 Gated:** Run `huggingface-cli login` before downloading.

---

```bash
# Install the Hugging Face Hub library if not already present
pip install huggingface_hub

# Log in once for gated models (SD3.5 🔒, FLUX.2 Klein 🔒)
huggingface-cli login

# General usage — replace <repo> and <local-dir> with values from the table above
huggingface-cli download <repo> --local-dir <local-dir>

# Example: SD3 Medium
huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers --local-dir models/sd3

# Example: FLUX.1-schnell
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir models/flux1-schnell
```

---

## Step 2: Export PyTorch Model to ONNX

> **Skip this step** if you already have `model.onnx` files for each component. This step is required for models that
> only provide safetensors.

The `onnx_utils preprocess` step in [Step 3](#step-3-preprocess--modelonnx--optimizedonnx) takes a
`model.onnx` as input. If the downloaded Hugging Face repo only contains `.safetensors` weights,
you must first export each component to ONNX format.

### Method A: Using `optimum-cli` (Recommended for most models)

[Hugging Face Optimum](https://huggingface.co/docs/optimum/index) provides an `optimum-cli`
command that exports diffusers pipelines to ONNX in one step. This is the recommended path for
SD 1.5, SD 2.1, SDXL, and similar models.

**Setup a dedicated conversion environment** (keep it separate from the inference env to avoid
PyTorch/ONNX Runtime version conflicts):

```bash
conda create -n convert_onnx python=3.10
conda activate convert_onnx
pip install torch==2.6.0
pip install diffusers==0.32.2
pip install optimum[exporters]
pip install onnx==1.17.0
```

**Export SD 1.5:**

```bash
optimum-cli export onnx \
    --model stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --task stable-diffusion \
    models/sd15/

# Copy vae config to vae_decoder and vae_encoder subdirs
cp models/sd15/vae/config.json models/sd15/vae_decoder/
cp models/sd15/vae/config.json models/sd15/vae_encoder/
```

**Export SDXL-turbo:**

```bash
optimum-cli export onnx \
    --model stabilityai/sdxl-turbo \
    --task stable-diffusion-xl \
    models/sdxl-turbo/

cp models/sdxl-turbo/vae/config.json models/sdxl-turbo/vae_decoder/
cp models/sdxl-turbo/vae/config.json models/sdxl-turbo/vae_encoder/
```

After export, switch back to the inference environment:

```bash
conda activate ryzenai-stable-diffusion   # or ryzen-ai-1.8.0
```

### Method B: Using `src/utils/sd15_export_local.py` (SD 1.5 / SD 1.5 ControlNet)

For SD 1.5, the repository provides a local export script that exports UNet, ControlNet, and
VAE Decoder directly from safetensors weights using `torch.onnx.export`:

```bash
conda activate convert_onnx   # separate env with torch + diffusers
cd models/sd15

# Edit the paths at the bottom of the script to match your model directory, then run:
python path/to/GenAI-SD/src/utils/sd15_export_local.py
```

The script exports the following components to `model.onnx`:
- `unet/model.onnx` — from `unet/diffusion_pytorch_model.fp16.safetensors`
- `controlnet/model.onnx` — from `controlnet/diffusion_pytorch_model.fp16.safetensors`
- `vae_decoder/model.onnx` — from `vae/diffusion_pytorch_model.fp16.safetensors`

Required packages for this method:
```bash
pip install torch==2.6.0 diffusers==0.32.2 onnx==1.17.0 safetensors onnxconverter-common
```
---

## Step 3: Preprocess — model.onnx → optimized.onnx

> **One-shot alternative:** Steps 3–5 (preprocess + strategy selection + partition) can be
> combined into a single command using `onnx_utils optimize sd`. See
> [Alternative: One-Shot with `onnx_utils optimize sd`](#alternative-one-shot-with-onnx_utils-optimize-sd)
> for details. The manual steps below are useful when you need fine-grained control over
> individual parameters.

The `onnx_utils preprocess` step applies AMD-specific graph optimizations to produce an
`optimized.onnx` that is suitable for NPU partitioning. The `--optimize` flag selects the
optimization profile matching the model architecture.

**Available optimization profiles:**

| `--optimize` value | Model architecture |
|-------------------|-------------------|
| `sd15_unet` | SD 1.5 / SD 2.1 UNet |
| `sd15_controlnet` | SD 1.5 ControlNet |
| `vae` | VAE decoder (all models) |
| `sdxl_unet` | SDXL UNet |
| `mmdit` | SD3 / SD3.5 MM-DiT transformer |
| `flux1` | FLUX.1-schnell DiT transformer |
| `flux2` | FLUX.2-klein DiT transformer |

**Common flags:**
- `--save-as-external` — required for large models (>2 GB); stores weights in a sidecar file
- For dynamic shape models: use `x` as a placeholder for dynamic dimensions

---

## Step 4: Partition Strategy YAML Files

Partition strategy files tell `onnx_utils partition` which AMD passes to run and how to map
subgraphs onto the NPU. They are **bundled inside the `ryzenai_onnx_utils` package** — you do
not need to download or copy them manually.

**Location:**
```
<conda_env>/Lib/site-packages/ryzenai_onnx_utils/data/partition_strategies/
```

Find the path in your active environment:
```bash
python -c "import ryzenai_onnx_utils, os; print(os.path.join(os.path.dirname(ryzenai_onnx_utils.__file__), 'data', 'partition_strategies'))"
```

**Usage:** Pass only the filename to `onnx_utils partition` — the tool resolves it automatically:
```bash
onnx_utils partition "optimized.onnx" "dd/" sd3_mmdit_bfp.yaml -v --force
# NOT: config/partition_strategies/sd3_mmdit_bfp.yaml
```

| Filename | Used for |
|---------|---------|
| `sd15_unet_bfp.yaml` | SD 1.5 / SD-Turbo / SD 2.1 UNet (BFP16) |
| `sd15_unet_bfp_preempt.yaml` | SD 1.5 UNet with preemptive scheduling |
| `sd15_vae_decoder.yaml` | SD 1.5 / SDXL VAE decoder |
| `sdxl_turbo_bfp_unet.yaml` | SDXL / SDXL-turbo / Segmind-Vega UNet (BFP16) |
| `sdxl_turbo_bfp_vae_decoder.yaml` | SDXL VAE decoder (BFP16) |
| `sd3_mmdit_bfp.yaml` | SD3 / SD3.5 MM-DiT transformer (BFP16) |
| `sd3_vae.yaml` | SD3 / SD3.5 / FLUX VAE decoder |
| `flux_transformer_bfp.yaml` | FLUX.1-schnell transformer (BFP16) |
| `flux2_transformer_bfp.yaml` | FLUX.2-klein transformer (BFP16) |

List all available strategy files in your environment:

```bash
python -c "import ryzenai_onnx_utils, os, glob; d=os.path.join(os.path.dirname(ryzenai_onnx_utils.__file__), 'data', 'partition_strategies'); print('\n'.join(sorted(os.path.basename(f) for f in glob.glob(d+'/*.yaml'))))"
```

**File format reference:**

Each strategy YAML uses AMD's pass-based architecture with pass inheritance:

```yaml
# Example: sd3_mmdit_bfp.yaml
---
attributes:
    domains: com.ryzenai
    xclbins: "stx_SD_unified_fastpm_2x4x4"
    op_namespaces: "sd3"
    model_type: SD30_MMDIT
    op_config: "ops-config.json"

inherit_passes_before: !include [sd3_mmdit_no_dd_bfp.yaml]
passes:
    - dd.optimize_toposort
    - dd.dynamic_sd_nodes_to_dd
inherit_passes_after: !include [sd15_dd.yaml]
```

- `attributes` — hardware target (`xclbins`), model type identifier, and namespace for op naming
- `inherit_passes_before` / `inherit_passes_after` — compose pass pipelines from reusable base files
- `passes` — AMD-internal graph transformation passes that replace eligible subgraphs with `DynamicDispatch` nodes

---

## Step 5: Partition — optimized.onnx → replaced.onnx

The `onnx_utils partition` step splits the optimized ONNX graph into NPU subgraphs and CPU
fallback ops. It produces `replaced.onnx` and the `cache/` directory.

**Common partition command structure:**

```bash
# Pass only the YAML filename — onnx_utils resolves it from the package data directory automatically
onnx_utils partition "path/to/optimized.onnx" "path/to/output/dd/" \
    "strategy_filename.yaml" \
    -v --force \
    [--attributes ...]
```

**Common `--attributes`:**

| Attribute | Description |
|-----------|-------------|
| `dynamic_shape_graph=true` | Enable dynamic-shape partitioning (dynamic shape models only) |
| `dynamic_shape_list=[...]` | List of shape dictionaries for dynamic models |
| `whitebox_checker_level=2` | Strictest correctness checking (recommended) |
| `fusion_max_init_bytes=4294967296` | 4 GiB weight buffer cap (required for large models like FLUX/SD3.5) |
| `--save-as-external` | Store output weights externally (required for large models) |

---

## Alternative: One-Shot with `onnx_utils optimize sd`

`onnx_utils optimize sd` is a convenience command that **combines Steps 3, 4, and 5** into a
single invocation: it runs preprocess, auto-selects the partition strategy YAML, and runs
partition in sequence. Use this when the defaults are suitable and you want a simpler workflow.

### Syntax

```bash
onnx_utils optimize \
    --input-model <path/to/model.onnx> \
    --output-model <path/to/output_dir/> \
    [--force] \
    sd \
    [--model-type <type>] \
    [--partition-strategy <yaml>] \
    [--fixed-shapes KEY=VALUE ...] \
    [--resolutions <preset_or_list>] \
    [--turbo] \
    [--attributes KEY=VALUE ...]
```

**Top-level options (before `sd`):**

| Option | Description |
|--------|-------------|
| `--input-model` | Path to input `model.onnx` |
| `--output-model` | Output directory (receives `optimized.onnx` and `dd/`) |
| `--force` | Regenerate all outputs even if they already exist |
| `--no-clean` | Keep intermediate temp files after execution |
| `--dry-run` | Print the commands that would be run without executing |
| `--save-intermediate-models` | Save model after each pass (debugging) |

**SD-specific options (after `sd`):**

| Option | Default | Description |
|--------|---------|-------------|
| `--model-type` | `infer_from_path` | Model architecture: `sd-unet`, `sd-dit`, `sd15-vae-decoder`, `sd15-vae-encoder`, `sd3-vae-decoder`, `sd3-vae-encoder`, `flux`, `flux2` |
| `--partition-strategy` | auto | Override partition strategy YAML filename |
| `--fixed-shapes` | (interactive) | Shape overrides to skip interactive prompts, e.g. `graphs=1 batch=2 height=64 width=64` |
| `--resolutions` | — | Dynamic shape preset: `5x`, `8x`, `20`, or custom `[512x512,1024x1024]` |
| `--turbo` | False | Use `batch=1` (instead of `batch=2`) for UNet/DiT/Flux partition |
| `--attributes` | — | Forward extra attributes to the internal partition step |

### Examples

**SD 1.5 UNet (fixed shape, 512×512):**

```bash
onnx_utils optimize \
    --input-model models/sd15/unet/model.onnx \
    --output-model models/sd15/unet/ \
    --force \
    sd \
    --model-type sd-unet \
    --fixed-shapes graphs=1 batch=2 channels=4 height=64 width=64 sequence=77
```

**SD 1.5 VAE decoder:**

```bash
onnx_utils optimize \
    --input-model models/sd15/vae_decoder/model.onnx \
    --output-model models/sd15/vae_decoder/ \
    --force \
    sd \
    --model-type sd15-vae-decoder \
    --fixed-shapes graphs=1 batch=1 channels=4 height=64 width=64
```

**SDXL UNet (fixed shape, 1024×1024):**

```bash
onnx_utils optimize \
    --input-model models/sdxl-base/unet/model.onnx \
    --output-model models/sdxl-base/unet/ \
    --force --save-as-external \
    sd \
    --model-type sd-unet \
    --partition-strategy sdxl_unet_bfp.yaml \
    --fixed-shapes graphs=1 batch=2 channels=4 height=128 width=128 sequence=77 \
    --attributes fusion_max_init_bytes=4294967296
```

**SD3.5 MM-DiT transformer (fixed shape, 512×512):**

```bash
onnx_utils optimize \
    --input-model models/sd3.5/normal/transformer/model.onnx \
    --output-model models/sd3.5/normal/transformer/ \
    --force --save-as-external \
    sd \
    --model-type sd-dit \
    --partition-strategy sd3_mmdit_bfp.yaml \
    --fixed-shapes graphs=1 batch_1=2 w=64 h=64 batch_2=2 batch_3=2 \
    --attributes fusion_max_init_bytes=4294967296
```

**FLUX.1-schnell transformer (dynamic shape, 8 resolutions):**

```bash
onnx_utils optimize \
    --input-model models/flux1-schnell/transformer/dynamic/model.onnx \
    --output-model models/flux1-schnell/transformer/dynamic/ \
    --force --save-as-external \
    sd \
    --model-type flux \
    --resolutions 8x \
    --attributes whitebox_checker_level=2 fusion_max_init_bytes=4294967296
```

**Preview without running (`--dry-run`):**

```bash
onnx_utils optimize \
    --input-model models/sd15/unet/model.onnx \
    --output-model models/sd15/unet/ \
    --dry-run \
    sd --model-type sd-unet
```

### When to use manual steps instead

| Situation | Use |
|-----------|-----|
| Standard model, no unusual shapes | `onnx_utils optimize sd` (this section) |
| Need to inspect `optimized.onnx` before partitioning | Manual Steps 3 + 5 |
| Custom partition strategy not supported by `optimize` | Manual Step 4 + 5 |
| Debugging a partition failure step-by-step | Manual Steps 3–5 |
| Dynamic shape list not covered by `5x`/`8x`/`20` presets | Manual Step 5 with custom `--attributes dynamic_shape_list` |

---

## Step 6: Dynamic Shape Configuration

For models that support multiple resolutions (SD3, SD3.5, FLUX), the dynamic shape configuration
JSON specifies which shape combinations are pre-compiled. The config files are in `config/`:

- `config/dynamic_shape_20x.json` — 20 resolutions for SD3/SD3.5/FLUX (default)
- `config/dynamic_shape_8x.json` — 8 resolutions (legacy)
- `config/dynamic_shape_5x.json` — 5 resolutions (minimal, for quick testing)

Each JSON entry maps model symbolic dimensions to concrete values:

```json
[
  {"height": 512, "width": 512, "latent_height": 64, "latent_width": 64},
  {"height": 768, "width": 512, "latent_height": 96, "latent_width": 64},
  {"height": 1024, "width": 1024, "latent_height": 128, "latent_width": 128}
]
```

---

## Step 7: Verify the Output

After partition completes, verify the expected files exist:

```bash
# Expected directory structure after successful partition
ls models/sd3/normal/transformer/dd/
# replaced.onnx
# replaced.onnx.data   (if --save-as-external was used)
# cache/
# config.json          (copy from models/sd3/normal/transformer/config.json)
```

Copy the `config.json` from the original model directory into the `dd/` directory:

```bash
cp models/sd3/normal/transformer/config.json models/sd3/normal/transformer/dd/config.json
```

Run a quick verification with the NPU offload report:

```bash
onnx_utils report dd_offload models/sd3/normal/transformer/optimized.onnx --summarize
```

The report shows which ops are offloaded to the NPU and the offload ratio. A ratio above 80% is
typical for a well-partitioned model.

**First inference run:** Locally built DD models require `--enable_compile` (`-c`) on the first
pipeline run to compile fusion kernels for your NPU. Pre-partitioned `amd/` models from Hugging
Face do not need this flag — see [docs/03-model-reference-guide.md](03-model-reference-guide.md).

```bash
python test/run.py --model_id amd/sdxl-turbo-amdnpu --model_path path/to/your/local/model --enable_compile
```

---

## Model-Specific Workflows

### SD 1.5 / SD Turbo / SD 2.1

**Components to partition:** `unet`, `vae_decoder`

```bash
# 1. Preprocess UNet
cd models/sd15/unet
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize sd15_unet --save-as-external

# 2. Preprocess VAE decoder
cd models/sd15/vae_decoder
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

# 3. Partition UNet
onnx_utils partition "models/sd15/unet/optimized.onnx" "models/sd15/unet/dd/" \
    "sd15_unet_bfp.yaml" -v --force

# 4. Partition VAE decoder
onnx_utils partition "models/sd15/vae_decoder/optimized.onnx" "models/sd15/vae_decoder/dd/" \
    "sd3_vae.yaml" -v --force

# 5. Copy config.json files
cp models/sd15/unet/config.json models/sd15/unet/dd/
cp models/sd15/vae_decoder/config.json models/sd15/vae_decoder/dd/
```

**Preprocess parameter reference:**

| Parameter | UNet (512×512) | VAE decoder (512×512) |
|-----------|---------------|----------------------|
| graphs | 1 | 1 |
| batch | 2 | 1 |

---

### SD 1.5 ControlNet

**Components to partition:** `unet`, `controlnet`, `vae_decoder`

```bash
# Preprocess
cd models/sd15_controlnet/unet
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize sd15_unet --save-as-external

cd models/sd15_controlnet/controlnet
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize sd15_controlnet --save-as-external

cd models/sd15_controlnet/vae_decoder
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

# Partition
onnx_utils partition "models/sd15_controlnet/unet/optimized.onnx" \
    "models/sd15_controlnet/unet/dd/" \
    "sd15_unet_bfp.yaml" -v --force

onnx_utils partition "models/sd15_controlnet/controlnet/optimized.onnx" \
    "models/sd15_controlnet/controlnet/dd/" \
    "sd15_unet_bfp.yaml" -v --force

onnx_utils partition "models/sd15_controlnet/vae_decoder/optimized.onnx" \
    "models/sd15_controlnet/vae_decoder/dd/" \
    "sd3_vae.yaml" -v --force
```

---

### SDXL / SDXL-turbo / Segmind-Vega

**Components to partition:** `unet`, `vae_decoder`, `vae_encoder`

```bash
# Preprocess
cd models/sdxl-turbo/unet
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize sdxl_unet --save-as-external

cd models/sdxl-turbo/vae_decoder
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

cd models/sdxl-turbo/vae_encoder
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

# Partition
onnx_utils partition "models/sdxl-turbo/unet/optimized.onnx" \
    "models/sdxl-turbo/unet/dd/" \
    "sdxl_unet_bfp.yaml" \
    -v --force --attributes fusion_max_init_bytes=4294967296

onnx_utils partition "models/sdxl-turbo/vae_decoder/optimized.onnx" \
    "models/sdxl-turbo/vae_decoder/dd/" \
    "sd3_vae.yaml" -v --force

onnx_utils partition "models/sdxl-turbo/vae_encoder/optimized.onnx" \
    "models/sdxl-turbo/vae_encoder/dd/" \
    "sd3_vae.yaml" -v --force
```

**Preprocess parameter reference:**

| Parameter | UNet 512×512 | UNet 1024×1024 | VAE |
|-----------|-------------|----------------|-----|
| graphs | 1 | 1 | 1 |
| batch | 2 | 2 | 1 |

---

### SD3 / SD3.5 (Fixed Shape)

**Components to partition:** `transformer`, `vae_decoder`, `vae_encoder`  
**Model source:** `https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers`

```bash
# Preprocess MM-DiT transformer (512×512)
cd models/sd3/normal/transformer
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize mmdit --save-as-external

# Preprocess VAE decoder
cd models/sd3/common/vae_decoder
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

# Preprocess VAE encoder
cd models/sd3/common/vae_encoder
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae

# Partition MM-DiT transformer
onnx_utils partition "models/sd3/normal/transformer/optimized.onnx" \
    "models/sd3/normal/transformer/dd/" \
    "sd3_mmdit_bfp.yaml" \
    -v --force --attributes fusion_max_init_bytes=4294967296 --save-as-external

# Partition VAE decoder
onnx_utils partition "models/sd3/common/vae_decoder/optimized.onnx" \
    "models/sd3/common/vae_decoder/dd/" \
    "sd3_vae.yaml" -v --force

# Copy config.json files
cp models/sd3/normal/transformer/config.json models/sd3/normal/transformer/dd/
cp models/sd3/common/vae_decoder/config.json models/sd3/common/vae_decoder/dd/
```

**MM-DiT preprocess parameter reference:**

| Parameter | 512×512 | 1024×1024 |
|-----------|---------|-----------|
| graphs | 1 | 1 |
| batch_1 | 2 | 2 |
| w | 64 | 128 |
| h | 64 | 128 |
| batch_2 | 2 | 2 |
| batch_3 | 2 | 2 |

---

### SD3 / SD3.5 ControlNet (Fixed Shape)

**Additional components:** `controlnet-canny`, `controlnet-tile`, `controlnet-pose`

```bash
# Preprocess ControlNet
cd models/sd3/normal/controlnet-canny
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize mmdit --save-as-external

# Partition ControlNet
onnx_utils partition "models/sd3/normal/controlnet-canny/optimized.onnx" \
    "models/sd3/normal/controlnet-canny/dd/" \
    "sd3_mmdit_bfp.yaml" \
    -v --force --attributes fusion_max_init_bytes=4294967296 --save-as-external

# Repeat for controlnet-tile and controlnet-pose
```

---

### SD3 / SD3.5 OutPainting / Removal / InPainting

Uses the same transformer preprocess/partition as SD3 ControlNet, with different model directories.

```bash
# Outpainting/Removal transformer
cd models/sd3/normal/transformer
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize mmdit --save-as-external

# InPainting transformer (separate model)
cd models/sd3/inpainting/transformer
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize mmdit --save-as-external

# Parameters for outpainting (1024×1024 with seq=160 for outpainting context)
# batch_1=2, w=128, h=128, batch_2=2, batch_3=2, seq=160, state_dim1=4096
```

---

### SD3 / SD3.5 Dynamic Shape

Dynamic shape models use `x` as a placeholder for symbolic dimensions and require the DD plugin.

```bash
# Set DD plugin (required for dynamic shape)
set DD_PLUGINS_ROOT=.\lib\transaction\stx

# Preprocess MM-DiT with dynamic shapes (x = dynamic placeholder)
cd models/sd3/normal/transformer/dynamic
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize mmdit --save-as-external
# Parameters: B=1, w=x, h=x, batch_2=2, batch_3=2

# Preprocess VAE decoder with dynamic shapes
cd models/sd3/common/vae_decoder/dynamic
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae
# Parameters: batch=1, channels=16, latent_height=x, latent_width=x

# Partition MM-DiT with dynamic shape list (SD3, 20 resolutions — from config/dynamic_shape_20x.json)
onnx_utils partition "optimized.onnx" "dd" \
    "sd3_mmdit_bfp.yaml" \
    -v --force \
    --attributes dynamic_shape_graph=true \
    --attributes "dynamic_shape_list=[{'B': 1, 'w': 64, 'h': 64, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 80, 'h': 80, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 96, 'h': 96, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 128, 'h': 128, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 64, 'h': 48, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 80, 'h': 60, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 96, 'h': 72, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 128, 'h': 96, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 48, 'h': 64, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 60, 'h': 80, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 72, 'h': 96, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 96, 'h': 128, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 64, 'h': 36, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 80, 'h': 48, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 96, 'h': 56, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 128, 'h': 72, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 36, 'h': 64, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 48, 'h': 80, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 56, 'h': 96, 'batch_2': 2, 'batch_3': 2}, {'B': 1, 'w': 72, 'h': 128, 'batch_2': 2, 'batch_3': 2}]" \
    --attributes whitebox_checker_level=2 \
    --attributes fusion_max_init_bytes=4294967296 \
    --save-as-external
```

**Supported dynamic resolutions for SD3 (20 resolutions):**

| Resolution | latent w×h |
|-----------|-----------|
| 512×512 | 64×64 |
| 640×640 | 80×80 |
| 768×768 | 96×96 |
| 1024×1024 | 128×128 |
| 512×384 | 64×48 |
| 640×480 | 80×60 |
| 768×576 | 96×72 |
| 1024×768 | 128×96 |
| 384×512 | 48×64 |
| 480×640 | 60×80 |
| 576×768 | 72×96 |
| 768×1024 | 96×128 |
| 512×288 | 64×36 |
| 640×384 | 80×48 |
| 768×448 | 96×56 |
| 1024×576 | 128×72 |
| 288×512 | 36×64 |
| 384×640 | 48×80 |
| 448×768 | 56×96 |
| 576×1024 | 72×128 |

---

### FLUX.1-schnell

**Components to partition:** `transformer` (DiT), `vae_decoder`  
**Model source:** `https://huggingface.co/black-forest-labs/FLUX.1-schnell`

FLUX uses a different latent structure than SD models. The latent is patchified into a 1-D token
sequence:

```
latent_dim = (H / 16) * (W / 16)
```

For example, 1024×1024 → `latent_dim = 64 * 64 = 4096`.

```bash
# Set DD plugin (required for FLUX dynamic shape)
set DD_PLUGINS_ROOT=.\lib\transaction\stx

# Preprocess FLUX transformer
cd models/flux1-schnell/transformer/dynamic
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize flux1 --save-as-external
# Parameters: B=1, latent_dim=x, txt_seq_len=x

# Preprocess FLUX VAE decoder
cd models/flux1-schnell/vae_decoder/dynamic
onnx_utils preprocess "model.onnx" "optimized.onnx" --optimize vae
# Parameters: batch=1, batch_size=1, channels=16, latent_height=x, latent_width=x

# Partition FLUX transformer (11 unique latent_dims covering 20 resolutions)
onnx_utils partition "optimized.onnx" "dynamic/dd" \
    "flux_transformer_bfp.yaml" \
    -v --force \
    --attributes dynamic_shape_graph=true \
    --attributes "dynamic_shape_list=[{'B': 1, 'latent_dim': 576, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 768, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 960, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 1024, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 1200, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 1344, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 1600, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 1728, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 2304, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 3072, 'txt_seq_len': 256}, {'B': 1, 'latent_dim': 4096, 'txt_seq_len': 256}]" \
    --attributes whitebox_checker_level=2 \
    --attributes fusion_max_init_bytes=4294967296 \
    --save-as-external

# Partition FLUX VAE decoder (20 shapes — VAE latent = image_size / 8)
onnx_utils partition \
    "models/flux1-schnell/vae_decoder/dynamic/optimized.onnx" \
    "models/flux1-schnell/vae_decoder/dynamic/dd" \
    "sd3_vae.yaml" \
    -v --force \
    --attributes dynamic_shape_graph=true \
    --attributes dynamic_shape_list="[{'batch_size': 1, 'latent_height': 64, 'latent_width': 64}, {'batch_size': 1, 'latent_height': 80, 'latent_width': 80}, {'batch_size': 1, 'latent_height': 96, 'latent_width': 96}, {'batch_size': 1, 'latent_height': 128, 'latent_width': 128}, {'batch_size': 1, 'latent_height': 64, 'latent_width': 48}, {'batch_size': 1, 'latent_height': 80, 'latent_width': 60}, {'batch_size': 1, 'latent_height': 96, 'latent_width': 72}, {'batch_size': 1, 'latent_height': 128, 'latent_width': 96}, {'batch_size': 1, 'latent_height': 48, 'latent_width': 64}, {'batch_size': 1, 'latent_height': 60, 'latent_width': 80}, {'batch_size': 1, 'latent_height': 72, 'latent_width': 96}, {'batch_size': 1, 'latent_height': 96, 'latent_width': 128}, {'batch_size': 1, 'latent_height': 64, 'latent_width': 36}, {'batch_size': 1, 'latent_height': 80, 'latent_width': 48}, {'batch_size': 1, 'latent_height': 96, 'latent_width': 56}, {'batch_size': 1, 'latent_height': 128, 'latent_width': 72}, {'batch_size': 1, 'latent_height': 36, 'latent_width': 64}, {'batch_size': 1, 'latent_height': 48, 'latent_width': 80}, {'batch_size': 1, 'latent_height': 56, 'latent_width': 96}, {'batch_size': 1, 'latent_height': 72, 'latent_width': 128}]"
```

**FLUX supported dynamic resolutions (20 resolutions, 11 unique latent_dims):**

| Resolution | latent_dim (transformer) | latent h×w (VAE) |
|-----------|--------------------------|-----------------|
| 288×512 / 512×288 | 576 | 36×64 / 64×36 |
| 384×512 / 512×384 | 768 | 48×64 / 64×48 |
| 384×640 / 640×384 | 960 | 48×80 / 80×48 |
| 512×512 | 1024 | 64×64 |
| 480×640 / 640×480 | 1200 | 60×80 / 80×60 |
| 448×768 / 768×448 | 1344 | 56×96 / 96×56 |
| 640×640 | 1600 | 80×80 |
| 576×768 / 768×576 | 1728 | 72×96 / 96×72 |
| 768×768 / 576×1024 / 1024×576 | 2304 | 96×96 / 72×128 / 128×72 |
| 768×1024 / 1024×768 | 3072 | 96×128 / 128×96 |
| 1024×1024 | 4096 | 128×128 |

After partitioning, copy the `config.json` from the transformer model directory into the `dd/` output:

```bash
cp models/flux1-schnell/transformer/config.json models/flux1-schnell/transformer/dynamic/dd/
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `onnx_utils: command not found` | Ryzen AI Software not installed or env not active | Run `conda activate ryzen-ai-1.8.0`; verify `pip show ryzenai-onnx-utils` |
| Partition fails with OOM | Fusion buffer too small | Add `--attributes fusion_max_init_bytes=4294967296` |
| `replaced.onnx` not generated | Whitebox checker failure | Try `--attributes whitebox_checker_level=1` to get more details |
| `cache/` directory is empty | DD plugin not set | Set `DD_PLUGINS_ROOT=.\lib\transaction\stx` before running |
| Low NPU offload ratio (<50%) | Wrong `--optimize` profile | Verify you used the correct profile for your model architecture |
| Black output images at inference | Missing `config.json` in `dd/` directory | Copy `config.json` from the model root into the `dd/` output directory |
| `LFS pointer file` errors | git-lfs not properly set up | Run `git lfs install && git lfs pull` in the model directory |
| Dynamic shape partition is slow | Large `dynamic_shape_list` | Start with 3–4 shapes during development; add all shapes for production |

For inference-time troubleshooting, see [docs/06-faq.md](06-faq.md).
For adding a new DD pipeline to the codebase, see [docs/07-developer-guide.md](07-developer-guide.md).
