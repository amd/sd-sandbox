# User Guide — Batch Pipelines and Run Orchestrator

This guide explains how to use `scripts/run_pipelines.py` to run one or more Stable Diffusion
pipelines in batch mode, configure YAML pipeline definitions, and interpret output files.

> Before running any pipelines, complete the setup in [docs/01-quick-start.md](01-quick-start.md).

---

## Table of Contents

- [What is the Pipeline Orchestrator?](#what-is-the-pipeline-orchestrator)
- [Supported Models](#supported-models)
- [Image Generation Methods](#image-generation-methods)
- [Hardware Requirements](#hardware-requirements)
- [Quick Start — Running Your First Pipeline](#quick-start--running-your-first-pipeline)
- [Pipelines and Pipeline Groups Configuration](#pipelines-and-pipeline-groups-configuration)
  - [YAML Files](#yaml-files)
  - [JSON Prompt Files](#json-prompt-files)
- [Running Pipelines](#running-pipelines)
  - [Execution Modes](#execution-modes)
  - [Argument Priority](#argument-priority)
  - [Pipeline Options Reference](#pipeline-options-reference)
- [Output Files](#output-files)
  - [Generated Images](#generated-images)
  - [Results Text Files](#results-text-files)
  - [Log Files](#log-files)
  - [Benchmark Files](#benchmark-files)
- [Community and Support](#community-and-support)

---

## What is the Pipeline Orchestrator?

The pipeline orchestrator (`scripts/run_pipelines.py`) is a no-code interface for running Stable
Diffusion models on AMD Ryzen AI hardware. It solves four common problems:

| Problem | Solution |
|---------|----------|
| **Model Selection Complexity** — dozens of SD models, each with different capabilities | **Unified Pipeline Interface** — run any model with consistent commands |
| **Hardware Optimization Guesswork** — need to compare NPU vs CPU performance | **Built-in Benchmarking** — automated profiling with timing and memory statistics |
| **Tedious Manual Testing** — running multiple prompts across multiple models | **Automated Batch Processing** — test hundreds of prompts across all models automatically |
| **Code Modification Required** — official examples require source code changes for simple customization | **No-Code Configuration** — configure everything via YAML and CLI arguments |

---

## Supported Models

All AMD NPU-optimized model IDs are available on Hugging Face under the `amd/` namespace and are
downloaded automatically on first use. Models are cached under `models/` in the project root by
default. Set the environment variable `RYZENAI_GENAI_SD_MODELS_ROOT` to redirect the cache to a
different directory:

```powershell
# Windows PowerShell
$env:RYZENAI_GENAI_SD_MODELS_ROOT = "D:\my_models"
python test/run.py --model_id amd/stable-diffusion-1.5-amdnpu --prompt "..."

# Or set permanently via System Properties → Environment Variables
```

When `RYZENAI_GENAI_SD_MODELS_ROOT` is set, all subsequent downloads and cache lookups use that
path instead of the built-in `models/` folder.

| Model | Hugging Face Model ID | Resolution | T2I | I2I | ControlNet | OutPainting |
|-------|---------------------|-----------|-----|-----|------------|-------------|
| SD 1.5 | `amd/stable-diffusion-1.5-amdnpu` | 512×512 | ✓ | | | |
| SD Turbo | `amd/sd-turbo-amdnpu` | 512×512 | ✓ | | | |
| SD 2.1 *(deprecated — no HF model ID)* | — | 512/768 | ✓ | | | |
| SD 1.5 ControlNet | `amd/sd15-controlnet-amdnpu` | 512×512 | ✓ | | Canny | |
| SD XL Base | `amd/sdxl-base-amdnpu` | 1024×1024 | ✓ | | | |
| SD XL Turbo | `amd/sdxl-turbo-amdnpu` | 512×512 | ✓ | | | |
| Segmind Vega | `amd/segmind-vega-amdnpu` | 1024×1024 | ✓ | ✓ | | |
| Playground 2.5 | `amd/playground-v2.5-1024px-aesthetic-amdnpu` | 1024×1024 | ✓ | | | |
| SSD-1B | `amd/SSD-1B-amdnpu` | 1024×1024 | ✓ | | | |
| DreamShaper XL | `amd/dreamshaper-xl-lightning-amdnpu` | 1024×1024 | ✓ | | | |
| SD 3.0 Medium | `amd/stable-diffusion-3-medium-amdnpu` | Multiple* | ✓ | | Canny/Tile/Pose/Depth/Union | ✓ |
| SD 3.5 Medium | `amd/stable-diffusion-3.5-medium-amdnpu` | Multiple* | ✓ | | Canny/Tile/Pose | ✓ |
| FLUX.1-schnell | `amd/FLUX.1-schnell-amdnpu` | Multiple* | ✓ | | | |
| FLUX.2 Klein | `amd/FLUX.2-klein-4B-amdnpu` | 1024×1024 | ✓ | ✓ | | |

*Multiple = 512×512, 512×768, 768×512, 576×1024, 1024×576, 768×1024, 1024×768, 1024×1024*

> **Kolors** is not auto-downloaded from the `amd/` namespace. You must build and prepare the DD
> model locally — see [docs/04-model-generate-guide.md](04-model-generate-guide.md).

See [docs/03-model-reference-guide.md](03-model-reference-guide.md) for model-specific commands and parameters.

---

## Image Generation Methods

**Text-to-Image (T2I):** Generate images from scratch using only text descriptions. Provide a
prompt like "a mountain landscape at sunset" and the model creates an image from random noise.
*Use cases: concept art, marketing visuals, creative exploration, rapid prototyping.*

**Image-to-Image (I2I):** Transform an existing image based on a text prompt while preserving
structural elements. Provide both a reference image and a prompt; the `--strength` parameter
controls how much of the original is preserved. *Use cases: style transfer, image variations,
artistic reinterpretation, photo enhancement.*

**ControlNet:** Guide image generation with a reference image (edge map, pose skeleton, depth map)
for precise spatial control over the generated content.

**InPainting:** Fill a masked region of an existing image with new content guided by a text prompt.
Requires a mask image (white = fill, black = preserve).

**OutPainting:** Extend the borders of an existing image outward. Generates new content beyond
the original image edges. *(Note: OutPainting controlnet weights are not included in the current
Hugging Face release. Use InPainting for masked-region editing.)*

**Object Removal:** Remove objects from an image by masking them. The model fills the masked region
with plausible background content. *(Note: Removal controlnet weights are not included in the
current Hugging Face release. Use InPainting for masked-region editing.)*

---

## Hardware Requirements

| Series | Codename | GPU | Year |
|--------|----------|-----|------|
| [Ryzen AI Max PRO 300](https://www.amd.com/en/partner/articles/ryzen-ai-pro-300-series-processors.html) | Strix Halo | Radeon 8000S | 2025 |
| [Ryzen AI PRO 300](https://www.amd.com/en/partner/articles/ryzen-ai-pro-300-series-processors.html) | Strix/Krackan | Radeon 800M | 2025 |
| [Ryzen AI Max 300](https://www.amd.com/en/partner/articles/ryzen-ai-300-series-processors.html) | Strix Halo | Radeon 8000S | 2025 |
| [Ryzen AI 300](https://www.amd.com/en/partner/articles/ryzen-ai-300-series-processors.html) | Strix Point | Radeon 800M | 2025 |

---

## Quick Start — Running Your First Pipeline

**1.** Complete [docs/01-quick-start.md](01-quick-start.md) to install drivers and activate the conda env.

**2.** The default `config/pipeline_configs.yaml` uses relative paths and works out of the box.
   Only edit the `paths:` section if you need custom locations:

```yaml
paths:
  test_path: "./test"
  control_images_path: "./test/assets"
  config_files_path: "./config"
```

**3.** Verify available pipelines:

```powershell
python scripts/run_pipelines.py --list
```

**4.** Run a pipeline:

```powershell
python scripts/run_pipelines.py --pipelines sdxl_turbo
```

Expected output:

```text
[AUTO] Automatic provider selection enabled
   • Will use best available: DML > CPU
   • Available providers: DmlExecutionProvider, CPUExecutionProvider

--- Pipeline 1/1: SDXL Turbo Pipeline ---
   [Output] Live Output:
      2026-01-01 10:00:00 [INFO] Load unet ...
      2026-01-01 10:00:30 [INFO] Total NPU memory usage: 1521.66MB (1.49GB)
[OK] SUCCESS in 36.5s
   [Dir] Images saved to: generated_images/

SUMMARY - QUICK TEST MODE
Total runs: 1 | Successful: 1/1 (100.0%) | Duration: 0:00:45
Results saved to: results/pipeline_results_quick_20260101_100000.txt
```

---

## Pipelines and Pipeline Groups Configuration

Pipelines are defined in `config/pipeline_configs.yaml`. Each pipeline maps to a Python run script
and a set of parameters. Pipelines can be grouped for convenience.

### YAML Files

The `config/pipeline_configs.yaml` file has four main sections:

**1. Project Paths** (`paths:`)

Set directories for test scripts, control images, and config files. Relative paths are resolved
from the project root:

```yaml
paths:
  test_path: "./test"
  control_images_path: "./test/assets"
  config_files_path: "./config"
```

**2. Global Pipeline Defaults** (`defaults:`)

Set parameters that apply to all pipelines unless overridden per-pipeline. Example:

```yaml
defaults:
  width: 512
  height: 512
  script: "run.py"
  prompt_file: "./config/geometry_prompt.json"
```

**3. Pipeline Definitions** (`pipelines:`)

Each pipeline must have a unique name and a `model_ids` list. Per-pipeline parameters are set
via the `args:` dictionary (converted internally to CLI flags). All pipelines use the unified
`test/run.py` entry point (configured globally via `defaults.script: "run.py"`):

```yaml
pipelines:
  sdxl_turbo:
    name: "SDXL Turbo Pipeline"
    width: 512
    height: 512
    model_ids:
      - "amd/sdxl-turbo-amdnpu"
    args:
      num_inference_steps: 4
      guidance_scale: 0.0

  sd35_base:
    name: "SD3.5 Pipeline"
    model_ids:
      - "amd/stable-diffusion-3.5-medium-amdnpu"
    args:
      num_inference_steps: 30
      width: 1024
      height: 1024
      controlnet: "None"
      guidance_scale: 4.5

  flux2_klein:
    name: "FLUX.2 Klein Pipeline"
    width: 1024
    height: 1024
    model_ids:
      - "amd/FLUX.2-klein-4B-amdnpu"
    args:
      num_inference_steps: 4
      guidance_scale: 1.0
      seed: 0
```

> **Note:** The flat `extra_args` list format is also supported. The `script` field is no longer
> needed per-pipeline — `defaults.script: "run.py"` routes all pipelines through the unified
> `test/run.py` entry point, which automatically selects the correct Runner based on `--model_id`.

**4. Pipeline Groups** (`pipeline_groups:`)

Groups are named lists of pipeline IDs for running multiple pipelines at once:

```yaml
pipeline_groups:
  all:
    - sd_15
    - sd_turbo
    - sdxl_turbo
    - sdxl_base
    - sd35_base
    - flux1_schnell
    - flux2_klein
    # ... see config/pipeline_configs.yaml for the full list

  my_custom_group:
    - sdxl_base
    - flux2_klein
```

Run a group:

```powershell
python scripts/run_pipelines.py --pipeline_groups all
```

> **SD3/SD3.5 Dynamic Shape:** SD3 and SD3.5 pipelines support dynamic shapes. The available
> resolutions are: 512×512, 512×768, 768×512, 576×1024, 1024×576, 768×1024,
> 1024×768, and 1024×1024. Enable in the YAML config with `args: { dynamic_shape: true }`.

Additional YAML config files:
- Custom YAML: pass with `--config path/to/custom.yaml`

### JSON Prompt Files

Prompt files are JSON arrays of text strings. Each string is used as a prompt for one generation pass.

```json
["a mountain landscape at sunset", "a futuristic city at night", "a flower garden in spring"]
```

Built-in prompt files in `config/`:
- `artistic_prompts.json` — diverse artistic styles
- `anime_prompts.json` — anime-style image prompts
- `alt_prompts.json` — alternative prompt collection

Pass a custom prompt file:

```powershell
python scripts/run_pipelines.py --pipelines sdxl_turbo --prompt_file config/my_prompts.json
```

Or pass a single inline prompt:

```powershell
python scripts/run_pipelines.py --pipelines sdxl_turbo --prompt "a beautiful mountain at sunrise"
```

---

## Running Pipelines

### Execution Modes

The framework supports 3 modes balancing speed and profiling depth:

**Quick Test Mode (default)** — 1 warmup + 1 profiled run, 2 images, basic timing.

```powershell
python scripts/run_pipelines.py --pipelines sdxl_turbo --prompt "a mountain landscape"
```

**Image-Only Mode (`--image-only`)** — 0 warmup + 1 run, 1 image, no profiling overhead.
Fastest mode for batch image generation.

```powershell
# Generate 5 images from a 5-prompt JSON file, no profiling
python scripts/run_pipelines.py --image-only --pipelines sd35_base \
    --prompt_file config/anime_prompts.json
```

**Benchmark Mode (`--benchmark`)** — 1 warmup + 10 profiled runs, 11 images, full timing stats.

```powershell
# Benchmark all pipelines with detailed profiling
python scripts/run_pipelines.py --benchmark --pipeline_groups all
```

### Argument Priority

Parameters are resolved in this order (highest to lowest):

1. Command-line arguments
2. Pipeline-specific YAML config
3. Global YAML defaults
4. Runner-level defaults (in `src/utils/model_config.py`)

### Pipeline Options Reference

All options can be passed on the command line or set in the YAML config file. Run
`python scripts/run_pipelines.py --help` for the latest reference.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--benchmark` | flag | False | Enable benchmark mode (10 profiling rounds) |
| `--image-only` | flag | False | Image generation only, no profiling |
| `--timeout` | int | 600 | Timeout in seconds per pipeline run |
| `--model_id` | list | | Run only pipelines with this model ID |
| `--pipelines` | list | | Run specific pipelines by name |
| `--pipeline_groups` | list | | Run a named pipeline group |
| `--list` | flag | False | List available pipelines and groups, then exit |
| `--prompt` | str | | Override prompt for all pipelines |
| `--prompt_file` | str | | JSON file with list of prompts |
| `--sd3-controlnet-mode` | str | `controlnet` | For SD3 ControlNet: `controlnet` or `text2img` |
| `--force-cpu` | flag | False | Force CPU-only execution (disables DML and all accelerators) |
| `--no-summary` | flag | False | Skip summary report and result file |
| `--config` | str | `config/pipeline_configs.yaml` | Custom YAML config file path |
| `--clean-images` | flag | False | Remove all images from `generated_images/` before run |
| `--clean-results` | flag | False | Remove all result files from `results/` before run |
| `--save-log` | flag | False | Save full console output to a log file |

> **CPU-only inference:** set environment variable `ORT_DISABLE_GPU=1` before running, or use
> `--force-cpu`. See [docs/07-developer-guide.md](07-developer-guide.md) §7.2.

---

## Output Files

### Generated Images

Images are saved to `generated_images/` (or your configured `output_path`). Filenames include
model name, resolution, step count, image index, and timestamp:

```
generated_images/stable-diffusion-v1-5_0_512x512_steps20_idx0_20260101_101132.png
```

Use `--clean-images` to clear the output directory before a run.

### Results Text Files

Execution summaries are saved in `results/` with the mode and timestamp in the filename:

```
results/pipeline_results_quick_20260101_100116.txt
results/pipeline_results_benchmark_20260101_102713.txt
```

Each file contains: execution status (success/failure), duration, timing data, model info, resolution,
and error messages for all pipelines in the session.

Use `--no-summary` to skip saving result files.

### Log Files

Use `--save-log` to save the full console output to a log file:

```
results/pipeline_log_quick_20260101_102713.txt
```

### Benchmark Files

After running in `--benchmark` mode, generate a CSV summary for Excel analysis:

```powershell
python scripts/process_benchmark_results.py
```

Output:

```csv
Model Name,Resolution,Inference Steps,NPU Models Load Time (s),All Models Load Time (s),
1st Gen Pipeline Time (s),Avg Pipeline Time (s),NPU Memory Usage (MB),Total Memory Usage (MB)
SD 1.5 Pipeline,512x512,20,3.90,3.90,5.98,5.98,1527.36,4412.52
```

---

## Community and Support

- **AMD Ryzen AI Documentation**: [https://ryzenai.docs.amd.com/](https://ryzenai.docs.amd.com/)
- **AMD Driver Support Portal**: [https://www.amd.com/en/support](https://www.amd.com/en/support)
- **AMD Developer Community Discord**: [https://discord.gg/amd-dev](https://discord.gg/amd-dev)
