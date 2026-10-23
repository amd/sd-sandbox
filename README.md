<div align="center">

# 🚀 AMD Stable Diffusion Sandbox

### High-Performance AI Stable Diffusion Image Generation on AMD Ryzen AI Hardware

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![AMD Ryzen AI](https://img.shields.io/badge/AMD-Ryzen%20AI-red.svg)](https://www.amd.com/en/products/ryzen-ai)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289da.svg)](https://discord.gg/amd-dev)

</div>

![sd_demo](test/assets/sd_demo.gif)

---

## 📑 Contents
- **[🎯 What is This?](#-what-is-this)** - Overview and introduction
- **[📜 Supported Models](#-supported-models)** - List of supported SD models
- **[🏞️ Image Generation Methods](#️-image-generation-methods)** - Description of text-to-image, image-to-image, inpainting, removal, outpainting
- **[📐 Hardware Prerequisites](#-hardware-prerequisites)** - AMD hardware requirements (Strix and MDS)
- **[🚀 Quick Start](#-quick-start)** - Installation process: get up and running
- **[🧩 Pipelines and Pipeline Groups Configuration](#-pipelines-and-pipeline-groups-configuration)** - Configure YAML and JSON files
- **[▶️ Running Pipelines](#️-running-pipelines)** - How to execute pipelines and pipeline groups
- **[📤 Output](#-output)** - Generated images, results, and logs
- **[💬 Community & Support](#-community--support)** - Get help and connect

---

## 🎯 What is This?

**AMD SD Sandbox** is a framework for running Stable Diffusion (SD) models accelerated by **AMD Ryzen AI hardware**. It supports two NPU platforms, **Strix (STX)** and **MDS**, from a single codebase and this single README; wherever the two differ, the section is labeled by platform. **MDS is a limited engagement and is not publicly available.** It provides an easy-to-use interface for evaluating, comparing, and deploying multiple SD pipelines. When developing AI-powered applications with SD, developers face several challenges:

| 🚫 Problem | ✅ Solution |
|-----------|------------|
| **Model Selection Complexity** - Dozens of SD models exist, each with different capabilities and tradeoffs, requiring separate tools and interfaces | **Unified Pipeline Interface** - Run any supported SD model with consistent commands and a single orchestration framework |
| **Hardware Optimization Guesswork** - Need to compare NPU vs GPU vs CPU performance to justify hardware investments and optimize deployments | **Built-in Performance Benchmarking** - Automated profiling with CPU baseline comparison to measure real acceleration gains with the NPU |
| **Tedious Manual Testing** - Testing multiple prompts across multiple models requires repetitive manual work and scattered results | **Automated Batch Processing** - Test hundreds of prompts across multiple models automatically with organized result management |
| **Code Modification Barriers** - Official examples require source code changes for simple customizations like prompts or resolution | **No-Code Customization** - Configure everything via YAML files and CLI arguments without adjusting source code |



The SD sandbox uses scripts to orchestrate GenAI-SD code distributed with Ryzen AI software. To use GenAI-SD without the orchestrator script please refer to the following: 
   [📚 GenAI-SD Documentation](#genai-sd-documentation)

---

## 📜 Supported Models

Here is a list of currently supported SD models. Note that these models have had nodes replaced with custom ops specifically designed for AMD NPUs. Models that aren't stored locally will be downloaded from Hugging Face on first use.

Models are published separately for each platform. MDS models use a `-medusa` suffix on the Hugging Face model ID (e.g. `amd/sdxl-turbo-amdnpu-medusa`). The platform is detected automatically from the NPU hardware, and can be forced with the `NPU_PLATFORM` environment variable (`stx` or `mds`). If detection fails, a `-medusa` model ID selects MDS; any other ID selects Strix.

These are the same NPU-optimized pipelines featured in the **[Ryzen AI Stable Diffusion Demo](https://ryzenai.docs.amd.com/en/latest/sd_demo.html#installation-steps)**. That demo is AMD's official preview, shipped with Ryzen AI software, of Stable Diffusion image generation running on Ryzen AI hardware. It shows how to drive each model directly from the `GenAI-SD` code — covering Text-to-Image, Image-to-Image, and ControlNet (Canny, Depth, Tile, Pose) plus SD3 inpainting (Strix only) — by calling the unified `run.py` entry point with a `--model_id`. This sandbox wraps those same pipelines in a configurable, batch-friendly orchestrator so you can run and benchmark many of them without editing source code.

The table below lists every model, with the Strix (RAI 1.8) and MDS Hugging Face model IDs side by side. Resolution and the Text-To-Image / Image-To-Image columns describe Strix (RAI 1.8). Anything that is different or unavailable on MDS is called out in the last column and marked with ❌ (not available on MDS) or ⚠️ (limited on MDS).

| Model | Strix (RAI 1.8) Hugging Face Model ID | MDS Hugging Face Model ID | Default Resolution | Text-To-Image | Image-To-Image | Differences on MDS |
|-------|---------------------------------------|------------------------------------|--------------------|---------------|----------------|--------------------|
| SD 1.5 | [amd/stable-diffusion-1.5-amdnpu](https://huggingface.co/amd/stable-diffusion-1.5-amdnpu) | [amd/stable-diffusion-1.5-amdnpu-medusa](https://huggingface.co/amd/stable-diffusion-1.5-amdnpu-medusa) | 512x512 | ✅ | | |
| SD 1.5 ControlNet (Canny) | [amd/sd1.5-controlnet-canny-amdnpu](https://huggingface.co/amd/sd1.5-controlnet-canny-amdnpu) | ❌ | 512x512 | | ☑️ | ❌ **Not available on MDS** |
| SD Turbo | [amd/sd-turbo-amdnpu](https://huggingface.co/amd/sd-turbo-amdnpu) | ❌ | 512x512 | ✅ | | ❌ **Not available on MDS** |
| SD XL Turbo | [amd/sdxl-turbo-amdnpu](https://huggingface.co/amd/sdxl-turbo-amdnpu) | [amd/sdxl-turbo-amdnpu-medusa](https://huggingface.co/amd/sdxl-turbo-amdnpu-medusa) | 512x512, 5x DynRes | ✅ | | ⚠️ No DynRes in the shipped config (fixed 512x512) |
| SD XL Base | [amd/sdxl-base-amdnpu](https://huggingface.co/amd/sdxl-base-amdnpu) | [amd/sdxl-base-amdnpu-medusa](https://huggingface.co/amd/sdxl-base-amdnpu-medusa) | 1024x1024, 20x DynRes | ✅ | | ⚠️ No DynRes in the shipped config (fixed 1024x1024) |
| SSD-1B | [amd/SSD-1B-amdnpu](https://huggingface.co/amd/SSD-1B-amdnpu) | [amd/SSD-1B-amdnpu-medusa](https://huggingface.co/amd/SSD-1B-amdnpu-medusa) | 1024x1024, 20x DynRes | ✅ | | ⚠️ No DynRes in the shipped config (fixed 1024x1024) |
| Segmind Vega | [amd/segmind-vega-amdnpu](https://huggingface.co/amd/segmind-vega-amdnpu) | [amd/segmind-vega-amdnpu-medusa](https://huggingface.co/amd/segmind-vega-amdnpu-medusa) | 1024x1024 | ✅ | ☑️ | |
| Playground 2.5 | [amd/playground-v2.5-1024px-aesthetic-amdnpu](https://huggingface.co/amd/playground-v2.5-1024px-aesthetic-amdnpu) | [amd/playground-v2.5-1024px-aesthetic-amdnpu-medusa](https://huggingface.co/amd/playground-v2.5-1024px-aesthetic-amdnpu-medusa) | 1024x1024, 20x DynRes | ✅ | | ⚠️ No DynRes in the shipped config (fixed 1024x1024) |
| DreamShaper XL | [amd/dreamshaper-xl-lightning-amdnpu](https://huggingface.co/amd/dreamshaper-xl-lightning-amdnpu) | [amd/dreamshaper-xl-lightning-amdnpu-medusa](https://huggingface.co/amd/dreamshaper-xl-lightning-amdnpu-medusa) | 1024x1024, 20x DynRes | ✅ | | ⚠️ No DynRes in the shipped config (fixed 1024x1024) |
| SD 3.0 Medium — Text-to-Image, ControlNet (Canny, Depth, Tile, Pose), Inpainting | [amd/stable-diffusion-3-medium-amdnpu](https://huggingface.co/amd/stable-diffusion-3-medium-amdnpu) | ❌ | 512x512, 20x DynRes (Inpainting: 1024x1024) | ✅ | ☑️ | ❌ **Not available on MDS** |
| SD 3.5 Medium — Text-to-Image | [amd/stable-diffusion-3.5-medium-amdnpu](https://huggingface.co/amd/stable-diffusion-3.5-medium-amdnpu) | [amd/stable-diffusion-3.5-medium-amdnpu-medusa](https://huggingface.co/amd/stable-diffusion-3.5-medium-amdnpu-medusa) | 512x512, 20x DynRes | ✅ | | ⚠️ Fixed 1024x1024 unless `dynamic_shape: true` is set |
| FLUX.1 Schnell | [amd/FLUX.1-schnell-amdnpu](https://huggingface.co/amd/FLUX.1-schnell-amdnpu) | [amd/FLUX.1-schnell-amdnpu-medusa](https://huggingface.co/amd/FLUX.1-schnell-amdnpu-medusa) | 512x512, 20x DynRes | ✅ | | ⚠️ No DynRes in the shipped config (fixed 1024x1024) |
| FLUX.2 Klein | [amd/FLUX.2-klein-4B-amdnpu](https://huggingface.co/amd/FLUX.2-klein-4B-amdnpu) | [amd/FLUX.2-klein-4B-amdnpu-medusa](https://huggingface.co/amd/FLUX.2-klein-4B-amdnpu-medusa) | 1024x1024 | ✅ | | ➕ **MDS only:** Image-To-Image reference-image editing via `--edit_image_path` |

**DynRes** indicates the number of dynamic-resolution presets a model supports (e.g. `5x`, `20x`) in addition to its default resolution. The exact width×height pairs for each preset are defined in the `config/dynamic_shape_*.json` files — the 20x grid spans 512x288 – 1024x1024 across 1:1, 3:4, 4:3, 9:16, and 16:9 aspect ratios.

**Which config file?** Strix uses `config/pipeline_configs_strx.yaml` (the default). MDS uses `config/pipeline_configs_mds.yaml`, passed with `--config`. See [Pipelines and Pipeline Groups Configuration](#-pipelines-and-pipeline-groups-configuration).

[Back to Table of Contents](#-contents)

---

## 🏞️ Image Generation Methods 

Here are some explanations of each of the SD methods. Note that pre- and post-processing padding and depadding are applied to the images where the resolutions do not match the model's inference resolution.

**🎨 Text-To-Image (T2I)** — Generate images from scratch using only text descriptions. Simply provide a prompt like "a mountain landscape at sunset" and the model creates a completely new image based on your description.
*Example use cases: concept art, marketing visuals, creative exploration, rapid prototyping*


![Text-to-Image Generation Process](test/assets/2025_Bousetouane_text-to-image.png)

> Image Source: Bousetouane, Fouad. 2025. "Generative AI for Vision: A Comprehensive Study of Frameworks and Applications." https://arxiv.org/abs/2501.18033v1.

<p align="center">• • •</p>

**🔄 Image-To-Image (I2I)** — Transform an existing image based on a text prompt while preserving certain structural or compositional elements. You provide both a reference image and a prompt, and the model generates a new image that follows the guidance of the input image while incorporating your text description.
*Example use cases: style transfer, image variations, artistic reinterpretation, photo enhancement*

> **FLUX.2 Klein image editing (MDS only).** On MDS, FLUX.2 Klein performs I2I as prompt-guided *image editing*: pass one or more reference images with `--edit_image_path`. With a single reference image you can restyle or alter it (e.g. *"Change the dress to red"*); with two reference images you can combine them (e.g. *"Replace the astronaut in image 1 with the girl in image 2"*). The `flux2_klein_i2i` (one image) and `flux2_klein_i2i_2` (two images) pipelines in `config/pipeline_configs_mds.yaml` are examples.

![Image-to-Image ControlNet Process](test/assets/2025_Bousetouane_image_to_image_controlnet.png)


---

## 📐 Hardware Prerequisites

Before getting started, make sure you meet the minimum hardware and OS requirements. **The requirements in the table below are for Strix (STX) only.**

> **MDS products are preview only.** MDS is the new-generation Ryzen AI NPU platform, available only as a limited engagement and not publicly available. These Strix requirements do not apply to MDS. MDS uses the `-medusa` model variants and its own transaction plugins, and is selected automatically when an MDS NPU is detected.

**Strix (STX) hardware requirements:**

| Series | Codename | Abbreviation | Graphics Model | Launch Year | Windows 11 | Linux |
|--------|----------|--------------|----------------|-------------------|-------------|---------|
| [Ryzen AI Max PRO 300 Series](https://www.amd.com/en/partner/articles/ryzen-ai-pro-300-series-processors.html) | Strix Halo | STX | Radeon 8000S Series | 2025 | ☑️ | |
| [Ryzen AI PRO 300 Series](https://www.amd.com/en/partner/articles/ryzen-ai-pro-300-series-processors.html) | Strix Point / Krackan Point | STX/KRK | Radeon 800M Series | 2025 | ☑️ | |
| [Ryzen AI Max 300 Series](https://www.amd.com/en/partner/articles/ryzen-ai-300-series-processors.html) | Strix Halo | STX | Radeon 8000S Series | 2025 | ☑️ | |
| [Ryzen AI 300 Series](https://www.amd.com/en/partner/articles/ryzen-ai-300-series-processors.html) | Strix Point | STX | Radeon 800M Series | 2025 | ☑️ | |

[Back to Table of Contents](#-contents)

---

## 🚀 Quick Start

#### Step 1: Install latest NPU Drivers and Ryzen AI SW

Follow the instructions here to download necessary NPU drivers and Ryzen AI SW: [Ryzen AI SW Installation Instructions](https://ryzenai.docs.amd.com/en/latest/inst.html). Please allow for around **30 minutes** to install all of the necessary components of Ryzen AI SW.

**MDS:** install the MDS Software instead of the Ryzen AI SW. The default install path is `C:\Program Files\RyzenAI\ng-1.8.0` and the default conda environment is `ryzen-ai-ng-1.8.0`.


#### Step 2: Copy GenAI-SD Code

The code found in `GenAI-SD` within the RAI installation directory is the only code needed to run the pipelines. Copy
the directory for your platform to a local directory of your choice:

| Platform | Source directory |
|----------|------------------|
| Strix | `C:\Program Files\RyzenAI\1.8.0\GenAI-SD` |
| MDS | `C:\Program Files\RyzenAI\ng-1.8.0\GenAI-SD` | 


#### Step 3: Configure Paths

Next, configure paths by editing [config/pipeline_configs_strx.yaml](config/pipeline_configs_strx.yaml) on Strix, or [config/pipeline_configs_mds.yaml](config/pipeline_configs_mds.yaml) on MDS. Relative paths are resolved from the sandbox root. The bare minimum that you need to change in order to run a pipeline is:

1. `test_path` : the location of the `test` directory in the `GenAI-SD` directory copied in Step 2

Optional paths that can be configured are:

2. `control_images_path`: the location of the `assets` directory in the `test` directory copied in Step 2. This is used for ControlNet pipelines that require control images. You can also specify a custom path here if desired.
3. `config_files_path`: the location of the `config` directory in the `GenAI-SD` directory copied in Step 2. This is used for dynamic shape configuration files used by SD3 pipelines. You can also specify a custom path here if desired.

```yaml
paths:
  test_path: "./test"
  control_images_path: "./test/assets"
  config_files_path: "./config"
```

More details on configurations can be found in the [Pipelines and Pipeline Groups Configuration](#-pipelines-and-pipeline-groups-configuration) section.

#### Step 4: Hugging Face Credentials

Supported models are downloaded from the Hugging Face Hub on first use, so you need to authenticate your machine with a Hugging Face access token.

1. Generate and copy a token from https://huggingface.co/settings/tokens. A token with **Read** access is sufficient to download the models. Some gated repos additionally require you to accept their license on the model page before the first download will succeed.
2. Log in from your terminal using the Hugging Face CLI (`hf`), which ships with the `huggingface_hub` package:
```powershell
hf auth login
```
Paste your token when prompted (or run it non-interactively with `hf auth login --token <YOUR_TOKEN>`). On success you should see, among other things:

```text
Login successful.
```

> **Note:** `hf auth login` is the current command and replaces the deprecated `huggingface-cli login`. For installation options (including running the CLI with `uv` via `uvx hf`, with no install), see the Hugging Face [CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and its [Using uv](https://huggingface.co/docs/huggingface_hub/en/guides/cli#using-uv) section.

#### Step 5: Activate RAI Conda Env

Activate the conda environment created during the Ryzen AI SW installation (Strix) or the MDS Software installation

```powershell
# Strix
conda activate ryzen-ai-1.8.0

# MDS
conda activate ryzen-ai-ng-1.8.0
```
#### Step 6: Verify Installation

Verify that the main run script shows the possible pipelines and pipeline groups: 

```bash
# Strix (uses config/pipeline_configs_strx.yaml by default)
python scripts/run_pipelines.py --list

# MDS
python scripts/run_pipelines.py --config config/pipeline_configs_mds.yaml --list
```

You should see a list of all available pipelines and groups of pipelines for your platform, similar to this (Strix output shown; MDS lists its own set of pipelines):
```text
[AUTO] Automatic provider selection (NPU > DML > CPU)

==================================================
Available Pipelines
==================================================
  sd_15                     - SD 1.5 Pipeline
  sd_turbo                  - SD Turbo Pipeline
  sdxl_base                 - SDXL Base Pipeline
  sdxl_turbo                - SDXL Turbo Pipeline
...

==================================================
Available Pipeline Groups
==================================================
  all                       - sd_15, sd_15_controlnet, sd_turbo, sdxl_turbo, sdxl_base, ...
...
```

You should now be able to run a pipeline with (on MDS, add `--config config/pipeline_configs_mds.yaml`):

```powershell
python scripts/run_pipelines.py --pipelines sd_15
```

And get an output similar to:

```text
[AUTO] Automatic provider selection (NPU > DML > CPU)

============================================================
Running 1 pipelines in QUICK TEST mode
============================================================

--- Pipeline 1/1: SD 1.5 Pipeline ---
     ---Loading ONNX Unet for DD
     2025-11-24 10:11:14,521 [INFO] Load unet/dd ...

...

     2025-11-24 10:11:44,580 [INFO] Total NPU memory usage: 1521.66MB (1.49GB)
[OK] SUCCESS in 36.5s
   [Dir] Images saved to: generated_images/

============================================================
SUMMARY - QUICK TEST MODE
============================================================
Total runs: 1
Successful: 1/1 (100.0%)
Total duration: 0:00:36.465767

Results saved to: C:\Users\User\bconsolv\sd-sandbox\results\pipeline_results_quick_20251124_101145.txt

```

[Back to Table of Contents](#-contents)

---

## 🧩 Pipelines and Pipeline Groups Configuration

Now that everything is installed, you are ready to configure and run SD pipelines. Pipelines are defined in a per-platform YAML configuration file, [config/pipeline_configs_strx.yaml](config/pipeline_configs_strx.yaml) for Strix and [config/pipeline_configs_mds.yaml](config/pipeline_configs_mds.yaml) for MDS, and can be run via the main [scripts/run_pipelines.py](scripts/run_pipelines.py) script.

A pipeline is a complete SD workflow configuration that includes a specific model (e.g., SD 1.5, SDXL, SD 3.0), execution script, and parameters (resolution, inference steps, guidance scale). Each pipeline represents a single configured image generation setup, from text/image input to final image output.

Pipelines can be arranged into **groups** for common workflows (e.g., all SDXL models, all ControlNet models, etc.) to simplify running multiple pipelines at once. To create a pipeline group, simply list the desired pipelines under a group name in your platform's YAML configuration file.

### 📁 Config Directory

An overview of the directory structure that contains pipeline files and configurations is given below. The [config](config) directory contains two file types: [YAML](#yaml-files) and [JSON](#json-files).

#### YAML Files
YAML files define pipeline configurations, parameters, and execution settings. Here are the important YAML files:
- [config/pipeline_configs_strx.yaml](config/pipeline_configs_strx.yaml) - Strix configuration; used by default when `--config` is not given
- [config/pipeline_configs_mds.yaml](config/pipeline_configs_mds.yaml) - MDS configuration; select it with `--config`
- `pipeline_custom_config.yaml` - If not using the default, custom YAML files can be used with the `--config` argument as below:
```powershell
python scripts/run_pipelines.py --pipelines sd_15 --config pipeline_custom_config.yaml
```

**Choosing the configuration for your hardware.** Both configuration files use the same format. Pick the one that matches your hardware with the `--config` argument:

```powershell
# Strix (default)
python scripts/run_pipelines.py --pipelines sd_15

# MDS
python scripts/run_pipelines.py --config config/pipeline_configs_mds.yaml --pipelines sd_15
```

The Strix configuration defines every pipeline; the MDS configuration defines a subset, plus two pipelines of its own. Pipelines marked ❌ are not available on MDS:

| Pipeline | Strix (`pipeline_configs_strx.yaml`) | MDS (`pipeline_configs_mds.yaml`) |
|---|---|---|
| `sd_15` | ✅ | ✅ |
| `sd_15_controlnet` | ✅ | ❌ **Not available on MDS** |
| `sd_turbo` | ✅ | ❌ **Not available on MDS** |
| `sdxl_turbo`, `sdxl_base`, `ssd_1b`, `playground_25`, `dreamshaper_xl` | ✅ | ✅ |
| `segmind_vega`, `segmind_vega_i2i` | ✅ | ✅ |
| `sd3_base` | ✅ | ❌ **Not available on MDS** |
| `sd3_controlnet_tile`, `sd3_controlnet_depth`, `sd3_controlnet_pose`, `sd3_controlnet_canny`, `sd3_controlnet_inpainting` | ✅ | ❌ **Not available on MDS** |
| `sd35_base` | ✅ | ✅ |
| `flux1_schnell`, `flux2_klein` | ✅ | ✅ |
| `flux2_klein_i2i`, `flux2_klein_i2i_2` | ❌ | ➕ **MDS only** |

Other differences between the two files:

| | Strix | MDS |
|---|---|---|
| Model IDs | `amd/<model>-amdnpu` | `amd/<model>-amdnpu-medusa` |
| Pipeline groups | `all`, `flux` | `all`, `flux` |

The `flux2_klein_i2i` and `flux2_klein_i2i_2` pipelines exist only in the MDS configuration. They use `edit_image_path` to pass one or two reference images to FLUX.2 Klein:

```yaml
  flux2_klein_i2i:
    name: "FLUX.2 Klein I2I Pipeline (1 image)"
    width: 1024
    height: 1024
    model_ids: ["amd/FLUX.2-klein-4B-amdnpu-medusa"]
    prompt: "Make the overalls red"
    args:
      edit_image_path: ["flux2_img3.png"]
      num_inference_steps: 4
      guidance_scale: 0.0

  flux2_klein_i2i_2:
    name: "FLUX.2 Klein I2I Pipeline (2 images)"
    width: 1024
    height: 1024
    model_ids: ["amd/FLUX.2-klein-4B-amdnpu-medusa"]
    prompt: "Put the man from img3 on the horse in img1"
    args:
      edit_image_path: ["flux2_img3.png", "flux2_img1.png"]
      num_inference_steps: 4
      guidance_scale: 0.0
```

Each configuration file has 3 main sections:
1. Global Pipeline Defaults
2. Pipeline Definitions
3. Pipeline Groups

**1. Global Pipeline Defaults**

Global defaults are defined at the top of the YAML file and are useful for setting parameters that apply to all pipelines, such as the output image resolution, or an external prompt file. The global defaults can always be overridden on a per-pipeline basis, by making a change to a variable in the pipeline definition.  

One example of setting a global default might be setting the `num_inference_steps` to 20. Every pipeline will use 20 inference steps unless overridden in the pipeline definition.

Under `paths`, `test_path` must point at the directory containing `run.py`. `models_path` is optional and defaults to `models`. A global `prompt_file` under `defaults` is optional; without it, each pipeline's `prompt` (or the global `prompt`) is used.

**2. Pipeline Definitions**

Pipelines are defined in the YAML configuration file under the `pipelines` section. Each pipeline has its own set of parameters that can override the global defaults. A pipeline definition uses the following keys:

1. A unique pipeline identifier (e.g., `sd_15`, `sdxl_turbo`). **(required)**
2. `name` — a human-readable display name (e.g., `"SD 1.5 Pipeline"`). **(required)**
3. `model_ids` — a **list** of one or more Hugging Face model IDs or local model folder names (e.g., `["amd/stable-diffusion-1.5-amdnpu"]`). **(required)**
4. `args` — a dictionary of model/runner parameters such as resolution, inference steps, and guidance scale.

> **🆕 New in RAI 1.8 — Unified entry point.** All pipelines now run through a single unified script, `run.py`. The previous per-model scripts (`run_sd.py`, `run_sd_xl.py`, `run_sd3.py`) have been consolidated, so you no longer specify a `script` for each pipeline — `run.py` is selected automatically. A `script` key is still accepted for advanced overrides but is rarely needed.

A minimal pipeline definition looks like this:

```yaml
  sd_15:
    name: "SD 1.5 Pipeline"
    model_ids: ["amd/stable-diffusion-1.5-amdnpu"]
    args:
      num_inference_steps: 20
```

If nothing else is specified, default values are used. Almost all hyperparameters can be overridden on a per-pipeline basis.

To see all available runner parameters, run `--help` on the unified entry point in the `test` directory:

```powershell
python test/run.py --help
```

**🆕 New in RAI 1.8 — the `args` dictionary (recommended)**

Pipeline parameters are now set using an `args` dictionary of `key: value` pairs. This replaces the older flat `extra_args` list. Each entry is converted to a `--key value` command-line flag automatically:

| YAML `args` value type | Example | Resulting CLI flag |
|------------------------|---------|--------------------|
| string / number | `num_inference_steps: 20` | `--num_inference_steps 20` |
| boolean `true` | `dynamic_shape: true` | `--dynamic_shape` (bare flag) |
| boolean `false` | `dynamic_shape: false` | *(omitted)* |
| list | `image_pads: [0, 64, 0, 64]` | `--image_pads 0 64 0 64` |

For example, a fully configured SD3 pipeline (Strix only; SD 3.0 is not available on MDS, which defines `sd35_base` instead):

```yaml
  sd3_base:
    name: "SD3 Pipeline"
    model_ids: ["amd/stable-diffusion-3-medium-amdnpu"]
    args:
      num_inference_steps: 30
      width: 1024
      height: 1024
      controlnet: "None"
      guidance_scale: 4.5
      dynamic_shape: true
```

> **Note:** The legacy `extra_args` list (e.g. `extra_args: ["--num_inference_steps", "20"]`) is still supported for backward compatibility. If both `args` and `extra_args` are present, the `args` entries are applied first.

Pipeline configurations are validated against the known set of runner arguments before each run, so unknown keys and out-of-range values are caught early with a clear warning rather than failing mid-run.

The **best** way to become familiar with the available parameters is to examine the [Strix](config/pipeline_configs_strx.yaml) or [MDS](config/pipeline_configs_mds.yaml) configuration file. 

**SD3/SD3.5 Dynamic Shape Support (Strix)**: All SD3 and SD3.5 pipelines support dynamic shapes, allowing a single model to handle multiple resolutions without recompilation. Enable it with `dynamic_shape: true` in the pipeline's `args`. The 20 supported resolutions are listed in [config/dynamic_shape_20x.json](config/dynamic_shape_20x.json) and span 512x288 – 1024x1024 across 1:1, 3:4, 4:3, 9:16, and 16:9 aspect ratios.

**SD3.5 Dynamic Shape Support (MDS)**: The shipped MDS configuration runs SD3.5 at a fixed 1024x1024. To run it at the resolutions listed above without recompilation, add `dynamic_shape: true` to the pipeline's `args`.

**3. Pipeline Groups**

For convenience, you can organize pipelines into **groups** for common workflows. A pipeline group is created by creating a unique group name and listing the desired pipelines under that group in the YAML file. 

For example, the `all_sdxl` tests all SDXL models:

```yaml
  all_sdxl:
    - sdxl_base
    - sdxl_turbo
    - ssd_1b
```

To run this pipeline group, you can run with the `--pipeline_groups` command as follows:

```powershell
python scripts/run_pipelines.py --pipeline_groups all_sdxl
```

You can also create your own pipeline groups by modifying the YAML file under `pipeline_groups` to specify which pipelines you would like to run:

```yaml
pipeline_groups:
  my_custom_pipeline_group: #group identifier
    - sd_turbo
    - sdxl_turbo
```

#### JSON Files
Each JSON file contains an array of text prompts passed sequentially to pipelines for batch testing. The example JSON files can be found here:
- [config/artistic_prompts.json](config/artistic_prompts.json), [config/anime_prompts.json](config/anime_prompts.json), [config/alt_prompts.json](config/alt_prompts.json) - Themed prompt collections
- `custom_prompts.json` - Your own custom prompt file
You can pass your own list of prompts with the `--prompt_file` argument. For example:

```powershell
python scripts/run_pipelines.py --pipelines sd_15 --prompt_file custom_prompts.json
```

[Back to Table of Contents](#-contents)

---

## ▶️ Running Pipelines

Pipelines are run by invoking the [scripts/run_pipelines.py](scripts/run_pipelines.py) with the desired pipeline or pipeline group specified. For example, we can run the `sd_15` pipeline with:

```powershell
python scripts/run_pipelines.py --pipelines sd_15
```

Or we can run the `all` pipelines group with: 

```powershell
python scripts/run_pipelines.py --pipeline_groups all
```

A full list of pipeline parameters and their descriptions can be found by running `--help` on the main script:

```powershell
python scripts/run_pipelines.py --help
```

### Execution Modes

The framework has three execution modes. They differ only in how many times each prompt is generated and how much timing information is reported. In every mode, **one image is saved per prompt and resolution**.

| Mode | Generations per prompt | Timing reported | Image filename tag | Results file |
|------|------------------------|-----------------|--------------------|--------------|
| **Image-Only** | 1 | Model load summary and a single pipeline execution time (includes first-run overhead) | `batch` | `pipeline_results_image_only_*.txt` |
| **Quick Test** | 2 (1 warmup + 1 profiled) | Model load summary, warmup (1st gen) time, average pipeline time, per-component times, memory | `profiling` | `pipeline_results_quick_*.txt` |
| **Benchmark** | 11 (1 warmup + 10 profiled) | Same as Quick Test, averaged over 10 rounds | `profiling` | `pipeline_results_benchmark_*.txt` |

The saved image always comes from the first generation; the profiled rounds are used for timing only. If a run mixes Image-Only and Quick Test pipelines (see [Setting the mode per pipeline](#setting-the-mode-per-pipeline)), the results file is named `pipeline_results_mixed_*.txt`. Log files saved with `--save-log` use the same mode in their name (e.g. `pipeline_log_quick_test_*.txt`).

**Quick Test Mode is the default** on both Strix and MDS; neither shipped configuration enables `image_only`. The mode for each pipeline is chosen in this order (highest first):

1. `--benchmark` — always Benchmark Mode, regardless of any `image_only` setting
2. `--image-only` — Image-Only Mode for every pipeline
3. `image_only` set on the pipeline in the YAML
4. `image_only` under `defaults` in the YAML
5. Otherwise, Quick Test Mode

#### 1. Image-Only Mode

Use this for generating images as fast as possible, e.g. batch-generating images from a prompt file. Each prompt is generated once with no warmup or profiling.

```powershell
# Strix
python scripts/run_pipelines.py --image-only --pipelines sd3_base --prompt_file config/anime_prompts.json

# MDS
python scripts/run_pipelines.py --config config/pipeline_configs_mds.yaml --image-only --pipelines sd35_base --prompt_file config/anime_prompts.json
```
*Output: 5 images (`anime_prompts.json` contains 5 prompts), no profiling metrics*

To make Image-Only the default for every pipeline without passing `--image-only`, set `image_only: true` under `defaults` in the YAML.

#### 2. Quick Test Mode

Use this for a quick performance check. Each prompt is generated once as a warmup and once more for profiling, so the report separates the first-generation time from the steady-state time.

```powershell
# Strix
python scripts/run_pipelines.py --pipelines sdxl_turbo

# MDS
python scripts/run_pipelines.py --config config/pipeline_configs_mds.yaml --pipelines sdxl_turbo
```
*Output: 1 image per prompt, warmup and average timing statistics*

#### 3. Benchmark Mode

Use this for performance measurement. Each prompt is generated once as a warmup, followed by 10 profiled rounds whose times are averaged. Benchmark Mode works the same on both platforms and needs no YAML changes.

```powershell
# Strix
python scripts/run_pipelines.py --benchmark --pipelines sdxl_turbo

# MDS
python scripts/run_pipelines.py --config config/pipeline_configs_mds.yaml --benchmark --pipelines sdxl_turbo
```
*Output: 1 image per prompt, timing averaged over 10 rounds.* The log shows `Running 10 profiling rounds...` followed by `__ROUND_COMPLETE__ 1/10` … `10/10`. Benchmark results files can be converted to CSV; see [Benchmark Files](#benchmark-files).

#### Setting the mode per pipeline

Add a top-level `image_only` key to a pipeline definition (next to `name` and `model_ids`, not inside `args`) to override the `defaults` value for that pipeline only. For example, to profile every pipeline except FLUX.1 Schnell, which only generates images:

```yaml
defaults:
  image_only: false

pipelines:
  flux1_schnell:
    name: "FLUX.1 Schnell Pipeline"
    image_only: true
    model_ids: ["amd/FLUX.1-schnell-amdnpu"]
    args:
      num_inference_steps: 4
```

The reverse also works: set `image_only: true` under `defaults` and `image_only: false` on the pipelines you want to profile. `--image-only` and `--benchmark` on the command line override per-pipeline settings.

### Argument priority 

The framework support argument priorities based on the following order (highest to lowest): 
1. Command-line arguments
2. Pipeline-specific YAML config
3. Global YAML defaults
4. Script-level defaults

### Pipeline Options

The following are framework-level options that can be passed at the command line, or set in the YAML configuration file. Please **read** this carefully as it describes all of the available options for running pipelines. You can run the `--help` argument to get the same information at the command line:

```powershell
python scripts/run_pipelines.py --help
```

| Option | Type | Default | Help |
|--------|------|---------|------|
| --benchmark | flag | False | Enable benchmark mode (10 profiling rounds). Default is quick mode (1 round). |
| --image-only | flag | False | Generate images only without profiling (batch mode). Disables warmup and profiling rounds for fastest execution. Overrides `image_only` in the YAML. `--benchmark` takes precedence over `--image-only`. |
| --timeout | int | 600 | Timeout in seconds for each run (default: 600) |
| --model_id | list |  | Run only these model IDs (e.g. --model_id amd/stable-diffusion-1.5-amdnpu) |
| --pipelines | list |  | Run specific pipelines by name (e.g. --pipelines sd_15 sd3_base) |
| --pipeline_groups | list |  | Run pipeline groups (e.g. --pipeline_groups all_sd3 quick_test) |
| --list | flag | False | List available pipelines and groups, then exit |
| --prompt | str |  | Custom prompt to use for all pipelines |
| --prompt_file | str |  | Path to JSON file containing list of prompts to run through each pipeline |
| --sd3-controlnet-mode | str | controlnet | For SD3 ControlNet with prompt files: 'controlnet' uses custom prompts with ControlNet enabled (recommended), 'text2img' converts to text-to-image mode without ControlNet guidance (default: controlnet) |
| --force-cpu | flag | False | Force CPU-only execution (disables DML and all accelerators). Overrides config setting. |
| --no-summary | flag | False | Skip printing the detailed summary report and skip saving results to disk |
| --config | str | config/pipeline_configs_strx.yaml | Path to custom YAML configuration file (default: config/pipeline_configs_strx.yaml; use config/pipeline_configs_mds.yaml for MDS) |
| --clean-images | flag | False | Remove all existing images and Excel files from generated_images folder before running pipelines |
| --clean-results | flag | False | Remove all existing result files from results folder before running pipelines |
| --save-log | flag | False | Save complete log output to a text file when any pipeline has streaming output enabled |
| --traceback | flag | False | Include full traceback in error messages for detailed debugging |

[Back to Table of Contents](#-contents)

---

## 📤 Output

When running [scripts/run_pipelines.py](scripts/run_pipelines.py), the outputs are images and result text files in specific locations with organized naming conventions.

### Generated Images

Images by default can be found in `test/generated_images/`. Filenames include model name, image index, run type, resolution, number of steps, prompt index, and timestamp (e.g., `test/generated_images/FLUX.1-schnell-amdnpu_img0_batch_1024x1024_steps4_prompt0_20261012_121816.png`).

> **Note:** You can use the `--clean-images` flag to remove all existing images from the `generated_images` folder before running any pipelines. 

### Results Text Files

Results files can be found in `results/` directory (e.g., `results/pipeline_results_quick_20251124_100116.txt`). They contain execution status (success/failure), duration, timing data, model information, resolution, and error messages for all pipelines run in a session.

> **Note:** You can disable the generation of summary files by using the `--no-summary` flag when running the main script. You can also use the `--clean-results` flag to remove all existing result files from the `results` folder before running any pipelines.

### Log Files

You can choose to save all lines printed to the console at runtime with the`--save-log` flag when running the main script. Log files are saved in the `results/` directory and contain the complete console output from all pipeline executions, including detailed progress messages, warnings, and debug information (e.g., `results\pipeline_log_quick_test_20251124_102713.txt`).

### Benchmark Files

When using the benchmarking mode (`--benchmark`), the results text file includes the benchmarking information (e.g., `results/pipeline_results_benchmark_20251124_102713.txt`). 

After creating the `pipeline_results_benchmark_*.txt` files, you can run the [scripts/process_benchmark_results.py](scripts/process_benchmark_results.py) script to extract benchmark data. This CSV can be imported into Excel for performance analysis and comparison. To generate the CSV file, simply run:

```powershell
python scripts/process_benchmark_results.py
```

And you should get an output of CSV files (e.g., `benchmark_summary_20251124_104406.csv`) with an output similar to:

```csv
Model Name,Resolution,Inference Steps,NPU Models Load Time (s),All Models Load Time (s),1st Gen Pipeline Time (s),Avg Pipeline Time (s),Seconds/Iter,NPU Memory Usage (MB),Total Memory Usage (MB)
SD 1.5 Pipeline,512x512,20,3.898902,3.899352,5.979673,5.978549,0.298927,1527.36,4412.52
```

[Back to Table of Contents](#-contents)

---

## GenAI-SD Documentation

These guides come from the `GenAI-SD` folder shipped with the **Ryzen AI 1.8 software installer** (`C:\Program Files\RyzenAI\1.8.0\GenAI-SD`) for Strix, or the MDS software installer (`C:\Program Files\RyzenAI\ng-1.8.0\GenAI-SD`) for MDS — the same folder you copy in [Step 2](#step-2-copy-genai-sd-code). They describe how to use the GenAI-SD code **directly** — without this sandbox's orchestrator — covering driver and conda setup, the per-model `run.py` commands, troubleshooting, and more. Once you've copied that folder locally you can follow these documents in it; the copies below are mirrored in this repository's [`docs/`](docs) directory for convenience.

| Document | Contents |
|----------|----------|
| [docs/01-quick-start.md](docs/01-quick-start.md) | Driver installation, conda setup, first run |
| [docs/02-user-guide.md](docs/02-user-guide.md) | Batch pipelines, YAML config, execution modes |
| [docs/03-model-reference-guide.md](docs/03-model-reference-guide.md) | All model commands and CLI arguments |
| [docs/04-model-generate-guide.md](docs/04-model-generate-guide.md) | DD model creation workflow (`onnx_utils`, Windows only) |
| [docs/05-linux-guide.md](docs/05-linux-guide.md) | Linux setup |
| [docs/06-faq.md](docs/06-faq.md) | Top 15 failure modes and solutions |
| [docs/07-developer-guide.md](docs/07-developer-guide.md) | Adding new pipelines to the codebase |
| [docs/08-glossary.md](docs/08-glossary.md) | NPU, DD, ControlNet, InPainting, and more |

[Back to Table of Contents](#-contents)

---

## 💬 Community & Support

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-Join%20AMD%20Developer%20Community-7289da.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/amd-dev)

</div>

- **AMD Ryzen AI Documentation**: [https://ryzenai.docs.amd.com/](https://ryzenai.docs.amd.com/)
- **AMD Driver Support Portal**: [https://www.amd.com/en/support](https://www.amd.com/en/support)

[Back to Table of Contents](#-contents)

---

<div align="center">

**⭐ If this framework helped your development, consider starring the repository! ⭐**

*Built with ❤️ by AMD for the AI developer community*

</div>

[Back to Table of Contents](#-contents)


---

## License

© 2024–2026 Advanced Micro Devices, Inc. See [LICENSE](LICENSE).

