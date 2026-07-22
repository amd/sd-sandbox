# Quick Start Guide

This guide walks you through installing all required software and running your first Stable Diffusion
image generation on an AMD Ryzen AI device. This is the single source of truth for environment setup —
all other documents link here instead of repeating these steps.

## Prerequisites

Before you begin, confirm you have:

- A supported AMD Ryzen AI device (see [Hardware Requirements in README](../README.md) or
  [ORCHESTRATOR_README.md](../ORCHESTRATOR_README.md#-hardware-prerequisites))
- Windows 11 (Linux: [docs/05-linux-guide.md](05-linux-guide.md))
- At least 16 GB system RAM (32 GB+ recommended for SDXL/SD3 models)
- At least 16 GB of disk space for models and the conda environment
- An internet connection for the initial setup

---

## Step 1: Install NPU Drivers

> Skip this step if you have already installed NPU drivers as part of the Ryzen AI MSI installer.

### NPU Driver

For Krackan/Strix devices, follow the official
[Ryzen AI installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html)
to download and install the NPU driver.

### iGPU Driver

Download and install the AMD iGPU
[driver](https://www.amd.com/en/support/download/drivers.html) if your machine does not already
have one.

### Configure NPU to Performance Mode (Optional) {#performance-mode}

To achieve peak performance, set the NPU to high-performance mode:

```bash
xrt-smi configure --pmode performance
```

For details, see the
[xrt-smi configure reference](https://ryzenai.docs.amd.com/en/latest/xrt_smi.html#xrt-smi-configure).

### Adjust NPU Available Memory (Optional) {#npu-memory}

By default NPU can use 50% of system main memory at maximum. On low memory machines such as 16GiB ones, 
8GiB NPU memory may not be sufficient for large SD models. To enable larger NPU available memory, open Registry
Editor (press Win + R to open the Run dialog, type `regedit`, and press Enter) and navigate to key 
[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager\SystemPartitionCommitLimitPercentage].
If no such key exist, create a new DWORD value with this key. Set its value to an integer in the range [50, 100] 
and restart system to enlarge NPU available memory. You can check if it takes effect in Task Manager.

---

## Step 2: Set Up the Conda Environment {#conda-environment}

Choose the setup path that matches your situation:

### Option A: Ryzen AI MSI Installer (Recommended)

1. Download and install the Ryzen AI Software from the
   [Ryzen AI installation page](https://ryzenai.docs.amd.com/en/latest/inst.html). Allow approximately
   30 minutes for the full installation.
2. Activate the installed environment:

```powershell
conda activate ryzen-ai-1.8.0
```

The MSI installer automatically installs all required packages including the `amd_atom` wheel and
the `ryzenai-dynamic-dispatch` and `ryzenai-onnx-utils` packages.

### Option B: Create a New Conda Environment

If you are not using the Ryzen AI MSI, create a conda environment from the provided YAML file.
Choose the Python version matching your use case:

**Python 3.13**

```bash
conda update -n base -c defaults conda
conda env create --file=env_py313.yaml
conda activate ryzenai-stable-diffusion
```

**Python 3.12** (recommended for broadest compatibility)

```bash
conda update -n base -c defaults conda
conda env create --file=env_py312.yaml
conda activate ryzenai-stable-diffusion
```

### Option C: VAIP Custom OP Flow

For the VAIP ORT custom operator flow (advanced), use the dedicated environment file:

```bash
conda env create --file=env_py312_vaip.yaml
conda activate vaip_sd
```

> **Note:** The Python and ONNX Runtime versions in this environment must remain compatible with
> your VOE package build. Update this file whenever there are incompatible VOE changes.

---

## Step 3: Install the ATOM Package {#package-installation}

> Skip this step if you used the Ryzen AI MSI installer — ATOM is already included.

The `amd_atom` package provides the AMD NPU runtime libraries. It is distributed as part of the
[Ryzen AI Software](https://ryzenai.docs.amd.com/en/latest/inst.html).

Once you have the wheel file from the Ryzen AI Software distribution, install it:

```bash
# Python 3.13
pip install amd_atom-1.0-cp313-cp313-win_amd64.whl

# Python 3.12
pip install amd_atom-1.0-cp312-cp312-win_amd64.whl
```

To verify the installation:

```bash
pip show amd_atom
```

---

## Step 4: Get the GenAI-SD Code

The GenAI-SD code is included with the Ryzen AI Software installation at:

```
C:\Program Files\RyzenAI\1.8.0\GenAI-SD
```

Copy the `GenAI-SD` directory to a local working directory of your choice:

```powershell
Copy-Item "C:\Program Files\RyzenAI\1.8.0\GenAI-SD" "C:\Users\YourName\projects\GenAI-SD" -Recurse
```

Then open a terminal in your copied directory:

```powershell
cd C:\Users\YourName\projects\GenAI-SD
```

---

## Step 5: Authenticate with Hugging Face {#huggingface-auth}

Some models (especially SD3 variants) require Hugging Face access. Complete this step once:

1. Visit [https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
   and request access to the gated model.
2. Generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   with *Read access to all public gated repos*.
3. Authenticate:

```powershell
huggingface-cli login
```

You should see:

```
Login successful.
```

---

## Step 6: Run Your First Model

With the environment active, run the SDXL Turbo pipeline. Models are automatically downloaded from
Hugging Face on first use and cached under `models/` in the project root.

> **Custom download path:** set `RYZENAI_GENAI_SD_MODELS_ROOT` to redirect the cache:
> ```powershell
> $env:RYZENAI_GENAI_SD_MODELS_ROOT = "D:\my_models"
> ```

For SD3 / SD3.5 / FLUX dynamic-shape models, set the DD plugin path before running:

```powershell
$env:DD_PLUGINS_ROOT = ".\lib\transaction\stx"
```

```powershell
# Make sure your conda environment is active
conda activate ryzen-ai-1.8.0

# Run SDXL Turbo — pre-partitioned amd/ models download automatically from Hugging Face
python scripts/run_pipelines.py --pipelines sdxl_turbo
```

> **Note:** Hugging Face `amd/` models are already partitioned and compiled — you do **not** need
> `--enable_compile` for them. Use `--enable_compile` (`-c`) only on the **first run with a locally
> built DD model** (see [docs/04-model-generate-guide.md](04-model-generate-guide.md)). The flag is
> supported by `test/run.py`; for the orchestrator, add `enable_compile: true` under the pipeline's
> `args` in `config/pipeline_configs.yaml`.

On success, you will see output similar to:

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
```

Generated images are saved to `generated_images/` with a timestamped filename.

Look for NPU activity in the log — e.g. `Total NPU memory usage` — to confirm the NPU is being used.

---

## Step 7: Verify Your Installation

Run the list command to confirm all pipelines are available:

```powershell
python scripts/run_pipelines.py --list
```

You should see all available pipelines and pipeline groups defined in `config/pipeline_configs.yaml`
(e.g. `sd_15`, `sdxl_turbo`, `sd3_base`, `sd35_base`, `flux1_schnell`, `flux2_klein`, and the `all` group).

---

## Next Steps

- **Run more models**: See [docs/03-model-reference-guide.md](03-model-reference-guide.md) for all supported
  models and their command-line options.
- **Batch runs**: See [docs/02-user-guide.md](02-user-guide.md) to configure YAML pipelines and
  run multiple models in one session.
- **Troubleshoot**: See [docs/06-faq.md](06-faq.md) if you encounter any issues.
- **Linux setup**: See [docs/05-linux-guide.md](05-linux-guide.md).
- **Build DD models**: See [docs/04-model-generate-guide.md](04-model-generate-guide.md) for the full
  Dynamic Dispatch model creation workflow.

---

## Driver Installation Reference {#driver-installation}

This section is the canonical location for driver installation instructions.

| Component | Source |
|-----------|--------|
| NPU driver | [Ryzen AI installation guide](https://ryzenai.docs.amd.com/en/latest/inst.html) |
| iGPU driver | [AMD driver support portal](https://www.amd.com/en/support/download/drivers.html) |
| Ryzen AI Software (MSI) | [Ryzen AI installation page](https://ryzenai.docs.amd.com/en/latest/inst.html) |
| `amd_atom` wheel package | Included in Ryzen AI Software; see Step 3 above |
| `ryzenai-dynamic-dispatch` | Included in Ryzen AI MSI; see [docs/04-model-generate-guide.md](04-model-generate-guide.md) |
| `ryzenai-onnx-utils` | Included in Ryzen AI MSI; see [docs/04-model-generate-guide.md](04-model-generate-guide.md) |
