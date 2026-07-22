# Linux Setup Guide

This guide walks you through installing all required software and running your first Stable Diffusion
image generation on an AMD Ryzen AI device under Linux.

> **Note:** Model preprocessing (`onnx_utils preprocess`) and partition (`onnx_utils partition`) are
> not currently supported on Linux. Use pre-partitioned models from Hugging Face (`amd/` namespace),
> or partition on Windows first and copy the model to your Linux machine for inference.

## Prerequisites

- AMD Ryzen AI device (STX/KRK — Strix / Krackan)
- [Ubuntu 24.04 LTS](https://ryzenai.docs.amd.com/en/latest/linux.html), kernel >= 6.10, Python 3.12.x
- At least 16 GB system RAM (32 GB+ for SDXL/SD3; 64 GB recommended for large batch runs)
- Sufficient disk space for models and the conda environment

---

## Step 1: Install NPU Drivers

Follow the [Ryzen AI Linux installation instructions](https://ryzenai.docs.amd.com/en/latest/linux.html)
to install system dependencies, NPU driver `.deb` packages, and XRT. Then verify:

```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

### Configure NPU to Performance Mode (Optional) {#performance-mode}

```bash
xrt-smi configure --pmode performance
```

Add the `export`/`source` lines above to `~/.bashrc` for permanent effect. See the
[xrt-smi reference](https://ryzenai.docs.amd.com/en/latest/xrt_smi.html#xrt-smi-configure) for details.

---

## Step 2: Set Up the Environment {#environment}

### Option A: Ryzen AI Software Installer (Recommended)

Download `ryzen_ai-*.tgz` from the
[Ryzen AI Linux installation page](https://ryzenai.docs.amd.com/en/latest/linux.html), then install:

```bash
mkdir ryzen_ai && cp ryzen_ai-*.tgz ryzen_ai/ && cd ryzen_ai
tar -xvzf ryzen_ai-*.tgz
./install_ryzen_ai.sh -a yes -p <TARGET-PATH>/venv
source <TARGET-PATH>/venv/bin/activate
echo $RYZEN_AI_INSTALLATION_PATH   # verify
```

### Option B: Create a Conda Environment

```bash
conda env create --file=env_linux_py312.yaml
conda activate ryzenai-stable-diffusion
```

Use this only when the Ryzen AI installer is unavailable. Runtime libraries must be set up manually in Step 3.

---

## Step 3: Install Custom Ops / Runtime Libraries {#runtime-libraries}

Linux inference requires `libonnxruntime_providers_ryzenai.so`. GenAI-SD auto-detects it from
`$RYZEN_AI_INSTALLATION_PATH/deployment/`, `$ONNX_UTILS_ROOT/build/install/bin/`, or `<project>/lib/`.

**Option A:** skip if the file exists under `$RYZEN_AI_INSTALLATION_PATH/deployment/`.

**Option B:** copy the `.so` and companion libraries into `GenAI-SD/lib/`:

```bash
cp "$RYZEN_AI_INSTALLATION_PATH/deployment/libonnxruntime_providers_ryzenai.so" ~/projects/GenAI-SD/lib/
cp "$RYZEN_AI_INSTALLATION_PATH/deployment/"*.so ~/projects/GenAI-SD/lib/
```

---

## Step 4: Get the GenAI-SD Code

GenAI-SD is included with the Ryzen AI Software installation:

```
$RYZEN_AI_INSTALLATION_PATH/GenAI-SD
```

Copy it to a working directory and `cd` there:

```bash
cp -r "$RYZEN_AI_INSTALLATION_PATH/GenAI-SD" ~/projects/GenAI-SD
cd ~/projects/GenAI-SD
```

---

## Step 5: Authenticate with Hugging Face {#huggingface-auth}

Pre-partitioned NPU models are hosted on Hugging Face under the `amd/` namespace. They download
automatically on first run — no local `onnx_utils` workflow is needed on Linux.

Some models (especially SD3) are gated. Complete authentication once:

1. Request access at the model's Hugging Face page and accept the license before logging in
   (e.g. [amd/stable-diffusion-3-medium-amdnpu](https://huggingface.co/amd/stable-diffusion-3-medium-amdnpu)).
2. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with
   *Read access to all public gated repos*.
3. Authenticate:

```bash
huggingface-cli login
```

Alternatively, copy a model tree partitioned on Windows and pass `--model_path` in Step 6.

Model IDs and per-model CLI flags are listed in
[docs/03-model-reference-guide.md](03-model-reference-guide.md).

---

## Step 6: Run Your First Model

### Session environment

Run these before each session (add to `~/.bashrc` to persist XRT setup):

```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=~/projects/GenAI-SD/lib:$LD_LIBRARY_PATH   # Option B only
```

For SD3 / SD3.5 / FLUX dynamic-shape models, also set:

```bash
export DD_PLUGINS_ROOT=~/projects/GenAI-SD/lib/transaction/stx
```

Override the model cache directory (default: `<project>/models/`):

```bash
export RYZENAI_GENAI_SD_MODELS_ROOT=/data/my_models
```

### Run

```bash
cd ~/projects/GenAI-SD
python scripts/run_pipelines.py --pipelines sdxl_turbo
```

Or run a single model directly:

```bash
python test/run.py --model_id amd/sdxl-turbo-amdnpu --prompt "a scenic mountain landscape"
```

Generated images are saved to `generated_images/`.

---

## Step 7: Verify Your Installation

```bash
xrt-smi examine                                          # NPU visible
cd "$RYZEN_AI_INSTALLATION_PATH/quicktest" && python quicktest.py   # Option A: expect "Test Finished"
python scripts/run_pipelines.py --list                   # pipelines available
```

Re-run Step 6 and confirm `generated_images/` contains a new image with no missing-library errors.

---

## Next Steps

- **More model commands**: [docs/03-model-reference-guide.md](03-model-reference-guide.md)
- **Batch runs**: [docs/02-user-guide.md](02-user-guide.md)
- **Windows setup**: [docs/01-quick-start.md](01-quick-start.md)
- **Build DD models**: [docs/04-model-generate-guide.md](04-model-generate-guide.md) (Windows only)

---

## Troubleshooting {#troubleshooting}

| Symptom | Fix |
|---------|-----|
| `could not find XRT shared library` | `source /opt/xilinx/xrt/setup.sh`; add `export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` |
| `libonnxruntime_providers_ryzenai.so` not found | Verify Step 3; set `RYZEN_AI_INSTALLATION_PATH` |
| `cannot open shared object file` | Add `GenAI-SD/lib/` to `LD_LIBRARY_PATH` |
| `Cannot find custom op: DynamicDispatch` | Custom ops plugin not loaded — see Step 3; or `--custom_op_path lib/libonnxruntime_providers_ryzenai.so` |
| `401 Unauthorized` / gated model on HF | Complete Step 5 (`huggingface-cli login`) |
| Dynamic shape `shape mismatch` | `export DD_PLUGINS_ROOT=~/projects/GenAI-SD/lib/transaction/stx` |

---

## Driver Installation Reference {#driver-installation}

| Component | Source |
|-----------|--------|
| NPU driver + XRT + Ryzen AI Software | [Ryzen AI Linux installation guide](https://ryzenai.docs.amd.com/en/latest/linux.html) |
| GenAI-SD code | `$RYZEN_AI_INSTALLATION_PATH/GenAI-SD` |
| Custom ops (`.so`) | `$RYZEN_AI_INSTALLATION_PATH/deployment/` or `lib/` |
| Pre-partitioned models | Hugging Face `amd/` namespace; or copy from Windows |
