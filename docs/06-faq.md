# FAQ and Troubleshooting

This document covers common failure modes encountered when running Stable Diffusion models
on AMD Ryzen AI hardware, along with step-by-step solutions.

> For DD model creation issues, see [docs/04-model-generate-guide.md#troubleshooting](04-model-generate-guide.md#troubleshooting).
> Linux: [docs/05-linux-guide.md](05-linux-guide.md).

---

## Installation Issues

### FAQ-01: `conda env create` fails with package conflict

**Symptom:**
```
UnsatisfiableError: The following specifications were found to be incompatible with each other
```

**Fix:**
1. Update conda base first:
   ```bash
   conda update -n base -c defaults conda
   ```
2. Try creating the environment with the `--no-default-packages` flag:
   ```bash
   conda env create --file=env_py312.yaml --no-default-packages
   ```
3. If using an existing Ryzen AI MSI environment, use that directly instead of creating a new one:
   ```bash
   conda activate ryzen-ai-1.8.0
   ```

---

### FAQ-02: `pip install amd_atom-1.0-*.whl` fails with "No module named pip" or permission error

**Symptom:**
```
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied
```

**Fix:**
1. Make sure the conda environment is active before installing:
   ```bash
   conda activate ryzenai-stable-diffusion
   ```
2. Do not use `sudo pip install` on Windows. If permission errors persist, add `--user`:
   ```bash
   pip install amd_atom-1.0-cp312-cp312-win_amd64.whl --user
   ```

---

### FAQ-03: `pip show ryzenai-dynamic-dispatch` shows "not found"

**Symptom:**
```
WARNING: Package(s) not found: ryzenai-dynamic-dispatch
```

**Fix:**
These packages are included in the Ryzen AI MSI installer. If you see this, you either:
- Are not using the correct conda environment: run `conda activate ryzen-ai-1.8.0` and check again.
- Did not install via MSI: install Ryzen AI Software following [docs/01-quick-start.md](01-quick-start.md).

---

## Model Loading Issues

### FAQ-04: Model download fails with "401 Unauthorized" or "403 Forbidden"

**Symptom:**
```
OSError: We couldn't connect to 'https://huggingface.co' to load model configuration file ...
You might be trying to access a private or gated repo.
```

**Fix:**
1. Log in to Hugging Face:
   ```bash
   huggingface-cli login
   ```
2. If accessing a gated model (SD3, SD3.5), accept the license at the model's HF page first.
3. If behind a corporate proxy, set:
   ```bash
   set HTTPS_PROXY=http://your-proxy:port
   ```
4. To redirect all downloads to a different drive or directory, set:
   ```powershell
   $env:RYZENAI_GENAI_SD_MODELS_ROOT = "D:\my_models"
   ```
   Models are cached under `models/` in the project root by default; setting this variable
   overrides that location for all subsequent runs.

---

### FAQ-05: "FileNotFoundError: model.onnx not found" when using `--model_path`

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/models/unet/model.onnx'
```

**Fix:**
The model directory structure must match what the runner expects. See [docs/03-model-reference-guide.md](03-model-reference-guide.md)
for the correct directory layout for each model family.

Common mistakes:
- Missing the intermediate `normal/` or `common/` directory in the SD3 layout
- Large files not downloaded from git-lfs (check with `git lfs status`)
- Model path pointing to the parent directory instead of the specific model subdirectory

---

## Inference Issues

### FAQ-06: First run hangs for several minutes without output

**Symptom:** The script starts but produces no output for 3–10 minutes, then resumes.

**Explanation:** This happens when running with `--enable_compile` on a **locally built DD model**
(not required for pre-partitioned Hugging Face `amd/` models). The DD fusion runtime compiles
kernels for your specific NPU hardware — a one-time cost per model per machine.
Typical compile times: SD1.5 ~3 min, SDXL ~8 min, SD3/SD3.5 ~15 min, FLUX ~20 min.

**Fix:** Wait for compilation to complete. Subsequent runs without `--enable_compile` start in seconds.
If you are using an `amd/` model from Hugging Face and see a long hang, check driver/runtime
setup instead — see [docs/01-quick-start.md](01-quick-start.md) or [docs/05-linux-guide.md](05-linux-guide.md).

---

### FAQ-07: Error "Cannot find custom op: DynamicDispatch"

**Symptom:**
```
[ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for DynamicDispatch
```

**Fix:**
The ONNX custom op DLL is not loaded. Try:
1. Specify the path explicitly:
   ```bash
   python test/run.py --custom_op_path lib/onnxruntime_providers_ryzenai.dll ...
   ```
2. Run from the repository root so the default path resolution works.
3. Verify the DLL exists:
   ```powershell
   Test-Path lib/onnxruntime_providers_ryzenai.dll
   ```

---

### FAQ-08: Generated image is black or entirely grey

**Symptom:** Inference completes without errors but the output image is all black or solid grey.

**Possible causes and fixes:**

1. **Missing `config.json` in `dd/` directory.** After partition, copy the config file:
   ```bash
   cp models/sd3/normal/transformer/config.json models/sd3/normal/transformer/dd/
   ```

2. **VAE decoder not partitioned** — the run script fell back to CPU VAE, which may behave
   differently. Verify `dd/replaced.onnx` exists in the VAE decoder directory.

3. **Wrong `--num_inference_steps`** — for distilled models (SD-turbo, SDXL-turbo, FLUX),
   use the model's default step count (1 for turbo, 4 for FLUX). Do not set a high step count.

---

### FAQ-09: "MemoryError" or process crashes during inference

**Symptom:**
```
MemoryError
```
or the Python process exits unexpectedly.

**Fix:**
1. Close other memory-intensive applications.
2. Reduce resolution: start with `--width 512 --height 512` and increase gradually.
3. For SDXL and larger models, ensure at least 16 GB of system RAM is available.
4. Set the NPU to performance mode before inference:
   ```bash
   xrt-smi configure --pmode performance
   ```

---

### FAQ-10: Slow inference — much slower than expected for the model

**Symptom:** SD1.5 takes 3+ minutes per image, SDXL takes 10+ minutes.

**Fix:**
1. Check the NPU is being used (not CPU fallback):
   ```bash
   # Look for DmlExecutionProvider or NPU provider in the logs
   python test/run.py --model_id ... --verbose
   ```
2. Set NPU to performance mode:
   ```bash
   xrt-smi configure --pmode performance
   ```
3. For **locally built DD models**, confirm `--enable_compile` was used on the first run.
   Pre-partitioned `amd/` models from Hugging Face do not need this flag.
4. Check that `dd/replaced.onnx` exists. If the runner falls back to the plain `model.onnx`,
   no NPU acceleration is used.

---

### FAQ-11: ControlNet image looks wrong or has no effect

**Symptom:** The control image (edges, pose) does not influence the generated image.

**Fix:**
1. Verify `--controlnet_conditioning_scale` is appropriate for your model:
   - SD1.5 ControlNet: `1.5` (default)
   - SD3 ControlNet: `0.5` (default)
   - Too low = no effect; too high = artifacts
2. For SD3 ControlNet, the control image should be the same resolution as the output:
   ```bash
   python test/run.py -C Canny -H 1024 -W 1024 --control_image_path ref/1024x1024_canny.png
   ```
3. Ensure the `controlnet-canny/dd/replaced.onnx` file exists for the model you are using.

---

## OutPainting / InPainting Issues

### FAQ-12: OutPainting or InPainting does not respect the mask

**Symptom:** InPainting fills the wrong region, or OutPainting extends in the wrong direction.

**Fix:**
- For **OutPainting**, `--image_pads` controls the border size in pixels `[top, bottom, left, right]`.
  Start with `[100, 100, 0, 0]` to extend the top and bottom.
- For **InPainting/Removal**, the mask must be a binary image:
  - **White pixels** = region to modify
  - **Black pixels** = region to preserve
- Verify mask and image have the same resolution before running.

---

## Dynamic Resolution Issues

### FAQ-13: Dynamic shape inference crashes with "shape mismatch"

**Symptom:**
```
[ONNXRuntimeError] : Invalid argument: Shape mismatch: ...
```

**Fix:**
1. Verify `DD_PLUGINS_ROOT` is set:
   ```bash
   set DD_PLUGINS_ROOT=.\lib\transaction\stx
   ```
2. Confirm the resolution you are requesting was included in the `dynamic_shape_list` during
   partitioning. See [docs/04-model-generate-guide.md#step-6-dynamic-shape-configuration](04-model-generate-guide.md#step-6-dynamic-shape-configuration).
3. Use the `-ds` flag and specify height/width explicitly:
   ```bash
   python test/run.py -C None --model_id ... -ds -H 768 -W 1024
   ```

---

### FAQ-14: Error "could not find XRT" on Linux

Linux-only issue. See [docs/05-linux-guide.md#troubleshooting](05-linux-guide.md#troubleshooting).

---

### FAQ-15: "OSError: [WinError 127] The specified procedure could not be found" on Windows

**Symptom:**
```
OSError: [WinError 127] The specified procedure could not be found
```

This error appears when loading the ONNX custom ops DLL on Windows.

**Fix:**
1. Reinstall the Visual C++ Redistributable (x64): download from
   [Microsoft's site](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
2. Verify your Python version matches the wheel you installed (e.g., `cp312` requires Python 3.12).
3. Run from an administrator PowerShell session if DLL loading requires elevated permissions.

---

## Still stuck?

1. Run with `--verbose` to see the full execution log.
2. Check the `results/` directory for pipeline run summaries when using `scripts/run_pipelines.py`.
3. See [docs/04-model-generate-guide.md#troubleshooting](04-model-generate-guide.md#troubleshooting) for DD
   model creation issues.
4. See the [Ryzen AI documentation](https://ryzenai.docs.amd.com/en/latest/) for hardware and
   driver-level troubleshooting.
