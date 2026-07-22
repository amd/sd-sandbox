# Model Reference

This document is the single reference for **running inference** on all supported models:
capabilities, model directory layouts, run commands, and CLI options.

> **Before running any model**, complete environment setup in [docs/01-quick-start.md](01-quick-start.md)
> (Linux: [docs/05-linux-guide.md](05-linux-guide.md)).
>
> **`--enable_compile`:** Not required for Hugging Face `amd/` models — they ship pre-partitioned
> and ready to run. Use `--enable_compile` (`-c`) only on the **first inference run with a locally
> built DD model** (see [docs/04-model-generate-guide.md](04-model-generate-guide.md)).

---

## Table of Contents

- [Supported Models Summary](#supported-models-summary)
- [Common Arguments](#common-arguments)
- [SD 1.5 Family](#sd-15-family)
  - [SD 1.5 Text-to-Image](#sd-15-text-to-image)
  - [SD-Turbo Text-to-Image](#sd-turbo-text-to-image)
  - [SD 2.1 Text-to-Image](#sd-21-text-to-image)
  - [SD 1.5 ControlNet (Canny)](#sd-15-controlnet-canny)
- [SDXL Family](#sdxl-family)
  - [SDXL-base Text-to-Image](#sdxl-base-text-to-image)
  - [SDXL-turbo Text-to-Image](#sdxl-turbo-text-to-image)
  - [Playground 2.5 Text-to-Image](#playground-25-text-to-image)
  - [SSD-1B Text-to-Image](#ssd-1b-text-to-image)
  - [DreamShaper XL Text-to-Image](#dreamshaper-xl-text-to-image)
  - [Segmind-Vega Text-to-Image](#segmind-vega-text-to-image)
  - [Segmind-Vega Image-to-Image](#segmind-vega-image-to-image)
- [SD3 / SD3.5 Family](#sd3--sd35-family)
  - [SD3 Text-to-Image](#sd3-text-to-image)
  - [SD3.5 Text-to-Image](#sd35-text-to-image)
  - [SD3 ControlNet (Canny/Tile/Pose)](#sd3-controlnet-cannytilepose)
  - [SD3.5 ControlNet (Canny/Tile/Pose)](#sd35-controlnet-cannytilepose)
  - [SD3 OutPainting / Removal / InPainting](#sd3-outpainting--removal--inpainting)
  - [SD3.5 OutPainting / Removal / InPainting](#sd35-outpainting--removal--inpainting)
  - [SD3 Dynamic Resolution](#sd3-dynamic-resolution)
  - [SD3.5 Dynamic Resolution](#sd35-dynamic-resolution)
  - [SD3 ControlNet Dynamic Resolution](#sd3-controlnet-dynamic-resolution)
  - [SD3.5 ControlNet Dynamic Resolution](#sd35-controlnet-dynamic-resolution)
- [FLUX.1-schnell](#flux1-schnell)
- [FLUX.2 Klein](#flux2-klein)
- [O1 Optimization Preset](#o1-optimization-preset)

---

## Supported Models Summary

| Model | Hugging Face ID | Resolution | T2I | I2I | ControlNet | OutPainting |
|-------|----------------|-----------|-----|-----|------------|-------------|
| SD 1.5 | `amd/stable-diffusion-1.5-amdnpu` | 512×512 | ✓ | | | |
| SD Turbo | `amd/sd-turbo-amdnpu` | 512×512 | ✓ | | | |
| SD 2.1 *(deprecated)* | — | 512/768 | ✓ | | | |
| SD 1.5 ControlNet | `amd/sd15-controlnet-amdnpu` | 512×512 | ✓ | | Canny | |
| SDXL-base | `amd/sdxl-base-amdnpu` | 1024×1024 | ✓ | | | |
| SDXL-turbo | `amd/sdxl-turbo-amdnpu` | 512×512 | ✓ | | | |
| Playground 2.5 | `amd/playground-v2.5-1024px-aesthetic-amdnpu` | 1024×1024 | ✓ | | | |
| SSD-1B | `amd/SSD-1B-amdnpu` | 1024×1024 | ✓ | | | |
| DreamShaper XL | `amd/dreamshaper-xl-lightning-amdnpu` | 1024×1024 | ✓ | | | |
| Segmind-Vega | `amd/segmind-vega-amdnpu` | 1024×1024 | ✓ | ✓ | | |
| SD3 Medium | `amd/stable-diffusion-3-medium-amdnpu` | 512–1024 | ✓ | | Canny/Tile/Pose/Depth/Union | ✓ |
| SD3.5 Medium | `amd/stable-diffusion-3.5-medium-amdnpu` | 512–1024 | ✓ | | Canny/Tile/Pose | ✓ |
| FLUX.1-schnell | `amd/FLUX.1-schnell-amdnpu` | 288–1024 | ✓ | | | |
| FLUX.2 Klein | `amd/FLUX.2-klein-4B-amdnpu` | 1024×1024 | ✓ | ✓ | | |

*Dynamic resolution models support: 512×512, 512×768, 768×512, 576×1024, 1024×576, 768×1024, 1024×768, 1024×1024*

All `amd/` models are downloaded automatically from Hugging Face on first use when `--model_path`
is omitted. **Kolors** requires a locally built DD model — see the [Kolors](#kolors) section.

---

## Common Arguments

These arguments apply to all run scripts:

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model_id` | | required | Hugging Face model ID or model type identifier |
| `--model_path` | | auto-download | Path to local model directory (auto-downloaded if omitted; see `RYZENAI_GENAI_SD_MODELS_ROOT`) |
| `--revision` | | None | Hugging Face branch, tag, or commit hash (e.g. `"1.8.0"`) |
| `--prompt` | | model default | Text description for image generation |
| `--prompt_2` | | None | Second prompt for dual-encoder models (e.g. FLUX T5 encoder) |
| `--n_prompt` | | `""` | Negative prompt (what to avoid) |
| `--width` | `-W` | model default | Output image width in pixels |
| `--height` | `-H` | model default | Output image height in pixels |
| `--num_inference_steps` | `-n` | model default | Number of denoising steps |
| `--guidance_scale` | | model default | CFG scale (higher = more prompt adherence) |
| `--seed` | | None | Random seed for reproducibility |
| `--t5_sequence_len` | | model default | T5 sequence length (e.g. 77, 83, or 256 for Flux) |
| `--edit_image_path` | | None | Reference image path(s) for FLUX.2 Klein image editing (accepts one or more paths) |
| `--output_path` | | `generated_images` | Output directory for saved images |
| `--enable_compile` | `-c` | False | Compile DD fusion runtime on first run with a **locally built** DD model (not needed for HF `amd/` models) |
| `--enable_profile` | `-p` | False | Enable performance profiling |
| `--profiling_rounds` | `-pr` | 4 | Number of profiling rounds |
| `--no_excel` | | False | Skip Excel metrics file; timing still printed to stdout |
| `--no_images` | | False | Skip image saving (pipeline still runs; useful for pure benchmarking) |
| `--custom_op_path` | | auto | `onnxruntime_providers_ryzenai.dll` / `libonnxruntime_providers_ryzenai.so` (auto-detected if omitted) |
| `--verbose` | `-v` | False | Enable verbose output |
| `--prompt_file_path` | | None | Path to JSON prompt list file |
| `--num_images_per_prompt` | | 1 | Number of images to generate per prompt |
| `-C` / `--controlnet` | `-C` | None | ControlNet mode: `Canny`, `Tile`, `Pose`, `Depth`, `Union`, `OutPainting`, `Removal`, `InPainting`, or `None` |

### Model Download Path

Models are cached under `models/` in the project root by default. Override with `RYZENAI_GENAI_SD_MODELS_ROOT`:

```powershell
# Windows
$env:RYZENAI_GENAI_SD_MODELS_ROOT = "D:\my_models"
python test/run.py --model_id "amd/sdxl-turbo-amdnpu" --prompt "..."
```

```bash
# Linux
export RYZENAI_GENAI_SD_MODELS_ROOT=/data/my_models
python test/run.py --model_id "amd/sdxl-turbo-amdnpu" --prompt "..."
```

### Environment Variables (inference)

| Variable | Windows | Linux |
|----------|---------|-------|
| `RYZENAI_GENAI_SD_MODELS_ROOT` | Model download/cache directory | Same |
| `DD_PLUGINS_ROOT` | `set DD_PLUGINS_ROOT=.\lib\transaction\stx` | `export DD_PLUGINS_ROOT=./lib/transaction/stx` |
| Session setup (XRT, `LD_LIBRARY_PATH`) | See [01-quick-start.md](01-quick-start.md) | See [05-linux-guide.md](05-linux-guide.md) Step 6 |

`DD_PLUGINS_ROOT` is required for SD3 / SD3.5 / FLUX dynamic-shape models.

---

## SD 1.5 Family

### Model Directory Layout

```
sd15/
├─ vae_decoder/
├─ unet/
├─ scheduler/
└─ tokenizer/
```

### SD 1.5 Text-to-Image

```bash
# Auto-download from Hugging Face (recommended)
python test/run.py --model_id "amd/stable-diffusion-1.5-amdnpu" --prompt "a beautiful landscape"

# Local model path
python test/run.py --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --model_path path/to/sd15/models/ --prompt "a beautiful landscape"
```

**Defaults:** 512×512, 20 steps, guidance_scale=7.5

### SD-Turbo Text-to-Image

```bash
# Batch size 1 (default)
python test/run.py --model_id "amd/sd-turbo-amdnpu" --prompt "a beautiful landscape"

# Batch size 2 (generates 2 images per prompt)
python test/run.py --model_id "amd/sd-turbo-amdnpu" --prompt "a beautiful landscape" \
    --num_images_per_prompt 2
```

**Defaults:** 512×512, 1 step, guidance_scale=0.0 (distilled)

### SD 2.1 Text-to-Image

```bash
# SD 2.1-base (512×512)
python test/run.py --model_id "stabilityai/stable-diffusion-2-1-base" \
    --model_path path/to/sd2.1-base/models/

# SD 2.1 (768×768)
python test/run.py --model_id "stabilityai/stable-diffusion-2-1" \
    --model_path path/to/sd2.1/models/
```

**Defaults:** sd2.1-base: 512×512, 50 steps; sd2.1: 768×768, 50 steps

### SD 1.5 ControlNet (Canny)

Generates images guided by an edge-detected control image.

**Model directory:**
```
sd15-controlnet/
├─ controlnet/
├─ vae_decoder/
├─ unet/
├─ scheduler/
└─ tokenizer/
```

```bash
python test/run.py \
    --model_id "amd/sd15-controlnet-amdnpu" \
    --prompt "a beautiful landscape" \
    --control_image_path path/to/canny/image.png
```

**Additional ControlNet arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--control_image_path` | required | Path to the edge/control image |
| `--controlnet_conditioning_scale` | 1.5 | Strength of ControlNet guidance |

**Defaults:** 512×512, 20 steps, guidance_scale=7.5

---

## SDXL Family

### Model Directory Layout

```
sdxl-turbo/
├─ vae_decoder/
├─ vae_encoder/
├─ unet/
├─ scheduler/
├─ tokenizer/
└─ tokenizer_2/
```

### SDXL-base Text-to-Image

```bash
python test/run.py --model_id "amd/sdxl-base-amdnpu" --prompt "a beautiful landscape"

# Dynamic resolution (20 presets, 512×288 – 1024×1024)
python test/run.py --model_id "amd/sdxl-base-amdnpu" --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v
```

**Defaults:** 1024×1024, 50 steps, guidance_scale=7.5

### SDXL-turbo Text-to-Image

```bash
python test/run.py --model_id "amd/sdxl-turbo-amdnpu" --prompt "a beautiful landscape"

# Dynamic resolution (20 presets, 512×288 – 1024×1024)
python test/run.py --model_id "amd/sdxl-turbo-amdnpu" --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v
```

**Defaults:** 512×512, 1 step (Runner default; orchestrator `sdxl_turbo` pipeline uses 4 steps), guidance_scale=0.0

### Playground 2.5 Text-to-Image

```bash
python test/run.py --model_id "amd/playground-v2.5-1024px-aesthetic-amdnpu" \
    --prompt "a beautiful landscape"

# Dynamic resolution (20 presets, 512×288 – 1024×1024)
python test/run.py --model_id "amd/playground-v2.5-1024px-aesthetic-amdnpu" \
    --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v
```

**Defaults:** 1024×1024, 50 steps, guidance_scale=3.0

### SSD-1B Text-to-Image

SSD-1B is a compact SDXL variant with faster inference while maintaining 1024×1024 output quality.

```bash
python test/run.py --model_id "amd/SSD-1B-amdnpu" --prompt "a beautiful landscape"
```

**Defaults:** 1024×1024, 50 steps, guidance_scale=7.5

### DreamShaper XL Text-to-Image

DreamShaper XL Lightning is a distilled SDXL model that generates high-quality images in very
few steps.

```bash
python test/run.py --model_id "amd/dreamshaper-xl-lightning-amdnpu" \
    --prompt "a beautiful landscape"

# Dynamic resolution (20 presets, 512×288 – 1024×1024)
python test/run.py --model_id "amd/dreamshaper-xl-lightning-amdnpu" \
    --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v
```

**Defaults:** 1024×1024, 6 steps (Lightning distilled; increase for higher quality), guidance_scale=7.5

### Segmind-Vega Text-to-Image

```bash
python test/run.py --model_id "amd/segmind-vega-amdnpu" --prompt "a beautiful landscape"
```

**Defaults:** 1024×1024, 50 steps, guidance_scale=7.5

### Segmind-Vega Image-to-Image

Transforms an existing image based on the prompt. The `--strength` parameter controls how much
the original image is preserved vs. replaced.

```bash
python test/run.py --model_id "amd/segmind-vega-amdnpu" \
    --prompt "a beautiful landscape" \
    --control_image_path test/assets/controlimg_input_1024x1024.png \
    --strength 0.95
```

**`--strength` reference:**
- `0.3` (default) — 15 denoising steps; original image mostly preserved
- `0.7` — 35 steps; balanced mix of prompt and original
- `1.0` — 50 steps; original image mostly replaced by prompt

---

## SD3 / SD3.5 Family

SD3 and SD3.5 use the MM-DiT (Multimodal Diffusion Transformer) architecture with three text
encoders (2× CLIP + T5) and a 16-channel VAE.

### Model Directory Layout

```
sd3/
├─ common/
│  ├─ scheduler/
│  ├─ tokenizer/
│  ├─ tokenizer_2/
│  ├─ tokenizer_3/
│  ├─ text_encoder/
│  ├─ text_encoder_2/
│  ├─ text_encoder_3_gptq_v2/
│  ├─ vae_decoder/
│  └─ vae_encoder/
└─ normal/
   ├─ transformer/
   ├─ controlnet-canny/   (ControlNet only)
   ├─ controlnet-pose/    (ControlNet only)
   └─ controlnet-tile/    (ControlNet only)
```

### SD3 Text-to-Image

```bash
# 512×512
python test/run.py -C None --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape"

# 1024×1024
python test/run.py -C None --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" -H 1024 -W 1024

# With profiling
python test/run.py -C None --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" -p
```

**Defaults:** 512×512, 8 steps, guidance_scale=3.0

### SD3.5 Text-to-Image

> **Note:** The SD3.5 Hugging Face release (`amd/stable-diffusion-3.5-medium-amdnpu`) includes
> **Text-to-Image (T2I) weights only**. For ControlNet and painting modes, copy the relevant
> controlnet directories from your SD3 model folder — see the
> [SD3.5 ControlNet](#sd35-controlnet-cannytilepose) and
> [SD3.5 OutPainting / Removal / InPainting](#sd35-outpainting--removal--inpainting) sections.

```bash
python test/run.py -C None --model_id "amd/stable-diffusion-3.5-medium-amdnpu" \
    --prompt "a beautiful landscape"
```

**Defaults:** 512×512, 8 steps, guidance_scale=4.5

### SD3 ControlNet (Canny/Tile/Pose)

```bash
# Canny
python test/run.py -C Canny --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" --control_image_path path/to/image.png

# Tile
python test/run.py -C Tile --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" --control_image_path path/to/image.png

# Pose
python test/run.py -C Pose --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" --control_image_path path/to/image.png
```

**ControlNet arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `-C` | required | ControlNet type: `Canny`, `Tile`, `Pose`, or `None` |
| `--control_image_path` | required | Path to control/reference image |
| `--controlnet_conditioning_scale` | 0.5 | ControlNet guidance strength |

### SD3.5 ControlNet (Canny/Tile/Pose)

> **Note:** The SD3.5 Hugging Face release does not include separate ControlNet weights. These
> modes reuse the SD3 controlnet weights — copy the relevant directories before running:
> ```bash
> cp -r models/sd3/normal/controlnet-canny  models/sd3.5/normal/
> cp -r models/sd3/normal/controlnet-tile   models/sd3.5/normal/
> cp -r models/sd3/normal/controlnet-pose   models/sd3.5/normal/
> ```

Same commands as SD3 ControlNet, using `stable-diffusion-3.5-medium-amdnpu`:

```bash
python test/run.py -C Canny --model_id "amd/stable-diffusion-3.5-medium-amdnpu" \
    --prompt "a beautiful landscape" --control_image_path path/to/image.png
```

### SD3 OutPainting / Removal / InPainting

> **Availability:** Only the **InPainting** controlnet weights are published on Hugging Face
> (`amd/stable-diffusion-3-medium-amdnpu`). OutPainting and Removal weights are not currently
> available publicly.

**Additional model directories:**
```
sd3/
└─ normal/
   ├─ controlnet-outpainting/    # weights not on Hugging Face
   ├─ controlnet-removal/        # weights not on Hugging Face
   └─ transformer/
sd3/inpainting/
   ├─ controlnet-inpainting/
   └─ transformer/
```

```bash
# OutPainting — extend image borders (requires local weights)
python test/run.py -C OutPainting \
    --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" \
    --control_image_path path/to/image.png \
    --image_pads 100 100 0 0    # [top, bottom, left, right] pixels

# Removal — remove objects from image (requires local weights)
python test/run.py -C Removal \
    --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "" \
    --control_image_path path/to/image.png \
    --control_mask_path path/to/mask.png

# InPainting — fill masked region guided by prompt (available on Hugging Face)
python test/run.py -C InPainting \
    --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "holiday decorations" \
    --control_image_path path/to/image.png \
    --control_mask_path path/to/mask.png
```

**OutPainting/Removal/InPainting arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `-C` | `OutPainting` | Mode: `OutPainting`, `Removal`, or `InPainting` |
| `--control_image_path` | required | Input image |
| `--control_mask_path` | required for Removal/InPainting | Binary mask (white = region to modify) |
| `--image_pads` | `[10,10,10,10]` | OutPainting border [top, bottom, left, right] in pixels |
| `--controlnet_conditioning_scale` | 0.95 (OutPainting/Removal), 0.7 (InPainting) | ControlNet strength |

**Defaults:** 1024×1024, 8 steps, guidance_scale=5.0

### SD3.5 OutPainting / Removal / InPainting

> **Note:** The SD3.5 Hugging Face release does not include separate ControlNet weights for these
> modes. These modes reuse the SD3 controlnet weights — copy the relevant directories from your
> SD3 model folder into the SD3.5 model directory:
> ```bash
> cp -r models/sd3/normal/controlnet-inpainting  models/sd3.5/normal/
> cp -r models/sd3/inpainting/                   models/sd3.5/
> ```

Same commands as SD3, using `stable-diffusion-3.5-medium-amdnpu`:

```bash
python test/run.py -C OutPainting \
    --model_id "amd/stable-diffusion-3.5-medium-amdnpu" \
    --prompt "a beautiful landscape" \
    --control_image_path path/to/image.png
```

### SD3 Dynamic Resolution

Dynamic resolution allows a single loaded model to generate images at multiple resolutions without
reloading. Supported resolutions: 512×512, 512×768, 768×512, 576×1024, 1024×576, 768×1024, 1024×768, 1024×1024.

**Dynamic model directory layout** (note `/dynamic/` subdirectory):
```
sd3/
├─ common/
│  ├─ vae_decoder/dynamic/
│  └─ vae_encoder/dynamic/
└─ normal/
   └─ transformer/dynamic/
```

```bash
# Text-to-image with dynamic resolution (20 resolutions)
python test/run.py -C None --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v

# Specify resolution explicitly
python test/run.py -C None --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v -H 768 -W 1024

# With profiling
python test/run.py -C None --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v -p
```

**Dynamic shape arguments:**

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--dynamic_shape` | `-ds` | False | Enable dynamic resolution |
| `--dynamic_shape_file_path` | | `config/dynamic_shape_20x.json` | Custom dynamic shape config (20 presets by default) |

### SD3.5 Dynamic Resolution

```bash
python test/run.py -C None --model_id "amd/stable-diffusion-3.5-medium-amdnpu" \
    --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v
```

> **Note for SD3.5 with LoRA:** Set `--guidance_scale 0.0` explicitly when using LoRA weights.

### SD3 ControlNet Dynamic Resolution

```bash
# SD3 ControlNet Canny with dynamic shapes
python test/run.py -C Canny --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" --control_image_path path/to/image.png \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v

# SD3 ControlNet Union — supports Canny, Tile, Pose, Depth, Scribble
python test/run.py -C Union --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v
```

**Union ControlNet recommended settings by task:**

| Task | Steps | Control Scale | CFG |
|------|-------|--------------|-----|
| Canny | 20 | 0.7 | 3.0 |
| Pose | 20 | 0.7 | 3.5 |
| Tile | 20 | 1.0 | 3.5 |
| Depth | 20 | 0.7 | 3.5 |
| Scribble | 20 | 0.7 | 3.5 |

```bash
# Union Canny example
python test/run.py -C Union --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v \
    --num_inference_steps 20 --controlnet_conditioning_scale 0.7 --guidance_scale 3.0 \
    --control_image_path ref/canny.jpg
```

### SD3.5 ControlNet Dynamic Resolution

```bash
python test/run.py -C Canny --model_id "amd/stable-diffusion-3.5-medium-amdnpu" \
    --prompt "a beautiful landscape" --control_image_path path/to/image.png \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v
```

---

## FLUX.1-schnell

FLUX.1-schnell is a distilled diffusion transformer with dual text encoders (CLIP on iGPU + T5 on NPU)
and a 16-channel VAE. It uses BFP16 quantization on the NPU for all major components.

### Model Directory Layout

```
flux1-schnell/
├─ scheduler/
├─ tokenizer/                   # CLIP tokenizer
├─ tokenizer_2/                 # T5 tokenizer
├─ text_encoder/                # CLIP encoder (runs on iGPU via DirectML)
├─ text_encoder_3_gptq_v2/      # T5 encoder (runs on NPU)
├─ vae_decoder/dynamic/
└─ transformer/dynamic/
```

### Running FLUX.1-schnell

```bash
# Run with dynamic resolution
python test/run.py --model_id "amd/FLUX.1-schnell-amdnpu" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v

# With custom resolution
python test/run.py --model_id "amd/FLUX.1-schnell-amdnpu" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v \
    -H 768 -W 1024

# With profiling
python test/run.py --model_id "amd/FLUX.1-schnell-amdnpu" \
    -ds --dynamic_shape_file_path config/dynamic_shape_20x.json -v -p
```

**Defaults:** 4 steps (distilled), guidance_scale=0.0 (ignored by FLUX.1-schnell)

**FLUX-specific arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt_2` | same as `--prompt` | Second prompt fed to the T5 encoder |
| `--dynamic_shape` / `-ds` | False | Enable dynamic resolution |
| `--dynamic_shape_file_path` | auto | Path to dynamic shape config JSON |

**Supported dynamic resolutions for FLUX transformer:**

| Resolution | latent_dim |
|-----------|-----------|
| 288×512 / 512×288 | 576 |
| 384×512 / 512×384 | 768 |
| 512×512 | 1024 |
| 640×640 | 1600 |
| 768×768 | 2304 |
| 1024×1024 | 4096 |

> FLUX.1-schnell is optimized for 4 steps. Using fewer steps yields diminishing quality gains.
> If you built the DD model locally, include `--enable_compile` on the first inference run.

---

## FLUX.2 Klein

FLUX.2 Klein is a 4B-parameter diffusion transformer using the Qwen3 text encoder. It supports
both text-to-image generation and image editing via `--edit_image_path`.

### Model Directory Layout

```
flux2-klein/
├─ scheduler/
├─ tokenizer/
├─ text_encoder/                # Qwen3 encoder (runs on NPU)
├─ vae_decoder/
└─ transformer/
```

### Running FLUX.2 Klein

```bash
# Text-to-image (recommended)
python test/run.py --model_id "amd/FLUX.2-klein-4B-amdnpu" \
    --prompt "a beautiful landscape" -v

# With profiling
python test/run.py --model_id "amd/FLUX.2-klein-4B-amdnpu" \
    --prompt "a beautiful landscape" -v -p

# Image editing — guide generation using a reference image
python test/run.py --model_id "amd/FLUX.2-klein-4B-amdnpu" \
    --prompt "turn the horse into a unicorn" \
    --edit_image_path path/to/input.png -v
```

**Defaults:** 1024×1024, 4 steps, guidance_scale=1.0, seed=0

**FLUX.2 Klein–specific arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--edit_image_path` | None | Reference image for image editing (I2I). Can pass multiple paths. |

---

## O1 Optimization Preset

For **SD3** and **SD3.5** models, pass `-O1` (or `--optimize_o1`) to enable a performance
optimization preset that skips certain DiT steps during denoising.

```bash
python test/run.py -C None --model_id "amd/stable-diffusion-3-medium-amdnpu" \
    --prompt "a beautiful landscape" -O1
```

**When O1 is beneficial:** Use with `num_inference_steps >= 8`. O1 adds overhead that can
dominate at very low step counts.

**When O1 has no effect:** For text-to-image (no ControlNet) with `num_inference_steps <= 4`,
the runner automatically disables O1 to avoid degraded performance.

---

## Output Files

- **Images**: saved to `generated_images/` (or `--output_path`) with names like
  `stable-diffusion-v1-5_0_512x512_steps20_idx0_20260101_100000.png`
- **Profiling data**: `.xlsx` files in the same directory when `--enable_profile` is used
- **Result summaries**: `results/` directory when using `scripts/run_pipelines.py`

See [docs/02-user-guide.md](02-user-guide.md) for batch pipeline configuration and execution modes.
