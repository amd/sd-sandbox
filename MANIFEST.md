# Software Manifest

Application: **sd-sandbox**: Stable Diffusion ONNX pipelines for AMD Ryzen™ AI
Version: `main` (no tagged releases yet)
License: MIT for code written by Advanced Micro Devices, Inc. Third-party components are listed below with their own licenses. See [LICENSE](LICENSE).

Every source file carries an SPDX license identifier or the full license header of its upstream project.

## 1. Third-party source code

These files are derived from or taken from external projects. They keep the original copyright notice and license header. Where AMD changed a file, a `Modifications Copyright Advanced Micro Devices, Inc.` line was added and the upstream notice was left in place.

| File | Origin | Upstream copyright holder | License | AMD modified |
|---|---|---|---|---|
| `src/pipeline/pipeline_stable_diffusion_onnx_amd.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | The HuggingFace Team | Apache-2.0 | Yes |
| `src/pipeline/pipeline_stable_diffusion_xl_onnx_amd.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | The HuggingFace Team | Apache-2.0 | Yes |
| `src/pipeline/pipeline_controlnet_onnx_amd.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | The HuggingFace Team | Apache-2.0 | Yes |
| `src/pipeline/pipeline_stable_diffusion_Kolors_onnx_amd.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | The HuggingFace Team | Apache-2.0 | Yes |
| `src/pipeline/pipeline_stable_diffusion_3_controlnet_onnx_amd.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | Stability AI, The HuggingFace Team and The InstantX Team | Apache-2.0 | Yes |
| `src/pipeline/pipeline_stable_diffusion_3_controlnet_outpainting_onnx_amd.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | Stability AI, The HuggingFace Team and The AlimamaCreative Team | Apache-2.0 | Yes |
| `src/pipeline/pipeline_flux_onnx_amd.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | Black Forest Labs and The HuggingFace Team | Apache-2.0 | Yes |
| `src/pipeline/pipeline_stable_diffusion_flux2_klein_onnx_amd.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | Black Forest Labs and The HuggingFace Team | Apache-2.0 | Yes |
| `src/pipeline/pipeline_stable_diffusion_NitroE_onnx_amd.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) `StableDiffusion3Pipeline` (v0.31.0) | Stability AI and The HuggingFace Team | Apache-2.0 | Yes |
| `src/flux2/image_processor.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers) | The Black Forest Labs Team and The HuggingFace Team | Apache-2.0 | Yes |
| `src/t5/modeling_t5.py` | [huggingface/transformers](https://github.com/huggingface/transformers) (`models/t5/modeling_t5.py`) | Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team | Apache-2.0 | Yes |
| `src/kolors/configuration_chatglm.py` | [Kwai-Kolors/Kolors @ 038818d](https://github.com/Kwai-Kolors/Kolors/blob/038818d244ed103056abd10f429729a26af4d239/kolors/models/configuration_chatglm.py) | Kwai-Kolors (no holder named upstream) | Apache-2.0 | No |
| `src/kolors/modeling_chatglm.py` | [Kwai-Kolors/Kolors @ 038818d](https://github.com/Kwai-Kolors/Kolors/blob/038818d244ed103056abd10f429729a26af4d239/kolors/models/modeling_chatglm.py) | Kwai-Kolors (no holder named upstream) | Apache-2.0 | No |
| `src/kolors/tokenization_chatglm.py` | [Kwai-Kolors/Kolors @ 038818d](https://github.com/Kwai-Kolors/Kolors/blob/038818d244ed103056abd10f429729a26af4d239/kolors/models/tokenization_chatglm.py) | Kwai-Kolors (no holder named upstream) | Apache-2.0 | Yes (`_pad()` `padding_side` argument) |

Note on the Nitro-E pipeline: the ONNX pipeline is adapted from the diffusers `StableDiffusion3Pipeline` (`check_inputs`, `encode_prompt`, and `prepare_latents` follow it), not from the [AMD-AGI/Nitro-E](https://github.com/AMD-AGI/Nitro-E) training repository (MIT). Nitro-E's own diffusers pipeline (`core/models/emmdit_pipeline.py`) is also Apache-2.0, for the same reason.

All other source files (`scripts/`, `src/runner/`, `src/trigger/`, `src/utils/`, `test/`, and the remaining files under `src/`) were written by Advanced Micro Devices, Inc. and are MIT-licensed (`SPDX-License-Identifier: MIT`).

## 2. Binaries

None. The repository contains no compiled binaries, libraries, or model weights. Models are downloaded at run time from Hugging Face or supplied by the user.

## 3. Dependencies

Dependencies are installed by the user through conda and pip. They are not redistributed in this repository. The authoritative, pinned lists are in the environment files:

| Environment file | Python | Platform |
|---|---|---|
| [`env.yaml`](env.yaml) | 3.10 | Windows |
| [`env_py312.yaml`](env_py312.yaml) | 3.12 | Windows |
| [`env_py312_vaip.yaml`](env_py312_vaip.yaml) | 3.12 | Windows (Vitis AI EP) |
| [`env_py313.yaml`](env_py313.yaml) | 3.13 | Windows |
| [`env_linux_py312.yaml`](env_linux_py312.yaml) | 3.12 | Linux |

Primary direct dependencies:

| Package | Version(s) pinned | License |
|---|---|---|
| diffusers | 0.31.0, 0.35.1 | Apache-2.0 |
| transformers | 4.42.4, 4.56.2, 4.57.0 | Apache-2.0 |
| torch | 2.4.1, 2.8.0 | BSD-3-Clause |
| onnxruntime / onnxruntime-directml | 1.21.0, 1.24.3 | MIT |
| onnxruntime-extensions | 0.12.0 | MIT |
| onnx | 1.16.1, 1.21.0 | Apache-2.0 |
| onnx-tool | 0.9.0 | MIT |
| optimum | 1.24.0, 2.0.0 | Apache-2.0 |
| accelerate | 0.32.0, 1.10.1 | Apache-2.0 |
| huggingface-hub | 0.23.4, 0.35.3, 0.36.0 | Apache-2.0 |
| safetensors | 0.4.3, 0.6.2 | Apache-2.0 |
| sentencepiece | 0.2.0, 0.2.1 | Apache-2.0 |
| numpy | 1.26.4, 2.3.3 | BSD-3-Clause |
| pillow | 10.4.0, 11.3.0 | MIT-CMU (HPND) |
| opencv-python | 4.11.0.86 | Apache-2.0 (binary wheels bundle additional components) |
| protobuf | 6.31.1, 6.32.1 | BSD-3-Clause |

TODO: a complete license inventory of every transitive package in the environment files has not been produced yet.

## 4. Media assets

Images used as test inputs and reference outputs. Their origin and license have not been confirmed yet. Images generated with the pipelines in this repository are marked "generated" once confirmed.

| File | Origin | License |
|---|---|---|
| `test/assets/2025_Bousetouane_image_to_image_controlnet.png` | TBD | TBD |
| `test/assets/2025_Bousetouane_text-to-image.png` | TBD | TBD |
| `test/assets/canny.jpg` | TBD | TBD |
| `test/assets/coffe_table.png` | TBD | TBD |
| `test/assets/control.png` | TBD | TBD |
| `test/assets/controlimg_input_1024x1024.png` | TBD | TBD |
| `test/assets/controlimg_output_strength_0.95_1024x1024.png` | TBD | TBD |
| `test/assets/depth.jpeg` | TBD | TBD |
| `test/assets/flux2_img1.png` | TBD | TBD |
| `test/assets/flux2_img3.png` | TBD | TBD |
| `test/assets/inpainting_input_1024x1024.png` | TBD | TBD |
| `test/assets/inpainting_mask_1024x1024.png` | TBD | TBD |
| `test/assets/inpainting_output_1024x1024.png` | TBD | TBD |
| `test/assets/living_room_mask_books_vase_1024x1024.png` | TBD | TBD |
| `test/assets/outpainting_input_512x512.png` | TBD | TBD |
| `test/assets/outpainting_output_1024x1024.png` | TBD | TBD |
| `test/assets/pose.jpg` | TBD | TBD |
| `test/assets/precise_mask.png` | TBD | TBD |
| `test/assets/removal_input_1024x1024.jpg` | TBD | TBD |
| `test/assets/removal_input_mask_1024x1024.jpg` | TBD | TBD |
| `test/assets/removal_output_1024x1024.png` | TBD | TBD |
| `test/assets/sd_demo.gif` | TBD | TBD |
| `test/assets/tile.jpg` | TBD | TBD |
| `test/ref/Canny_dog_condition.jpg` | TBD | TBD |
| `test/ref/canny.jpg` | TBD | TBD |
| `test/ref/control.png` | TBD | TBD |
| `test/ref/depth.jpeg` | TBD | TBD |
| `test/ref/inpainting/kolors_mask.png` | TBD | TBD |
| `test/ref/inpainting/kolors_origin.png` | TBD | TBD |
| `test/ref/inpainting/mask.jpg` | TBD | TBD |
| `test/ref/inpainting/origin.jpg` | TBD | TBD |
| `test/ref/outpainting.jpg` | TBD | TBD |
| `test/ref/pose.jpg` | TBD | TBD |
| `test/ref/removal/mask.jpg` | TBD | TBD |
| `test/ref/removal/origin.jpg` | TBD | TBD |
| `test/ref/tile.jpg` | TBD | TBD |
| `test/ref/union.jfif` | TBD | TBD |

## 5. Internal-only files

None. Every file in this repository is intended for public release.
