# Software Manifest

Application: **sd-sandbox**: Stable Diffusion ONNX pipelines for AMD Ryzen™ AI
Version: 1.9.0 (for AMD Ryzen™ AI Software 1.9.0)
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
| `src/pipeline/base_amd_pipeline.py` | [huggingface/diffusers v0.35.1](https://github.com/huggingface/diffusers/blob/0f252be0ed42006c125ef4429156cb13ae6c1d60/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) (`prepare_latents`, `prepare_extra_step_kwargs`, `get_guidance_scale_embedding`) | The HuggingFace Team | Apache-2.0 | Yes (AMD mixin; three methods adapted from upstream) |
| `third_party/diffusers/flux2/image_processor.py` | [huggingface/diffusers @ 7448258](https://github.com/huggingface/diffusers/blob/7448258505c504d2070a48abd3ca56543324c016/src/diffusers/pipelines/flux2/image_processor.py) | The Black Forest Labs Team and The HuggingFace Team | Apache-2.0 | Yes (import paths; `@register_to_config` removed) |
| `third_party/diffusers/pipeline_utils.py` | [huggingface/diffusers v0.35.1](https://github.com/huggingface/diffusers/blob/0f252be0ed42006c125ef4429156cb13ae6c1d60/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) (`rescale_noise_cfg`, `retrieve_timesteps`) | The HuggingFace Team | Apache-2.0 | Reformatted only |
| `third_party/t5/modeling_t5.py` | [huggingface/transformers](https://github.com/huggingface/transformers) (`models/t5/modeling_t5.py`) | Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team | Apache-2.0 | Yes |
| `third_party/kolors/configuration_chatglm.py` | [Kwai-Kolors/Kolors @ 038818d](https://github.com/Kwai-Kolors/Kolors/blob/038818d244ed103056abd10f429729a26af4d239/kolors/models/configuration_chatglm.py) | Kwai-Kolors (no holder named upstream) | Apache-2.0 | No |
| `third_party/kolors/modeling_chatglm.py` | [Kwai-Kolors/Kolors @ 038818d](https://github.com/Kwai-Kolors/Kolors/blob/038818d244ed103056abd10f429729a26af4d239/kolors/models/modeling_chatglm.py) | Kwai-Kolors (no holder named upstream) | Apache-2.0 | No |
| `third_party/kolors/tokenization_chatglm.py` | [Kwai-Kolors/Kolors @ 038818d](https://github.com/Kwai-Kolors/Kolors/blob/038818d244ed103056abd10f429729a26af4d239/kolors/models/tokenization_chatglm.py) | Kwai-Kolors (no holder named upstream) | Apache-2.0 | Yes (`_pad()` `padding_side` argument) |

Placement: files that are upstream code with at most minor AMD edits live under [`third_party/`](third_party/). The `src/pipeline/pipeline_*_onnx_amd.py` pipelines and `src/pipeline/base_amd_pipeline.py` stay in `src/pipeline/`. They started from diffusers, but AMD rewrote large parts of each file to run the models through ONNX Runtime on the NPU, so the upstream code and the AMD code cannot be split into separate files. They keep the upstream copyright notice and Apache-2.0 license header, add a `Modifications Copyright Advanced Micro Devices, Inc.` line, and carry `SPDX-License-Identifier: Apache-2.0`.

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
| protobuf | 6.31.1, 6.33.5 | BSD-3-Clause |
| scipy | 1.15.1, 1.16.2 | BSD-3-Clause |

TODO: a complete license inventory of every transitive package in the environment files has not been produced yet.

## 4. Media assets

Images used as test inputs, reference outputs, and README diagrams, with the origin and license of each. Files under `test/ref/` that duplicate a file under `test/assets/` are listed separately because they are separate files in the repository.

### 4.1 Third-party images

These images are redistributed unmodified from public upstream sources. The copyright holder and license of each source are retained here.

| File | Origin | Copyright holder | License | AMD modified |
|---|---|---|---|---|
| `test/assets/canny.jpg` | [InstantX/SD3-Controlnet-Canny @ e42ecba](https://huggingface.co/InstantX/SD3-Controlnet-Canny/blob/e42ecba9741dee3de8c54039eba8a879c0070ced/canny.jpg) `canny.jpg` | The InstantX Team | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (see note 1) | No |
| `test/ref/canny.jpg` | [InstantX/SD3-Controlnet-Canny @ e42ecba](https://huggingface.co/InstantX/SD3-Controlnet-Canny/blob/e42ecba9741dee3de8c54039eba8a879c0070ced/canny.jpg) `canny.jpg` | The InstantX Team | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (see note 1) | No |
| `test/assets/control.png` | [lllyasviel/sd-controlnet-canny @ 7f2f691](https://huggingface.co/lllyasviel/sd-controlnet-canny/blob/7f2f69197050967007f6bbd23ab5e52f0384162a/images/bird_canny.png) `images/bird_canny.png` | Lvmin Zhang (lllyasviel) | [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) (see note 2) | No (same pixels, re-encoded) |
| `test/ref/control.png` | [lllyasviel/sd-controlnet-canny @ 7f2f691](https://huggingface.co/lllyasviel/sd-controlnet-canny/blob/7f2f69197050967007f6bbd23ab5e52f0384162a/images/bird_canny.png) `images/bird_canny.png` | Lvmin Zhang (lllyasviel) | [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) (see note 2) | No (same pixels, re-encoded) |
| `test/assets/depth.jpeg` | [InstantX/SD3-Controlnet-Depth @ 7765870](https://huggingface.co/InstantX/SD3-Controlnet-Depth/blob/7765870fa5dfbb7b5399112bae404270b736f61c/images/depth.jpeg) `images/depth.jpeg` | The InstantX Team | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | No |
| `test/ref/depth.jpeg` | [InstantX/SD3-Controlnet-Depth @ 7765870](https://huggingface.co/InstantX/SD3-Controlnet-Depth/blob/7765870fa5dfbb7b5399112bae404270b736f61c/images/depth.jpeg) `images/depth.jpeg` | The InstantX Team | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | No |
| `test/assets/pose.jpg` | [InstantX/SD3-Controlnet-Pose @ 1c550db](https://huggingface.co/InstantX/SD3-Controlnet-Pose/blob/1c550dbb8f7b35098ae7c91f498d1a753c9f49ea/pose.jpg) `pose.jpg` | The InstantX Team | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (see note 1) | No |
| `test/ref/pose.jpg` | [InstantX/SD3-Controlnet-Pose @ 1c550db](https://huggingface.co/InstantX/SD3-Controlnet-Pose/blob/1c550dbb8f7b35098ae7c91f498d1a753c9f49ea/pose.jpg) `pose.jpg` | The InstantX Team | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (see note 1) | No |
| `test/assets/tile.jpg` | [InstantX/SD3-Controlnet-Tile @ 48005f2](https://huggingface.co/InstantX/SD3-Controlnet-Tile/blob/48005f2448d5ea7aad3d383c2bb8b017723244d6/tile.jpg) `tile.jpg` | The InstantX Team | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (see note 1) | No |
| `test/ref/tile.jpg` | [InstantX/SD3-Controlnet-Tile @ 48005f2](https://huggingface.co/InstantX/SD3-Controlnet-Tile/blob/48005f2448d5ea7aad3d383c2bb8b017723244d6/tile.jpg) `tile.jpg` | The InstantX Team | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) (see note 1) | No |
| `test/ref/Canny_dog_condition.jpg` | [Kwai-Kolors/Kolors @ 038818d](https://github.com/Kwai-Kolors/Kolors/blob/038818d244ed103056abd10f429729a26af4d239/controlnet/outputs/Canny_dog_condition.jpg) `controlnet/outputs/Canny_dog_condition.jpg` | Kuaishou Technology (Kwai-Kolors) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | No |
| `test/ref/inpainting/kolors_origin.png` | [Kwai-Kolors/Kolors @ 038818d](https://github.com/Kwai-Kolors/Kolors/blob/038818d244ed103056abd10f429729a26af4d239/inpainting/asset/4.png) `inpainting/asset/4.png` | Kuaishou Technology (Kwai-Kolors) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | No |
| `test/ref/inpainting/kolors_mask.png` | [Kwai-Kolors/Kolors @ 038818d](https://github.com/Kwai-Kolors/Kolors/blob/038818d244ed103056abd10f429729a26af4d239/inpainting/asset/4_mask.png) `inpainting/asset/4_mask.png` | Kuaishou Technology (Kwai-Kolors) | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | No |

### 4.2 Partner-provided images and derivatives

| File | Origin | License |
|---|---|---|
| `test/assets/removal_input_1024x1024.jpg` | Provided to Advanced Micro Devices, Inc. by a partner for use in this project | Redistributed with the permission of the provider (see note 3) |
| `test/ref/removal/origin.jpg` | Copy of `test/assets/removal_input_1024x1024.jpg` | Same as `test/assets/removal_input_1024x1024.jpg` |
| `test/assets/removal_input_mask_1024x1024.jpg` | Mask created by Advanced Micro Devices, Inc. for `removal_input_1024x1024.jpg` | MIT |
| `test/ref/removal/mask.jpg` | Copy of `test/assets/removal_input_mask_1024x1024.jpg` | MIT |
| `test/assets/removal_output_1024x1024.png` | Generated by Advanced Micro Devices, Inc. from `removal_input_1024x1024.jpg` and its mask | MIT for the AMD contribution; includes content from the partner-provided input (see note 3) |

### 4.3 Images created by AMD

These images were generated or drawn by Advanced Micro Devices, Inc. and are distributed under the MIT license of this repository. Where a generative model was used, the model and its license are named. None of those licenses claim ownership of generated output.

| File | Origin | License |
|---|---|---|
| `test/assets/coffee_table.png` | Generated with SD 3.0 Medium by Advanced Micro Devices, Inc. (commit 3377d8a) | MIT |
| `test/assets/controlimg_input_1024x1024.png` | Generated by Advanced Micro Devices, Inc. with [Segmind Vega](https://huggingface.co/segmind/Segmind-Vega) (Apache-2.0) | MIT |
| `test/assets/controlimg_output_strength_0.95_1024x1024.png` | Generated by Advanced Micro Devices, Inc. with [Segmind Vega](https://huggingface.co/segmind/Segmind-Vega) (Apache-2.0) | MIT |
| `test/assets/flux2_img1.png` | Generated by Advanced Micro Devices, Inc. with [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0) | MIT |
| `test/assets/flux2_img2.png` | Generated by Advanced Micro Devices, Inc. with [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0) | MIT |
| `test/assets/flux2_img3.png` | Generated by Advanced Micro Devices, Inc. with [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache-2.0) | MIT |
| `test/assets/image-to-image-controlnet-workflow.png` | Original diagram by Advanced Micro Devices, Inc. (commit 3377d8a) | MIT |
| `test/assets/image-to-image-workflow.png` | Original diagram by Advanced Micro Devices, Inc. (commit bead652) | MIT |
| `test/assets/inpainting_output_1024x1024.png` | Generated with SD 3.0 Medium by Advanced Micro Devices, Inc. (commit 3377d8a) | MIT |
| `test/assets/living_room_mask_books_vase_1024x1024.png` | Generated or created by Advanced Micro Devices, Inc. with AI models running on AMD hardware | MIT |
| `test/assets/outpainting_input_512x512.png` | Generated by Advanced Micro Devices, Inc. with [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) (Stability AI Community License) | MIT |
| `test/assets/outpainting_output_1024x1024.png` | Generated or created by Advanced Micro Devices, Inc. with AI models running on AMD hardware | MIT |
| `test/assets/precise_mask.png` | Created by Advanced Micro Devices, Inc. to match `coffee_table.png` (commit 3377d8a) | MIT |
| `test/assets/sd_demo.gif` | Generated or created by Advanced Micro Devices, Inc. with AI models running on AMD hardware | MIT |
| `test/assets/text-to-image-workflow.png` | Original diagram by Advanced Micro Devices, Inc. (commit 3377d8a) | MIT |
| `test/ref/inpainting/mask.jpg` | Generated or created by Advanced Micro Devices, Inc. with AI models running on AMD hardware | MIT |
| `test/ref/inpainting/origin.jpg` | Generated or created by Advanced Micro Devices, Inc. with AI models running on AMD hardware | MIT |
| `test/ref/outpainting.jpg` | Generated or created by Advanced Micro Devices, Inc. with AI models running on AMD hardware | MIT |
| `test/ref/union.jfif` | Generated or created by Advanced Micro Devices, Inc. with AI models running on AMD hardware | MIT |

Notes:

1. The InstantX SD3 ControlNet repositories are published as a set, and `SD3-Controlnet-Depth` is licensed Apache-2.0. The `SD3-Controlnet-Canny`, `SD3-Controlnet-Pose`, and `SD3-Controlnet-Tile` model cards do not repeat the license, so their sample images are treated as Apache-2.0 on the same basis.
2. `images/bird_canny.png` is published in the `lllyasviel/sd-controlnet-canny` repository, which is licensed under CreativeML OpenRAIL-M. The use-based restrictions of that license continue to apply to this image.
3. This image was supplied to Advanced Micro Devices, Inc. by a partner for use in this project. It is not covered by the MIT license of this repository.

## 5. Internal-only files

None. Every file in this repository is intended for public release.
