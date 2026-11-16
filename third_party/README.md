# Third-party code

Source files in this directory come from external open-source projects. They are **not** covered by the MIT License at the repository root. Each file keeps its upstream copyright notice and license header, and files changed by Advanced Micro Devices, Inc. carry a `Modifications Copyright` line that describes the change.

| Path | Upstream | License |
|---|---|---|
| `kolors/configuration_chatglm.py` | [Kwai-Kolors/Kolors](https://github.com/Kwai-Kolors/Kolors/tree/038818d244ed103056abd10f429729a26af4d239/kolors/models) | Apache-2.0 |
| `kolors/modeling_chatglm.py` | [Kwai-Kolors/Kolors](https://github.com/Kwai-Kolors/Kolors/tree/038818d244ed103056abd10f429729a26af4d239/kolors/models) | Apache-2.0 |
| `kolors/tokenization_chatglm.py` | [Kwai-Kolors/Kolors](https://github.com/Kwai-Kolors/Kolors/tree/038818d244ed103056abd10f429729a26af4d239/kolors/models) (modified) | Apache-2.0 |
| `diffusers/flux2/image_processor.py` | [huggingface/diffusers](https://github.com/huggingface/diffusers/blob/7448258505c504d2070a48abd3ca56543324c016/src/diffusers/pipelines/flux2/image_processor.py) `pipelines/flux2/image_processor.py` (modified) | Apache-2.0 |
| `diffusers/pipeline_utils.py` | [huggingface/diffusers v0.35.1](https://github.com/huggingface/diffusers/blob/0f252be0ed42006c125ef4429156cb13ae6c1d60/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py) `rescale_noise_cfg`, `retrieve_timesteps` | Apache-2.0 |
| `t5/modeling_t5.py` | [huggingface/transformers](https://github.com/huggingface/transformers) `models/t5/modeling_t5.py` (modified) | Apache-2.0 |

The `__init__.py` files are package markers written by AMD (MIT).

The full Apache License 2.0 text is in the root [LICENSE](../LICENSE). [MANIFEST.md](../MANIFEST.md) lists all third-party code. That includes the diffusers-derived pipelines in `src/pipeline/`, which stay there because AMD rewrote large parts of them for ONNX Runtime and the NPU. They keep their upstream Apache-2.0 headers.
