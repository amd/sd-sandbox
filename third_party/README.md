# Third-party code

Source files in this directory come from external open-source projects. They are **not** covered by the MIT License at the repository root. Each file keeps its upstream copyright notice and license header, and files changed by Advanced Micro Devices, Inc. carry a `Modifications Copyright` line that describes the change.

| Path | Upstream | License |
|---|---|---|
| `kolors/configuration_chatglm.py` | [Kwai-Kolors/Kolors](https://github.com/Kwai-Kolors/Kolors/tree/038818d244ed103056abd10f429729a26af4d239/kolors/models) | Apache-2.0 |
| `kolors/modeling_chatglm.py` | [Kwai-Kolors/Kolors](https://github.com/Kwai-Kolors/Kolors/tree/038818d244ed103056abd10f429729a26af4d239/kolors/models) | Apache-2.0 |
| `kolors/tokenization_chatglm.py` | [Kwai-Kolors/Kolors](https://github.com/Kwai-Kolors/Kolors/tree/038818d244ed103056abd10f429729a26af4d239/kolors/models) (modified) | Apache-2.0 |
| `t5/modeling_t5.py` | [huggingface/transformers](https://github.com/huggingface/transformers) `models/t5/modeling_t5.py` (modified) | Apache-2.0 |

The `__init__.py` files are package markers written by AMD (MIT).

The full Apache License 2.0 text is in the root [LICENSE](../LICENSE). [MANIFEST.md](../MANIFEST.md) lists all third-party code, including the diffusers-derived pipelines under `src/pipeline/`, which still live in `src/`.
