<div align="center">

# GenAI-SD — Stable Diffusion on AMD Ryzen AI

### High-performance AI image generation accelerated by the AMD NPU

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![AMD Ryzen AI](https://img.shields.io/badge/AMD-Ryzen%20AI-red.svg)](https://www.amd.com/en/products/ryzen-ai)

</div>

GenAI-SD runs Stable Diffusion pipelines (SD1.5, SDXL, SD3/3.5, FLUX.1-schnell, FLUX.2 Klein,
and more) accelerated by the AMD Ryzen AI NPU via Dynamic Dispatch (DD). Models
run >90% of their compute on the NPU at BFP16 precision.

---

## Documentation

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

---

## Quick Start

```powershell
# 1. Install Ryzen AI Software (drivers, conda env, wheels):
#    https://ryzenai.docs.amd.com/en/latest/inst.html

# 2. Activate the environment
conda activate ryzen-ai-1.8.0

# 3. Run SDXL Turbo (model downloads automatically from Hugging Face)
python scripts/run_pipelines.py --pipelines sdxl_turbo
```

See [docs/01-quick-start.md](docs/01-quick-start.md) for full setup instructions. Linux: [docs/05-linux-guide.md](docs/05-linux-guide.md).

---

## Supported Models

| Model | Resolution | NPU Offload |
|-------|-----------|-------------|
| SD 1.5 / SD Turbo / SD 2.1 | 512×512 | ~94% |
| SDXL-base / SDXL-turbo / Playground 2.5 / SSD-1B / DreamShaper XL / Segmind-Vega | 512–1024 | ~90% |
| SD3 / SD3.5 Medium (T2I + ControlNet + OutPainting) | 512–1024 dynamic | ~90% |
| FLUX.1-schnell | 288–1024 dynamic | ~90% |
| FLUX.2 Klein (T2I + image editing) | 1024×1024 | ~90% |

---

## Hardware Requirements

Requires an AMD Ryzen AI processor (Strix Point, Strix Halo, or Krackan Point). See
[docs/01-quick-start.md](docs/01-quick-start.md) for setup (Linux: [docs/05-linux-guide.md](docs/05-linux-guide.md)).

---

## Repository Layout

```
GenAI-SD/
├─ scripts/run_pipelines.py      # Batch pipeline orchestrator (start here)
├─ test/run.py                   # Unified single-model runner (all pipelines)
├─ config/
│  ├─ pipeline_configs.yaml      # Pipeline definitions and groups
│  └─ dynamic_shape_*.json       # Dynamic resolution configs
├─ docs/                         # Full documentation (see table above)
├─ src/                          # Pipeline source code
└─ models/                       # Model cache (auto-downloaded; override with
                                 #   RYZENAI_GENAI_SD_MODELS_ROOT env var)
```

---

## License

© 2024–2026 Advanced Micro Devices, Inc. See [LICENSE](LICENSE).
