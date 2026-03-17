# OpenEMMA-UI

A real-time UI wrapper that adapts [OpenEMMA](https://github.com/taco-group/OpenEMMA) for live autonomous driving in [CARLA Simulator](https://carla.org/) (0.9.16).

OpenEMMA was originally designed for offline trajectory prediction on the nuScenes dataset. This project brings it into a **real-time CARLA environment** with a visual UI, multi-model VLM support, and a Chain-of-Thought (CoT) reasoning display.

![Architecture](https://img.shields.io/badge/CARLA-0.9.16-blue) ![Python](https://img.shields.io/badge/Python-3.12-green) ![License](https://img.shields.io/badge/License-Apache%202.0-red)

---

## Features

- **Real-time CARLA driving** with route-based pure-pursuit controller
- **4-step Chain-of-Thought pipeline** displayed in UI:
  1. Scene Description
  2. Critical Object Detection
  3. Driving Intent
  4. Motion Prediction
- **4 VLM backends** supported:
  - LLaVA-v1.5-7b (local)
  - LLaMA-3.2-11B-Vision (local, recommended)
  - Qwen2-VL-7B-Instruct (local)
  - GPT-4o (OpenAI API)
- **Safety systems**: red light detection, lane-keeping correction, stall recovery
- **Chase camera** with real-time info panel (speed, steering, VLM I/O)

---

## Architecture

```
openemmaUI.py                    # Main launcher & OpenEMMA agent
├── ui_common/
│   ├── agent_runner.py          # CARLA simulation runner + SafetyLimiter
│   ├── panel.py                 # Info panel with color-coded LLM I/O
│   ├── camera.py                # Third-person chase camera
│   ├── renderer.py              # Window compositor (camera + panel)
│   ├── carla_setup.py           # Auto-detect CARLA PythonAPI
│   └── carla_utils.py           # CARLA connection utilities
└── OpenEMMA/                    # Original OpenEMMA (cloned separately)
    ├── openemma/
    └── llava/
```

**Control flow:**
- **Primary control**: Route-based pure-pursuit steering (no VLM dependency)
- **VLM advisory**: CoT runs every 20 frames in a background thread; results modulate target speed and are displayed in the UI panel
- **Safety layer**: `SafetyLimiter` overrides control for red lights, off-road correction, and stall recovery

---

## Experimental Environment

All experiments and benchmarks were conducted on the following hardware:

| Component | Specification |
|---|---|
| **CPU** | Intel Core i9-14900K |
| **RAM** | 128 GB DDR5 |
| **GPU** | NVIDIA RTX 5090 (32 GB VRAM) |
| **OS** | Windows 11 Pro |
| **CUDA** | 12.8 |
| **Python** | 3.12 |
| **CARLA** | 0.9.16 |

---

## Prerequisites

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 / Linux | Windows 11 Pro |
| **GPU** | 16 GB VRAM (LLaVA) | 24+ GB VRAM (LLaMA-3.2-11B) |
| **RAM** | 16 GB | 32+ GB |
| **CUDA** | 12.1+ | 12.8+ |
| **Python** | 3.10 | 3.12 |
| **CARLA** | 0.9.16 | 0.9.16 |

---

## Installation

### Step 1: Install CARLA 0.9.16

Download and extract CARLA 0.9.16 from the [official releases](https://github.com/carla-simulator/carla/releases/tag/0.9.16/).

```bash
# Example: extract to a known location
# Windows: C:\CARLA_0.9.16\
# Linux: /opt/carla/
```

### Step 2: Clone this repository

```bash
git clone https://github.com/justinbrianhwang/OpenEMMA-UI.git
cd OpenEMMA-UI
```

### Step 3: Clone the original OpenEMMA

```bash
git clone https://github.com/taco-group/OpenEMMA.git
```

### Step 4: Create conda environment

```bash
conda create -n openemma python=3.12 -y
conda activate openemma
```

### Step 5: Install PyTorch (match your CUDA version)

```bash
# CUDA 12.8 example
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1 example
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 6: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 7: Install CARLA PythonAPI

```bash
# Find the .whl file in your CARLA installation
pip install /path/to/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl
```

### Step 8: Download VLM model weights

Choose one or more backends:

```bash
# LLaMA-3.2-11B-Vision (recommended, ~22GB VRAM)
# Downloads automatically from HuggingFace on first run
# Or pre-download:
python -c "from transformers import MllamaForConditionalGeneration; MllamaForConditionalGeneration.from_pretrained('meta-llama/Llama-3.2-11B-Vision-Instruct')"

# LLaVA-v1.5-7b (~14GB VRAM)
python -c "from transformers import LlavaForConditionalGeneration; LlavaForConditionalGeneration.from_pretrained('llava-hf/llava-1.5-7b-hf')"

# Qwen2-VL-7B (~16GB VRAM)
python -c "from transformers import Qwen2VLForConditionalGeneration; Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')"

# GPT-4o: No download needed, just set API key
export OPENAI_API_KEY=sk-...
```

---

## Usage

### 1. Start CARLA server

```bash
# Windows
cd C:\CARLA_0.9.16
CarlaUE4.exe

# Linux
cd /opt/carla
./CarlaUE4.sh
```

### 2. Run OpenEMMA-UI

```bash
conda activate openemma

# LLaMA-3.2-11B-Vision (recommended)
python openemmaUI.py --llama

# LLaVA-v1.5-7b
python openemmaUI.py --llava

# Qwen2-VL-7B
python openemmaUI.py --qwen

# GPT-4o (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python openemmaUI.py --gpt

# Specify town
python openemmaUI.py --llama --town Town02

# Custom model path
python openemmaUI.py --model-path /path/to/custom/model
```

### 3. Controls

| Key | Action |
|---|---|
| **ESC** | Quit |
| **Mouse** | UI interaction |

---

## Demo Videos

Watch each VLM backend driving in CARLA Town01 (click thumbnails to play):

| LLaVA-v1.5-7b (★★☆☆☆) | LLaMA-3.2-11B-Vision (★★★★☆) |
|:---:|:---:|
| [![LLaVA Demo](https://img.youtube.com/vi/-hqY_IJ-oLM/0.jpg)](https://youtu.be/-hqY_IJ-oLM?si=WCQXpIlqWg8wZzTO) | [![LLaMA Demo](https://img.youtube.com/vi/UnAts81OVHg/0.jpg)](https://youtu.be/UnAts81OVHg?si=jEuLYNvl4D_2DqHr) |

| Qwen2-VL-7B (★★★☆☆) | GPT-4o (★★★★★) |
|:---:|:---:|
| [![Qwen Demo](https://img.youtube.com/vi/nGFZfqjub6I/0.jpg)](https://youtu.be/nGFZfqjub6I?si=DFYdzwhONPxmxK7t) | [![GPT-4o Demo](https://img.youtube.com/vi/3Ib4MReRG60/0.jpg)](https://youtu.be/3Ib4MReRG60?si=kfxFXGmOxH8WLecc) |

---

## VLM Model Comparison

We benchmarked 4 VLM backends on Town01 with identical routes and traffic conditions.

| Model | Scene Quality | Hallucination | Intent Accuracy | VRAM | Cost | Rating |
|---|---|---|---|---|---|---|
| **GPT-4o** | Excellent | Minimal | High | 0 GB | ~$0.01/frame | ★★★★★ |
| **LLaMA-3.2-11B** | Good | Minimal | High | ~22 GB | Free | ★★★★☆ |
| **Qwen2-VL-7B** | Decent | Moderate | Medium | ~16 GB | Free | ★★★☆☆ |
| **LLaVA-v1.5-7b** | Poor | Severe | Low | ~14 GB | Free | ★★☆☆☆ |

> See [VLM_Model_Comparison.md](VLM_Model_Comparison.md) for detailed analysis with examples.

**Key findings:**
- **LLaMA-3.2-11B-Vision** is the best local model with minimal hallucination
- **LLaVA and Qwen** suffer from persistent "red traffic light" hallucination on empty roads
- **GPT-4o** provides the most detailed scene descriptions but requires API costs

---

## Project Structure

```
OpenEMMA-UI/
├── README.md                    # This file
├── LICENSE                      # Apache 2.0
├── requirements.txt             # Python dependencies
├── openemmaUI.py                # Main launcher & agent (route + VLM CoT)
├── VLM_Model_Comparison.md      # Detailed VLM benchmark results
├── ui_common/                   # Shared UI & simulation framework
│   ├── __init__.py
│   ├── agent_runner.py          # AgentRunner + SafetyLimiter
│   ├── panel.py                 # InfoPanel (speed, steer, LLM I/O)
│   ├── camera.py                # ChaseCameraManager
│   ├── renderer.py              # UIRenderer (compositor)
│   ├── carla_setup.py           # CARLA PythonAPI auto-detection
│   └── carla_utils.py           # CarlaConnection helper
└── OpenEMMA/                    # Clone from taco-group/OpenEMMA
    ├── openemma/
    ├── llava/
    └── ...
```

---

## Troubleshooting

<details>
<summary><b>CARLA connection refused / module 'carla' has no attribute 'Client'</b></summary>

- Make sure `CarlaUE4.exe` is running before launching OpenEMMA-UI
- Verify the CARLA PythonAPI wheel is installed in your conda env:
  ```bash
  pip install /path/to/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl
  ```
- Check that your Python version matches the wheel (e.g., `cp312` = Python 3.12)

</details>

<details>
<summary><b>CUDA out of memory</b></summary>

- LLaMA-3.2-11B requires ~22 GB VRAM. If your GPU has less, try:
  - `--llava` (14 GB) or `--qwen` (16 GB) instead
  - `--gpt` uses no local VRAM (cloud API)
- Close other GPU-intensive applications before running

</details>

<details>
<summary><b>Model download stuck / HuggingFace authentication error</b></summary>

- Some models (e.g., LLaMA-3.2) require accepting the license on HuggingFace first
- Login via: `huggingface-cli login`
- Or download manually and use `--model-path /local/path`

</details>

<details>
<summary><b>Black screen / no camera output</b></summary>

- CARLA rendering may take a few seconds to initialize
- Try switching to a simpler town: `--town Town01`
- Ensure your GPU drivers are up to date

</details>

<details>
<summary><b>IMU sensor timeout warning</b></summary>

- `[WARNING] Sensor timeout. Missing: {'imu'}` is non-critical
- The system falls back to speed-only estimation automatically
- This occurs occasionally in CARLA 0.9.16 under high GPU load

</details>

---

## Known Limitations

- **VLM is advisory only**: The VLM Chain-of-Thought pipeline provides scene understanding displayed in the UI, but actual vehicle control relies on the route-based pure-pursuit controller. The VLM modulates target speed but does not directly steer.
- **Hallucination in smaller models**: LLaVA-v1.5-7b and Qwen2-VL-7B frequently hallucinate "red traffic light" on empty roads, causing unnecessary stops. Use LLaMA-3.2-11B or GPT-4o for more reliable scene understanding.
- **Single-town evaluation**: Current benchmarks are conducted on Town01. Results may vary on more complex maps (Town03, Town05) with different traffic patterns.
- **No multi-agent traffic**: Testing is performed with CARLA's default traffic manager. Dense traffic scenarios have not been extensively evaluated.
- **Windows-only testing**: While the codebase should work on Linux, it has only been tested on Windows 11.

---

## Citation

If you use this project in your research, please cite both the original OpenEMMA paper and this repository:

```bibtex
@misc{openemma2024,
    title={OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving},
    author={Shuo Xing and Chengyuan Qian and Hongyuan Hua and Kexin Tian and Yu Zhang and Siheng Chen and Zhengzhong Tu},
    year={2024},
    eprint={2412.15208},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{openemmaui2025,
    title={OpenEMMA-UI: Real-Time VLM-Driven Autonomous Driving in CARLA},
    author={Sunjun Hwang},
    year={2025},
    url={https://github.com/justinbrianhwang/OpenEMMA-UI}
}
```

---

## Acknowledgments

This project builds upon the excellent work of the original OpenEMMA authors:

> **OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving**
> Shuo Xing, Chengyuan Qian, Hongyuan Hua, Kexin Tian, Yu Zhang, Siheng Chen, Zhengzhong Tu
> *TACO Research Group, Texas A&M University*
> Paper: [arXiv:2412.15208](https://arxiv.org/abs/2412.15208)
> Repository: [taco-group/OpenEMMA](https://github.com/taco-group/OpenEMMA)

We sincerely thank the OpenEMMA team for making their research open-source, which made this real-time adaptation possible.

Additional thanks to:
- [CARLA Simulator](https://carla.org/) team for the open-source driving simulator
- [Hugging Face](https://huggingface.co/) for hosting the VLM model weights
- The developers of [LLaVA](https://github.com/haotian-liu/LLaVA), [LLaMA](https://ai.meta.com/llama/), [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL), and [GPT-4o](https://openai.com/) for their VLM models

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
