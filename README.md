# GAMP-Diffusion: Generalized Approximate Message Passing-Based Diffusion Posterior Sampling

This repository implements **GAMP-Diffusion**, a novel approach that combines Generalized Approximate Message Passing (GAMP) with diffusion models for solving inverse problems.

## Key Features

- **GAMP Integration**: Implements GAMP algorithm for efficient posterior sampling in diffusion models
- **Multiple Algorithms**: Supports MMPS, PGDM, DPS, GAMP-MM, and GAMP-GA algorithms
- **Flexible Configuration**: Easy-to-use YAML configuration files for different tasks
- **Non-Differentiable Observations**: Supports element-wise non-differentiable measurement functions $y = g(Ax + n)$ (e.g., quantization), solved via GAMP's decoupled output-step likelihood estimation

## Non-Differentiable Observation Support

GAMP-Diffusion extends the standard linear observation model to handle **non-differentiable element-wise measurement functions** $y = g(Ax + n)$, where $g$ can be any element-wise function (e.g., quantization, saturation). This is achieved by replacing only the output-step likelihood in GAMP, leaving the diffusion prior unchanged. Traditional gradient-based methods (DPS, MMPS, PGDM) are inapplicable here — GAMP-based algorithms are required.

### Quantized Compressed Sensing

Uniform quantization $y = Q_\Delta(Ax + n)$ with $Q_\Delta(\cdot) = \Delta \cdot \text{round}(\cdot / \Delta)$.

```bash
python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/quantized_CS_config.yaml \
    --gpu=0 \
    --save_dir=./results
```

See `interface.md` for detailed mathematical derivation and implementation notes.

## Prerequisites

- Python 3.8+
- PyTorch 1.11.0+
- CUDA 11.3+ (GPU recommended)
- NVIDIA-Docker (optional, for containerized deployment)

Lower CUDA versions are supported with appropriate PyTorch versions (e.g., CUDA 10.2 with PyTorch 1.7.0).

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TiancanXia/GAMP-diffusion.git
cd GAMP-diffusion
```

### 2. Download Pretrained Checkpoints

Download the pretrained checkpoint `ffhq_10m.pt` from [Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) and place it in the `models/` directory:

```bash
mkdir models
mv {DOWNLOAD_DIR}/ffhq_10m.pt ./models/
```

### 3. Setup Environment

#### Option 1: Local Environment Setup

Create conda environment and install dependencies:

```bash
conda create -n gamp python=3.8
conda activate gamp
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

#### Option 2: Docker Container

Build and run the Docker container (requires Docker >= 19.03 with GPU support):

```bash
docker build -t gamp-diffusion:latest .
docker run -it --rm --gpus=all gamp-diffusion
```

### 4. Run Inference

Execute the sampling script with your desired configuration:

```bash
python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/CS_config.yaml
```

## Project Structure

```
GAMP-diffusion/
├── GAMP.py                 # Core GAMP algorithm implementation
├── sample_condition.py     # Main inference script
├── compute_metric.py       # Metric computation utilities
├── guided_diffusion/       # Diffusion model components
│   ├── measurements.py      # Measurement operators
│   └── gaussian_diffusion.py # Diffusion sampler with GAMP integration
├── configs/                # Configuration files for different tasks
├── models/                 # Pretrained checkpoints
├── results/                # Output directory for reconstructed images
└── util/                   # Utility functions
```

## Algorithm Details

Example results will be saved in the `results/` directory with the following structure:
- `input/`: Measurement inputs
- `recon/`: Reconstructed images
- `progress/`: Intermediate sampling results
- `label/`: Ground truth images (for comparison)

## Citation

If you find this work useful in your research, please cite the original DPS paper:

```bibtex
@inproceedings{
chung2023diffusion,
title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation builds upon the foundation of [Diffusion Posterior Sampling](https://github.com/DPS2022/diffusion-posterior-sampling) and incorporates GAMP algorithms for improved inverse problem solving.

