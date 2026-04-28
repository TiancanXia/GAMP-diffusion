# GAMP-Diffusion: Generalized Approximate Message Passing-Based Diffusion Posterior Sampling

This repository implements **GAMP-Diffusion**, a novel approach that combines Generalized Approximate Message Passing (GAMP) with diffusion models for solving linear and nonlinear inverse problems. The framework supports various tasks including compressed sensing (CS), image deblurring, inpainting, super-resolution, and phase retrieval.

## Key Features

- **GAMP Integration**: Implements GAMP algorithm for efficient posterior sampling in diffusion models
- **Multiple Algorithms**: Supports MMPS, PGDM, DPS, GAMP-MMPS, GAMP-PGDM, and VAMP algorithms
- **Versatile Inverse Problems**: Handles both linear (CS, deblurring, inpainting) and nonlinear inverse problems
- **Flexible Configuration**: Easy-to-use YAML configuration files for different tasks

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

**Note**: ImageNet checkpoint is also available in the same location.

### 3. Setup Environment

#### Option 1: Local Environment Setup

Clone external repositories for motion blurring and nonlinear deblurring:

```bash
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse
git clone https://github.com/LeviBorodenko/motionblur motionblur
```

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

**Note**: For ImageNet experiments, use `configs/imagenet_model_config.yaml`.

## Supported Tasks

### Linear Inverse Problems
- **Compressed Sensing** (`configs/CS_config.yaml`)
- **Super Resolution** (`configs/super_resolution_config.yaml`)
- **Gaussian Deblurring** (`configs/gaussian_deblur_config.yaml`)
- **Motion Deblurring** (`configs/motion_deblur_config.yaml`)
- **Inpainting** (`configs/inpainting_config.yaml`)

### Nonlinear Inverse Problems
- **Nonlinear Deblurring** (`configs/nonlinear_deblur_config.yaml`)
- **Phase Retrieval** (`configs/phase_retrieval_config.yaml`)

## Configuration File Structure

Task configurations are defined in YAML files. Here's an example structure:

```yaml
conditioning:
  method: ps  # Options: ps, projection, mcg, vanilla (see guided_diffusion/condition_methods.py)
  params:
    scale: 2.0

algorithm:
  name: gamp_mmps  # Options: mmps, pgdm, dps, gamp_mmps, gamp_pgdm, vamp

data:
  name: ffhq  # Options: ffhq, bedroom, cat, celeba-hq
  root: ./data/ffhq_samples/

measurement:
  operator:
    name: CS  # Operator type
  noise:
    name: gaussian  # Options: gaussian, poisson
    sigma: 0.05
```

### Key Parameters

- **`conditioning.method`**: The conditioning method for posterior sampling
- **`algorithm.name`**: The reconstruction algorithm (GAMP-based or baseline methods)
- **`measurement.operator.name`**: The measurement operator type
- **`noise.sigma`**: Noise level for Gaussian noise (adjust based on your scenario)

## Project Structure

```
GAMP-diffusion/
├── GAMP.py                 # Core GAMP algorithm implementation
├── sample_condition.py     # Main inference script
├── compute_metric.py       # Metric computation utilities
├── guided_diffusion/       # Diffusion model components
│   ├── condition_methods.py # Conditioning methods (PS, MCG, etc.)
│   ├── measurements.py      # Measurement operators
│   └── gaussian_diffusion.py # Diffusion sampler with GAMP integration
├── configs/                # Configuration files for different tasks
├── models/                 # Pretrained checkpoints
├── results/                # Output directory for reconstructed images
└── util/                   # Utility functions
```

## Algorithm Details

The GAMP-Diffusion framework integrates GAMP iterations within the diffusion sampling process:

1. **Output Node Update**: Computes estimates using the likelihood function
2. **Input Node Update**: Incorporates prior information through the diffusion model
3. **Gradient-based Refinement**: Uses diffusion model gradients for enhanced reconstruction

Supported algorithms:
- **MMPS**: Manifold-constrained Message Passing Sampling
- **PGDM**: Projected Gradient Descent with Diffusion Models  
- **DPS**: Diffusion Posterior Sampling
- **GAMP-MMPS**: GAMP-enhanced MMPS
- **GAMP-PGDM**: GAMP-enhanced PGDM
- **VAMP**: Vector Approximate Message Passing

## Results

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

