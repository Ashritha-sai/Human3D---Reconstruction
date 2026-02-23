# Human3D – Context-Aware 3D Reconstruction from a Single Image

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Ashritha-sai/Human3D---Reconstruction/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashritha-sai/Human3D---Reconstruction/actions/workflows/ci.yml)

**Human3D** is a modular, end-to-end pipeline for reconstructing 3D human geometry from a single RGB image.
The system supports both **single-person** and **multi-person** scenes and is designed as a clean research foundation for future biomechanically-aware and differentiable human modeling.

This repository focuses on **robust perception, segmentation, depth estimation, and geometric reconstruction**.
Biomechanical constraints and parametric body fitting are active research directions and are intentionally excluded from this public codebase.

---

## Table of Contents

- [Key Features](#-key-features)
- [Pipeline Architecture](#-pipeline-architecture)
- [Gaussian Splatting Pipeline](#-gaussian-splatting-pipeline) ⭐ **NEW**
- [Example Outputs](#-example-outputs)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## Key Features

- Monocular depth estimation from a single RGB image
- Automatic **multi-person detection and pose estimation**
- Instance-aware **human segmentation**
- Dense **RGB-D point cloud reconstruction**
- **3D Gaussian Splatting** for high-quality novel view synthesis ⭐ **NEW**
- Works on both single-subject and group photographs
- Modular, research-friendly code structure

---

## Pipeline Architecture

```mermaid
graph LR
    A[RGB Image] --> B[MiDaS<br/>Depth Estimation]
    A --> C[YOLOv8-Pose<br/>Human Detection]
    C --> D[SAM<br/>Segmentation]
    B --> E[3D Point Cloud<br/>Reconstruction]
    D --> E
    E --> F[Output<br/>depth + pose + mask + .ply]
```

For a given input image, the pipeline performs the following steps:

| Stage | Model | Description |
|-------|-------|-------------|
| 1. Depth Estimation | **MiDaS** | Monocular depth prediction (DPT_Large, DPT_Hybrid, MiDaS_small) |
| 2. Pose Detection | **YOLOv8-Pose** | Multi-person detection and keypoint estimation |
| 3. Segmentation | **SAM** | Person-level instance segmentation guided by detected bounding boxes |
| 4. Reconstruction | **Pinhole Camera** | Conversion of depth maps into colored 3D point clouds |

**Auto-mode behavior:**
- **0 people detected**: Skip pose/segmentation; depth and point cloud still processed
- **1 person detected**: Single-person mode (selected subject analyzed)
- **2+ people detected**: Group mode (union of all subjects)

---

## Gaussian Splatting Pipeline

> ⭐ **New Feature**: High-quality 3D reconstruction using differentiable Gaussian Splatting

### Overview

The Gaussian Splatting module provides an alternative to traditional point cloud reconstruction, offering:
- **Photorealistic rendering** with view-dependent effects
- **Differentiable optimization** for fine-tuning geometry
- **Compact representation** suitable for real-time applications
- **Export compatibility** with popular 3DGS viewers

### Architecture

```mermaid
graph TB
    subgraph Input
        A[RGB Image] --> B[MiDaS Depth]
        A --> C[SAM Mask]
    end

    subgraph Initialization
        B --> D[Depth to XYZ]
        C --> D
        D --> E[Initialize Gaussians]
        A --> F[RGB to SH]
        F --> E
    end

    subgraph Optimization
        E --> G[Differentiable Rasterizer]
        G --> H[Compute Loss]
        H --> I{Converged?}
        I -->|No| J[Update Parameters]
        J --> G
        I -->|Yes| K[Densify & Prune]
        K --> L{Done?}
        L -->|No| G
    end

    subgraph Output
        L -->|Yes| M[Export PLY]
        M --> N[View in Browser]
    end

    style E fill:#e1f5fe
    style G fill:#fff3e0
    style M fill:#e8f5e9
```

### Gaussian Representation

Each 3D Gaussian primitive contains:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| Position (μ) | (N, 3) | 3D center coordinates |
| Covariance (Σ) | (N, 3, 3) | 3D extent and orientation |
| Color (SH) | (N, K, 3) | Spherical harmonics coefficients |
| Opacity (α) | (N,) | Transparency value |

### Quick Start

```python
from human3d.reconstruct.gaussian_trainer import GaussianTrainer, GaussianConfig, CameraParams
from human3d.export.ply_exporter import save_gaussian_ply

# Load your RGB-D data
rgb = ...      # (H, W, 3) uint8
depth = ...    # (H, W) float32
mask = ...     # (H, W) binary

# Configure camera
camera = CameraParams(fx=500, fy=500, cx=128, cy=128, width=256, height=256)

# Configure training
config = GaussianConfig(
    sh_degree=0,           # 0 = view-independent color
    num_iterations=1000,   # Training iterations
)

# Train Gaussians
trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cuda")
trainer.initialize_gaussians()
history = trainer.optimize(num_iterations=1000)

# Export for viewing
trainer.export_ply("output/gaussians.ply")
print("View at: https://antimatter15.com/splat/")
```

### Comparison: Point Cloud vs Gaussian Splatting

| Feature | Point Cloud | Gaussian Splatting |
|---------|-------------|-------------------|
| **Rendering Quality** | Discrete points, holes visible | Smooth, continuous surfaces |
| **View Synthesis** | Limited (fixed viewpoint) | Novel views with parallax |
| **File Size** | ~1 MB per 100K points | ~5 MB per 100K Gaussians |
| **Rendering Speed** | Fast (simple projection) | Real-time (GPU rasterization) |
| **Optimization** | None (direct conversion) | Iterative refinement |
| **Color Model** | RGB per point | Spherical harmonics (view-dependent) |
| **Best For** | Quick preview, CAD import | Visualization, VR/AR, games |

### Output Formats

The exported PLY file is compatible with:

| Viewer | URL | Features |
|--------|-----|----------|
| **antimatter15/splat** | [antimatter15.com/splat](https://antimatter15.com/splat/) | Browser-based, drag & drop |
| **SuperSplat** | [playcanvas.com/supersplat](https://playcanvas.com/supersplat/editor) | Editor, export options |
| **Luma AI** | [lumalabs.ai](https://lumalabs.ai/) | Cloud rendering |
| **3D Gaussian Viewer** | Various | Desktop applications |

### Hyperparameter Guide

```yaml
# configs/gaussian_config.yaml
training:
  num_iterations: 3000      # More = better quality, slower
  sh_degree: 0              # 0-3, higher = view-dependent effects

learning_rates:
  position: 0.00016         # Move Gaussian centers
  scale: 0.005              # Adjust Gaussian size
  rotation: 0.001           # Orient Gaussians
  color: 0.0025             # Refine appearance
  opacity: 0.05             # Adjust transparency

densification:
  from_iter: 500            # Start adding Gaussians
  until_iter: 3000          # Stop densification
  interval: 100             # Check every N iterations
  grad_threshold: 0.0002    # Sensitivity for splitting
```

### Performance Benchmarks

| Configuration | Time | Quality (PSNR) | Memory |
|--------------|------|----------------|--------|
| 1K iterations, SH=0 | ~30s | ~22 dB | 2 GB |
| 3K iterations, SH=0 | ~90s | ~26 dB | 2 GB |
| 5K iterations, SH=3 | ~5min | ~30 dB | 4 GB |

*Tested on RTX 3050, 256×256 input image*

### Documentation

For detailed technical documentation, see:
- [Gaussian Splatting Technical Guide](docs/GAUSSIAN_SPLATTING.md)
- [API Reference](docs/api/)

---

## Example Outputs

For each run, the pipeline generates a timestamped output directory:

```
outputs/run_YYYYMMDD_HHMMSS/
├── depth.npy           # Raw depth array (float32)
├── depth.png           # Depth visualization (inferno colormap)
├── pose_overlay.png    # Keypoints overlaid on input image
├── seg_mask.png        # Binary segmentation mask
├── seg_overlay.png     # Segmentation overlay visualization
├── pointcloud.ply      # Colored 3D point cloud
└── summary.json        # Run metadata and statistics
```

The pipeline automatically adapts to:
- **Single-person images** – focused reconstruction
- **Multi-person group images** – processes all detected individuals

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended) or CPU

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Ashritha-sai/Human3D---Reconstruction.git
cd Human3D

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Download Model Checkpoints

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download SAM checkpoint (ViT-B)
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mv checkpoints/sam_vit_b_01ec64.pth checkpoints/sam_vit_b.pth

# YOLOv8-Pose models are auto-downloaded on first run
# MiDaS models are auto-downloaded via torch.hub
```

---

## Usage

### Command Line

```bash
# Run the pipeline on an image
python scripts/run_pipeline.py --config configs/pipeline.yaml --input data/sample.jpg

# Specify output directory
python scripts/run_pipeline.py --config configs/pipeline.yaml --input data/sample.jpg --output results/
```

### Jupyter Notebooks

Interactive demos are available in the `notebooks/` directory:

| Notebook | Description |
|----------|-------------|
| `01_single_person_demo.ipynb` | Single-subject full pipeline walkthrough |
| `02_multi_person_demo.ipynb` | Multi-person group photo handling |
| `03_depth_to_pointcloud.ipynb` | Depth-to-3D geometry conversion details |

### Python API

```python
from human3d.pipeline import Human3DPipeline
from human3d.utils.config import load_config

config = load_config("configs/pipeline.yaml")
pipeline = Human3DPipeline(config)
results = pipeline.run("path/to/image.jpg")
```

---

## Project Structure

```
Human3D/
├── configs/                 # Configuration files
│   ├── pipeline.yaml       # Main pipeline configuration
│   └── gaussian_config.yaml # Gaussian splatting settings
├── checkpoints/            # Model weights (git-ignored)
├── data/                   # Input images (git-ignored)
├── docs/                   # Documentation
│   ├── GAUSSIAN_SPLATTING.md # Technical documentation
│   └── api/                # Generated API docs
├── notebooks/              # Jupyter notebook demos
│   ├── 01_single_person_demo.ipynb
│   ├── 02_multi_person_demo.ipynb
│   ├── 03_depth_to_pointcloud.ipynb
│   └── 04_gaussian_splatting.ipynb  # NEW
├── outputs/                # Pipeline outputs (git-ignored)
├── scripts/                # Entry point scripts
│   ├── run_pipeline.py
│   └── train_gaussians.py  # NEW: Gaussian training script
├── src/human3d/            # Core source code
│   ├── __init__.py
│   ├── pipeline.py         # Main orchestrator
│   ├── models/             # Model wrappers
│   │   ├── depth_midas.py  # MiDaS depth estimation
│   │   ├── pose_yolo.py    # YOLOv8 pose detection
│   │   └── seg_sam.py      # SAM segmentation
│   ├── reconstruct/        # 3D reconstruction
│   │   ├── pointcloud.py   # Depth-to-pointcloud conversion
│   │   ├── gaussian_trainer.py  # NEW: Gaussian optimization
│   │   ├── gaussian_utils.py    # NEW: Utility functions
│   │   └── losses.py            # NEW: Loss functions
│   ├── export/             # NEW: Export formats
│   │   └── ply_exporter.py # PLY file export
│   ├── utils/              # Utilities
│   │   ├── config.py       # Configuration loading
│   │   ├── device.py       # GPU/CPU selection
│   │   └── io.py           # File I/O helpers
│   └── viz/                # Visualization
│       └── overlays.py     # Image overlay utilities
├── tests/                  # Test suite
│   ├── test_ply_format.py  # NEW: PLY format tests
│   ├── test_ply_exporter.py # NEW: Exporter tests
│   ├── test_end_to_end.py  # NEW: Pipeline tests
│   └── fixtures/           # Test data
├── .github/                # GitHub templates and CI
├── requirements.txt        # Dependencies
├── requirements-lock.txt   # Locked dependencies
├── pyproject.toml          # Package metadata
├── CONTRIBUTING.md         # Contribution guidelines
├── KAEDIM_PORTFOLIO.md     # NEW: Portfolio documentation
├── CITATION.cff            # Citation file
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## Configuration

The pipeline is configured via `configs/pipeline.yaml`:

```yaml
run:
  output_dir: "outputs"
  name_prefix: "run"

device:
  prefer: "cuda"  # cuda or cpu

depth:
  enabled: true
  model_type: "DPT_Large"  # DPT_Large, DPT_Hybrid, MiDaS_small

pose:
  enabled: true
  model: "yolov8m-pose.pt"
  conf: 0.15  # Confidence threshold

segmentation:
  enabled: true
  method: "sam"
  sam:
    model_type: "vit_b"
    checkpoint_path: "checkpoints/sam_vit_b.pth"

reconstruction:
  enabled: true
  pointcloud:
    fx: 1000.0  # Focal length (approximate)
    fy: 1000.0
```

---

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/human3d --cov-report=html
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Citation

If you use this software in your research, please cite it:

```bibtex
@software{human3d2025,
  title = {Human3D: Context-Aware 3D Reconstruction from a Single Image},
  author = {Ashritha},
  year = {2025},
  url = {https://github.com/Ashritha-sai/Human3D---Reconstruction},
  license = {MIT}
}
```

See [CITATION.cff](CITATION.cff) for more citation formats.

---

## Acknowledgments

Human3D builds upon several excellent open-source projects:

- **[MiDaS](https://github.com/isl-org/MiDaS)** (Intel ISL) – Monocular depth estimation
- **[YOLOv8](https://github.com/ultralytics/ultralytics)** (Ultralytics) – Object detection and pose estimation
- **[Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)** (Meta AI) – Instance segmentation
- **[gsplat](https://github.com/nerfstudio-project/gsplat)** (Nerfstudio) – Differentiable Gaussian rasterization
- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** (INRIA) – Original 3DGS implementation
- **[Open3D](http://www.open3d.org/)** – 3D data processing
- **[PyTorch](https://pytorch.org/)** – Deep learning framework

We thank the authors and contributors of these projects for making their work available.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with <code>PyTorch</code> + <code>MiDaS</code> + <code>YOLOv8</code> + <code>SAM</code>
</p>
