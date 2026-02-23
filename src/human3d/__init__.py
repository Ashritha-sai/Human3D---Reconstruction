"""
Human3D - 3D Human Reconstruction from Single Images

This package provides a complete pipeline for extracting 3D representations
of humans from single RGB images using depth estimation, pose detection,
segmentation, and 3D Gaussian Splatting.

Main Components:
    Human3DPipeline: Main orchestration class for the full pipeline
    GaussianTrainer: 3D Gaussian Splatting trainer for single images

Subpackages:
    models: Neural network wrappers (MiDaS, YOLO, SAM)
    reconstruct: 3D reconstruction (point clouds, Gaussians)
    utils: Configuration, device, and I/O utilities
    viz: Visualization functions
    export: Export functionality for Gaussian splats
    configs: Default configuration files

Example:
    >>> from human3d import Human3DPipeline
    >>> from human3d.utils import load_config
    >>> cfg = load_config("config.yaml")
    >>> pipeline = Human3DPipeline(cfg)
    >>> output_dir = pipeline.run("input.jpg")

    >>> # For Gaussian Splatting
    >>> from human3d.reconstruct import GaussianTrainer, GaussianConfig, CameraParams
    >>> trainer = GaussianTrainer(rgb, depth, mask, camera, config)
    >>> trainer.initialize_gaussians()
    >>> trainer.optimize()
    >>> trainer.export_ply("gaussians.ply")
"""

__version__ = "0.1.0"

# Main pipeline
from .pipeline import Human3DPipeline

# Reconstruction components
from .reconstruct import (
    GaussianTrainer,
    GaussianConfig,
    CameraParams,
    depth_to_pointcloud,
)

# Convenience imports
from .utils import load_config, pick_device

__all__ = [
    "__version__",
    # Pipeline
    "Human3DPipeline",
    # Gaussian Splatting
    "GaussianTrainer",
    "GaussianConfig",
    "CameraParams",
    # Utilities
    "depth_to_pointcloud",
    "load_config",
    "pick_device",
]
