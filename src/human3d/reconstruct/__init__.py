"""
Reconstruction module for Human3D.

Provides 3D reconstruction capabilities from RGB-D data, including
point cloud generation and Gaussian splatting.

Submodules:
    pointcloud: Depth-to-point-cloud unprojection and PLY export
    gaussian_trainer: 3D Gaussian Splatting training
    gaussian_utils: Mathematical utilities for Gaussian operations
    losses: Loss functions for Gaussian optimization

Classes:
    GaussianTrainer: Main trainer class for Gaussian Splatting
    GaussianConfig: Configuration dataclass for training
    CameraParams: Camera intrinsic/extrinsic parameters
    GaussianLosses: Combined loss functions

Functions:
    depth_to_pointcloud: Convert depth map to Open3D point cloud
    save_ply: Save Open3D point cloud to PLY file
    depth_to_xyz: Unproject depth to 3D coordinates
    estimate_point_scales: Estimate Gaussian scales from point density
"""

# Original point cloud functionality
from .pointcloud import depth_to_pointcloud, save_ply

# Gaussian splatting (new)
from .gaussian_trainer import (
    GaussianTrainer,
    GaussianConfig,
    CameraParams,
)

from .gaussian_utils import (
    normalize_depth_to_metric,
    depth_to_xyz,
    estimate_point_scales,
    rgb_to_spherical_harmonics,
    spherical_harmonics_to_rgb,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    build_covariance_3d,
    project_covariance_to_2d,
    sigmoid,
    inverse_sigmoid,
    # Camera utilities
    create_camera_intrinsics,
    create_front_camera_pose,
    create_orbit_camera_pose,
    get_view_direction,
)

from .losses import (
    GaussianLosses,
    gaussian_window,
    compute_ssim_map,
)

__all__ = [
    # Point cloud
    "depth_to_pointcloud",
    "save_ply",
    # Gaussian trainer
    "GaussianTrainer",
    "GaussianConfig",
    "CameraParams",
    # Gaussian utilities
    "normalize_depth_to_metric",
    "depth_to_xyz",
    "estimate_point_scales",
    "rgb_to_spherical_harmonics",
    "spherical_harmonics_to_rgb",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "build_covariance_3d",
    "project_covariance_to_2d",
    "sigmoid",
    "inverse_sigmoid",
    # Camera utilities
    "create_camera_intrinsics",
    "create_front_camera_pose",
    "create_orbit_camera_pose",
    "get_view_direction",
    # Losses
    "GaussianLosses",
    "gaussian_window",
    "compute_ssim_map",
]
