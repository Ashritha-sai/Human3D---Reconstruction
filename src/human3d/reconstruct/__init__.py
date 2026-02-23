"""
Reconstruction module for Human3D.

Provides 3D reconstruction capabilities from RGB-D data, including
point cloud generation and Gaussian splatting.
"""

# Gaussian splatting (core — always available)
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


# Point cloud functionality requires open3d — lazy import
def __getattr__(name):
    if name in ("depth_to_pointcloud", "save_ply"):
        from . import pointcloud
        return getattr(pointcloud, name)
    raise AttributeError(f"module 'human3d.reconstruct' has no attribute {name!r}")


__all__ = [
    # Point cloud (lazy)
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
