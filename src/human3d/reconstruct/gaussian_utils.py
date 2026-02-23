"""
Gaussian Splatting Utility Functions

This module provides mathematical utilities for 3D Gaussian Splatting,
including depth unprojection, scale estimation, color space conversion,
and rotation representations.

Functions accept both NumPy arrays and PyTorch tensors where needed.
Internally, the trainer operates in PyTorch; NumPy paths are kept only
for the lightweight pointcloud.py utilities.
"""

from __future__ import annotations

from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import Tensor


def _to_numpy(x: Union[np.ndarray, Tensor]) -> np.ndarray:
    """Convert input to numpy array if it's a tensor."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


def normalize_depth_to_metric(
    depth: Union[np.ndarray, Tensor],
    min_depth: float = 0.5,
    max_depth: float = 2.5,
) -> Union[np.ndarray, Tensor]:
    """
    Normalize relative depth to pseudo-metric range.

    Args:
        depth: Raw depth map (relative or inverse depth from MiDaS).
            Shape: (H, W), dtype: float32
        min_depth: Minimum depth value in output range (meters).
        max_depth: Maximum depth value in output range (meters).

    Returns:
        Normalized depth map in [min_depth, max_depth].
    """
    if isinstance(depth, Tensor):
        d = depth.clone()
        d = d - torch.min(d[torch.isfinite(d)])
        denom = torch.max(d[torch.isfinite(d)]) + 1e-6
        d = d / denom
        return min_depth + (max_depth - min_depth) * d
    else:
        d = depth.copy()
        d = d - np.nanmin(d)
        denom = np.nanmax(d) + 1e-6
        d = d / denom
        z = min_depth + (max_depth - min_depth) * d
        return z.astype(np.float32)


def depth_to_xyz(
    depth: Union[np.ndarray, Tensor],
    mask: Union[np.ndarray, Tensor],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Union[np.ndarray, Tensor]:
    """
    Unproject depth map to 3D point cloud using pinhole camera model.

    Args:
        depth: Depth map. Shape: (H, W), dtype: float32
        mask: Binary mask for valid pixels. Shape: (H, W)
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point coordinates in pixels.

    Returns:
        xyz: 3D coordinates of valid masked pixels. Shape: (N, 3)

    Mathematical Operations:
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d
    """
    if isinstance(depth, Tensor):
        h, w = depth.shape[:2]
        device = depth.device

        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )

        z = depth.clone()
        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy

        xyz_full = torch.stack([X, Y, z], dim=-1)

        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        valid = mask_bool & (z > 0) & torch.isfinite(z)

        xyz = xyz_full.reshape(-1, 3)[valid.reshape(-1)]
        return xyz.float()
    else:
        h, w = depth.shape[:2]

        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.copy()

        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy

        xyz_full = np.stack([X, Y, z], axis=-1)

        mask_bool = mask.astype(bool) if mask.dtype != bool else mask
        valid = mask_bool & (z > 0) & np.isfinite(z)

        xyz = xyz_full.reshape(-1, 3)[valid.reshape(-1)]
        return xyz.astype(np.float32)


def estimate_point_scales(
    xyz_points: Union[np.ndarray, Tensor],
    k_neighbors: int = 8,
) -> Union[np.ndarray, Tensor]:
    """
    Estimate initial Gaussian scales based on local point density via k-NN.

    Args:
        xyz_points: 3D point positions. Shape: (N, 3), dtype: float32
        k_neighbors: Number of nearest neighbors to consider.

    Returns:
        scales: Isotropic scale for each point. Shape: (N, 3), dtype: float32
    """
    from scipy.spatial import KDTree

    is_tensor = isinstance(xyz_points, Tensor)

    if is_tensor:
        device = xyz_points.device
        xyz_np = xyz_points.detach().cpu().numpy()
    else:
        xyz_np = xyz_points

    n_points = xyz_np.shape[0]
    if n_points == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return torch.from_numpy(empty).to(device) if is_tensor else empty

    tree = KDTree(xyz_np)
    k_query = min(k_neighbors + 1, n_points)
    distances, _ = tree.query(xyz_np, k=k_query)

    if k_query > 1:
        distances = distances[:, 1:]  # exclude self

    mean_dist = np.mean(distances, axis=1)
    scale = np.clip(mean_dist / 2.0, 1e-6, 10.0)
    scales = np.stack([scale, scale, scale], axis=1).astype(np.float32)

    if is_tensor:
        return torch.from_numpy(scales).to(device)
    return scales


# Spherical Harmonics DC constant: C0 = 0.5 / sqrt(pi)
_C0 = 0.28209479177387814


def rgb_to_spherical_harmonics(
    rgb_colors: Union[np.ndarray, Tensor],
    degree: int = 0,
) -> Union[np.ndarray, Tensor]:
    """
    Convert RGB colors to spherical harmonic coefficients.

    Args:
        rgb_colors: RGB color values. Shape: (N, 3), range [0, 1]
        degree: SH degree (0=DC only, up to 3).

    Returns:
        sh_coeffs: Shape: (N, (degree+1)^2, 3)
    """
    n_points = rgb_colors.shape[0]
    num_coeffs = (degree + 1) ** 2

    if isinstance(rgb_colors, Tensor):
        sh_coeffs = torch.zeros(
            (n_points, num_coeffs, 3), dtype=torch.float32, device=rgb_colors.device
        )
        sh_coeffs[:, 0, :] = (rgb_colors - 0.5) / _C0
        return sh_coeffs
    else:
        sh_coeffs = np.zeros((n_points, num_coeffs, 3), dtype=np.float32)
        sh_coeffs[:, 0, :] = (rgb_colors - 0.5) / _C0
        return sh_coeffs


def spherical_harmonics_to_rgb(
    sh_coeffs: Union[np.ndarray, Tensor],
    view_directions: Optional[Union[np.ndarray, Tensor]] = None,
) -> Union[np.ndarray, Tensor]:
    """
    Convert SH coefficients back to RGB (DC term only).

    Args:
        sh_coeffs: Shape: (N, num_coeffs, 3)
        view_directions: Unused for degree-0. Shape: (N, 3) or None.

    Returns:
        rgb: Shape: (N, 3), clipped to [0, 1].
    """
    dc = sh_coeffs[:, 0, :]
    rgb = dc * _C0 + 0.5

    if isinstance(rgb, Tensor):
        return torch.clamp(rgb, 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)


def quaternion_to_rotation_matrix(
    quats: Union[np.ndarray, Tensor],
) -> Union[np.ndarray, Tensor]:
    """
    Convert quaternions (w, x, y, z) to 3x3 rotation matrices.

    Args:
        quats: Shape: (N, 4) or (4,)

    Returns:
        rot_matrices: Shape: (N, 3, 3) or (3, 3)
    """
    is_tensor = isinstance(quats, Tensor)
    single = quats.ndim == 1
    if single:
        quats = quats[None, :]

    if is_tensor:
        quats = quats / (torch.norm(quats, dim=-1, keepdim=True) + 1e-8)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        R = torch.stack(
            [
                torch.stack(
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    dim=-1,
                ),
                torch.stack(
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    dim=-1,
                ),
                torch.stack(
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    dim=-1,
                ),
            ],
            dim=1,
        )
        return R[0].float() if single else R.float()
    else:
        quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        R = np.stack(
            [
                np.stack(
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    axis=-1,
                ),
                np.stack(
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    axis=-1,
                ),
                np.stack(
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    axis=-1,
                ),
            ],
            axis=1,
        )
        return R[0].astype(np.float32) if single else R.astype(np.float32)


def sigmoid(x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """Sigmoid activation: logit-space to [0, 1]."""
    if isinstance(x, Tensor):
        return torch.sigmoid(x)
    return 1.0 / (1.0 + np.exp(-x))


def inverse_sigmoid(x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """Inverse sigmoid (logit): [0, 1] to logit-space."""
    if isinstance(x, Tensor):
        return torch.log(x / (1.0 - x))
    return np.log(x / (1.0 - x))


# =============================================================================
# Camera Utility Functions
# =============================================================================


def create_camera_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: str = "cpu",
) -> Tensor:
    """Create a 3x3 camera intrinsics matrix K."""
    return torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )


def create_front_camera_pose(device: str = "cpu") -> Tensor:
    """Create identity camera pose (camera at origin)."""
    return torch.eye(4, dtype=torch.float32, device=device)


def create_orbit_camera_pose(
    azimuth: float,
    elevation: float,
    radius: float,
    target: Tuple[float, float, float] = (0.0, 0.0, 1.5),
    device: str = "cpu",
) -> Tensor:
    """
    Create a camera pose orbiting around a target point.

    Args:
        azimuth: Horizontal rotation angle in radians.
        elevation: Vertical rotation angle in radians.
        radius: Distance from target point in meters.
        target: 3D point to orbit around.
        device: Torch device.

    Returns:
        viewmat: 4x4 world-to-camera transformation matrix.
    """
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)

    camera_pos = np.array([x, y, z]) + np.array(target)

    forward = np.array(target) - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    world_up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    R = np.stack([right, -up, forward], axis=1)

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R.T
    viewmat[:3, 3] = -R.T @ camera_pos

    return torch.tensor(viewmat, dtype=torch.float32, device=device)


def get_view_direction(means: Tensor, camera_pos: Tensor) -> Tensor:
    """
    Compute normalized view directions from Gaussian centers to camera.

    Args:
        means: Gaussian center positions. Shape: (N, 3)
        camera_pos: Camera position. Shape: (3,)

    Returns:
        view_dirs: Normalized direction vectors. Shape: (N, 3)
    """
    dirs = camera_pos.unsqueeze(0) - means
    return dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)
