"""
Gaussian Splatting Utility Functions

This module provides mathematical utilities for 3D Gaussian Splatting,
including depth unprojection, scale estimation, color space conversion,
and rotation representations.

All functions are designed to work with both NumPy arrays and PyTorch tensors
where applicable, with explicit type hints for clarity.
"""

from __future__ import annotations

from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import Tensor


def normalize_depth_to_metric(
    depth: Union[np.ndarray, Tensor],
    min_depth: float = 0.5,
    max_depth: float = 2.5,
) -> Union[np.ndarray, Tensor]:
    """
    Normalize relative depth to pseudo-metric range.

    This matches the normalization used in the original pointcloud.py:
    the depth is normalized to [0, 1] and then scaled to [min_depth, max_depth].

    Args:
        depth: Raw depth map (relative or inverse depth from MiDaS).
            Shape: (H, W)
            dtype: float32

        min_depth: Minimum depth value in output range (meters).
            Default: 0.5

        max_depth: Maximum depth value in output range (meters).
            Default: 2.5

    Returns:
        Normalized depth map in pseudo-metric range.
            Shape: (H, W)
            Range: [min_depth, max_depth]
    """
    is_tensor = isinstance(depth, Tensor)

    if is_tensor:
        d = depth.clone()
        d = d - torch.min(d[torch.isfinite(d)])
        denom = torch.max(d[torch.isfinite(d)]) + 1e-6
        d = d / denom
        z = min_depth + (max_depth - min_depth) * d
        return z
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

    Converts 2D pixel coordinates with depth values to 3D world coordinates
    using the standard pinhole camera unprojection equations.

    Args:
        depth: Depth map with metric or relative depth values.
            Shape: (H, W)
            dtype: float32
            Range: Positive values, where higher = farther from camera
            Units: Meters (for metric depth) or arbitrary (for relative)

        mask: Binary mask indicating which pixels to unproject.
            Shape: (H, W)
            dtype: bool, uint8, or float
            Values: Non-zero values indicate valid pixels

        fx: Focal length in x direction.
            Units: pixels
            Typical range: 500-2000 for standard cameras

        fy: Focal length in y direction.
            Units: pixels
            Typical range: 500-2000 for standard cameras

        cx: Principal point x coordinate (optical center).
            Units: pixels
            Typical value: width / 2

        cy: Principal point y coordinate (optical center).
            Units: pixels
            Typical value: height / 2

    Returns:
        xyz: 3D coordinates of valid (masked) pixels.
            Shape: (N, 3) where N = number of non-zero mask pixels
            dtype: Same as input depth (float32)
            Columns: [X, Y, Z] in camera coordinate system
                - X: Right (positive) / Left (negative)
                - Y: Down (positive) / Up (negative)
                - Z: Forward (positive, same as depth)

    Mathematical Operations:
        For each pixel (u, v) with depth d where mask[v, u] != 0:
            X = (u - cx) * d / fx
            Y = (v - cy) * d / fy
            Z = d

        This is the inverse of the projection:
            u = fx * X / Z + cx
            v = fy * Y / Z + cy

    Coordinate System:
        Camera coordinates (right-handed):
            - Origin: Camera optical center
            - X-axis: Right
            - Y-axis: Down
            - Z-axis: Forward (into the scene)

    Example:
        >>> depth = np.random.rand(480, 640).astype(np.float32) * 5  # 0-5 meters
        >>> mask = np.ones((480, 640), dtype=np.uint8)
        >>> xyz = depth_to_xyz(depth, mask, fx=525, fy=525, cx=320, cy=240)
        >>> print(xyz.shape)  # (307200, 3)

        >>> # With sparse mask
        >>> mask = np.zeros((480, 640), dtype=np.uint8)
        >>> mask[100:200, 150:250] = 1  # Only unproject a region
        >>> xyz = depth_to_xyz(depth, mask, fx=525, fy=525, cx=320, cy=240)
        >>> print(xyz.shape)  # (10000, 3) = 100 * 100 pixels
    """
    is_tensor = isinstance(depth, Tensor)

    if is_tensor:
        # PyTorch implementation
        h, w = depth.shape[:2]
        device = depth.device

        # Create meshgrid of pixel coordinates
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Get the Z values (depth)
        z = depth.clone()

        # Compute X and Y using pinhole model
        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        Z = z

        # Stack to (H, W, 3)
        xyz_full = torch.stack([X, Y, Z], dim=-1)

        # Apply mask
        mask_bool = mask.bool() if not mask.dtype == torch.bool else mask

        # Also filter out invalid depth (zero or negative)
        valid = mask_bool & (z > 0) & torch.isfinite(z)

        # Flatten and filter
        xyz_flat = xyz_full.reshape(-1, 3)
        valid_flat = valid.reshape(-1)

        xyz = xyz_flat[valid_flat]

        return xyz.float()

    else:
        # NumPy implementation
        h, w = depth.shape[:2]

        # Create meshgrid of pixel coordinates
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))

        # Get the Z values (depth)
        z = depth.copy()

        # Compute X and Y using pinhole model
        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        Z = z

        # Stack to (H, W, 3)
        xyz_full = np.stack([X, Y, Z], axis=-1)

        # Apply mask - convert to boolean
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask

        # Also filter out invalid depth (zero or negative)
        valid = mask_bool & (z > 0) & np.isfinite(z)

        # Flatten and filter
        xyz_flat = xyz_full.reshape(-1, 3)
        valid_flat = valid.reshape(-1)

        xyz = xyz_flat[valid_flat]

        return xyz.astype(np.float32)


def estimate_point_scales(
    xyz_points: Union[np.ndarray, Tensor],
    k_neighbors: int = 8,
) -> Union[np.ndarray, Tensor]:
    """
    Estimate initial Gaussian scales based on local point density.

    Computes an appropriate scale for each Gaussian by finding the average
    distance to k nearest neighbors. This ensures Gaussians are sized
    appropriately for the local point cloud density.

    Args:
        xyz_points: 3D point positions.
            Shape: (N, 3)
            dtype: float32
            Units: Meters (same as depth input)

        k_neighbors: Number of nearest neighbors to consider.
            Default: 8
            Range: 1-20 typically
            Higher values = smoother scale estimates

    Returns:
        scales: Estimated scale for each point.
            Shape: (N, 3) - isotropic scale repeated for x, y, z
            dtype: float32
            Units: Meters
            Range: Positive values

    Mathematical Operations:
        For each point p_i:
            1. Find k nearest neighbors: {p_j1, p_j2, ..., p_jk}
            2. Compute distances: d_j = ||p_i - p_j||
            3. Scale estimate: s_i = mean(d_j) / 2

        The division by 2 ensures neighboring Gaussians overlap
        appropriately (each Gaussian extends ~2 sigma).

    Performance Notes:
        - Uses scipy.spatial.KDTree for CPU arrays
        - Uses torch-cluster or manual computation for GPU tensors
        - Complexity: O(N log N) for KDTree construction + O(N * k) for queries

    Example:
        >>> xyz = np.random.rand(1000, 3).astype(np.float32)
        >>> scales = estimate_point_scales(xyz, k_neighbors=8)
        >>> print(scales.shape)  # (1000, 3)
        >>> print(scales.mean())  # ~0.05 for uniform [0,1] distribution
    """
    is_tensor = isinstance(xyz_points, Tensor)

    if is_tensor:
        device = xyz_points.device
        n_points = xyz_points.shape[0]

        # For CPU tensors or when pytorch3d not available, use scipy via numpy
        use_scipy = device.type == "cpu"

        if not use_scipy:
            try:
                from pytorch3d.ops import knn_points
            except ImportError:
                use_scipy = True

        if use_scipy:
            # Convert to numpy, use scipy, convert back
            from scipy.spatial import KDTree

            xyz_np = xyz_points.detach().cpu().numpy()

            if n_points == 0:
                return torch.zeros((0, 3), dtype=torch.float32, device=device)

            tree = KDTree(xyz_np)
            k_query = min(k_neighbors + 1, n_points)
            distances, _ = tree.query(xyz_np, k=k_query)

            if k_query > 1:
                distances_neighbors = distances[:, 1:]
            else:
                distances_neighbors = distances

            mean_dist = np.mean(distances_neighbors, axis=1)
            scale = mean_dist / 2.0
            scale = np.clip(scale, 1e-6, 10.0)
            scales = np.stack([scale, scale, scale], axis=1)

            return torch.from_numpy(scales.astype(np.float32)).to(device)

        else:
            # PyTorch implementation using pytorch3d.ops.knn_points
            from pytorch3d.ops import knn_points

            # knn_points expects (B, N, 3) format
            points = xyz_points.unsqueeze(0)  # (1, N, 3)

            # Find k+1 nearest neighbors (includes self at index 0)
            k_query = min(k_neighbors + 1, n_points)
            knn_result = knn_points(points, points, K=k_query, return_nn=False)

            # knn_result.dists is squared distances, shape (1, N, K)
            # First column is distance to self (should be 0)
            sq_dists = knn_result.dists[0]  # (N, K)

            # Take distances to neighbors (exclude self at index 0)
            if k_query > 1:
                sq_dists_neighbors = sq_dists[:, 1:]  # (N, K-1)
            else:
                # Edge case: only 1 point
                sq_dists_neighbors = sq_dists

            # Compute mean distance (take sqrt of squared distances)
            dists = torch.sqrt(sq_dists_neighbors + 1e-10)
            mean_dist = dists.mean(dim=1)  # (N,)

            # Scale estimate = mean_dist / 2
            scale = mean_dist / 2.0

            # Clamp to reasonable range
            scale = torch.clamp(scale, min=1e-6, max=10.0)

            # Repeat for isotropic (N, 3)
            scales = scale.unsqueeze(1).repeat(1, 3)

            return scales.float()

    else:
        # NumPy implementation using scipy KDTree
        from scipy.spatial import KDTree

        n_points = xyz_points.shape[0]

        if n_points == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Build KDTree
        tree = KDTree(xyz_points)

        # Query k+1 nearest neighbors (includes self)
        k_query = min(k_neighbors + 1, n_points)
        distances, _ = tree.query(xyz_points, k=k_query)

        # distances shape: (N, k_query)
        # First column is distance to self (0), exclude it
        if k_query > 1:
            distances_neighbors = distances[:, 1:]
        else:
            distances_neighbors = distances

        # Compute mean distance
        mean_dist = np.mean(distances_neighbors, axis=1)

        # Scale estimate = mean_dist / 2
        scale = mean_dist / 2.0

        # Clamp to reasonable range
        scale = np.clip(scale, 1e-6, 10.0)

        # Repeat for isotropic (N, 3)
        scales = np.stack([scale, scale, scale], axis=1)

        return scales.astype(np.float32)


def rgb_to_spherical_harmonics(
    rgb_colors: Union[np.ndarray, Tensor],
    degree: int = 0,
) -> Union[np.ndarray, Tensor]:
    """
    Convert RGB colors to spherical harmonic coefficients.

    For view-independent color (degree=0), this simply converts RGB to
    the DC (constant) term of spherical harmonics. Higher degrees enable
    view-dependent color effects.

    Args:
        rgb_colors: RGB color values.
            Shape: (N, 3)
            dtype: float32
            Range: [0, 1] normalized

        degree: Maximum spherical harmonic degree.
            0: DC only (1 coefficient per channel) - view-independent
            1: DC + 3 first-order (4 coefficients per channel)
            2: DC + first + second order (9 coefficients per channel)
            3: Full (16 coefficients per channel)
            Default: 0

    Returns:
        sh_coeffs: Spherical harmonic coefficients.
            Shape: (N, num_coeffs, 3) where num_coeffs = (degree + 1)^2
            dtype: float32
            Range: Unbounded (not normalized)

    Mathematical Operations:
        DC term (l=0, m=0):
            The zeroth-order SH is constant: Y_0^0 = 1 / (2 * sqrt(pi))
            To encode color c, we set: sh_dc = c / Y_0^0 = c * 2 * sqrt(pi)

        Higher orders encode view-dependent variations.

    Spherical Harmonic Basis:
        Degree 0: 1 function (constant)
        Degree 1: 3 functions (linear in x, y, z)
        Degree 2: 5 functions (quadratic)
        Degree 3: 7 functions (cubic)
        Total for degree d: (d+1)^2 functions

    Example:
        >>> rgb = np.array([[1.0, 0.0, 0.0],   # Red
        ...                  [0.0, 1.0, 0.0],   # Green
        ...                  [0.0, 0.0, 1.0]])  # Blue
        >>> sh = rgb_to_spherical_harmonics(rgb, degree=0)
        >>> print(sh.shape)  # (3, 1, 3)
    """
    # Spherical Harmonics DC constant: C0 = 0.5 / sqrt(pi) ≈ 0.28209
    # The zeroth-order SH basis function is: Y_0^0 = 1 / (2*sqrt(pi))
    # When evaluating SH: color = sh_dc * C0 + 0.5
    # To encode color: sh_dc = (color - 0.5) / C0
    # The 0.5 offset centers colors around zero for better optimization
    C0 = 0.28209479177387814

    is_tensor = isinstance(rgb_colors, Tensor)
    n_points = rgb_colors.shape[0]
    num_coeffs = (degree + 1) ** 2

    if is_tensor:
        device = rgb_colors.device

        # Initialize output with zeros: (N, num_coeffs, 3)
        sh_coeffs = torch.zeros((n_points, num_coeffs, 3), dtype=torch.float32, device=device)

        # Set DC term: convert RGB [0,1] to SH space
        # Following 3DGS convention: sh_dc = (rgb - 0.5) / C0
        sh_coeffs[:, 0, :] = (rgb_colors - 0.5) / C0

        return sh_coeffs

    else:
        # NumPy implementation
        sh_coeffs = np.zeros((n_points, num_coeffs, 3), dtype=np.float32)

        # Set DC term
        sh_coeffs[:, 0, :] = (rgb_colors - 0.5) / C0

        return sh_coeffs


def spherical_harmonics_to_rgb(
    sh_coeffs: Union[np.ndarray, Tensor],
    view_directions: Optional[Union[np.ndarray, Tensor]] = None,
) -> Union[np.ndarray, Tensor]:
    """
    Convert spherical harmonic coefficients back to RGB colors.

    Evaluates the spherical harmonic representation at given view directions
    to produce RGB colors. For degree-0 SH, the result is view-independent.

    Args:
        sh_coeffs: Spherical harmonic coefficients.
            Shape: (N, num_coeffs, 3)
            dtype: float32

        view_directions: Unit vectors pointing from surface to camera.
            Shape: (N, 3) or None
            dtype: float32
            Required if sh_coeffs has degree > 0
            Each row should be normalized (||v|| = 1)
            Default: None (assumes degree 0, returns DC color)

    Returns:
        rgb: RGB color values.
            Shape: (N, 3)
            dtype: float32
            Range: Clipped to [0, 1]

    Example:
        >>> sh = np.random.rand(100, 1, 3).astype(np.float32)
        >>> rgb = spherical_harmonics_to_rgb(sh)
        >>> print(rgb.shape)  # (100, 3)
    """
    C0 = 0.28209479177387814

    is_tensor = isinstance(sh_coeffs, Tensor)

    # Get DC term (first coefficient)
    dc = sh_coeffs[:, 0, :]  # (N, 3)

    # Convert back to RGB: rgb = dc * C0 + 0.5
    rgb = dc * C0 + 0.5

    # Clamp to [0, 1]
    if is_tensor:
        rgb = torch.clamp(rgb, 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    return rgb


def quaternion_to_rotation_matrix(
    quats: Union[np.ndarray, Tensor],
) -> Union[np.ndarray, Tensor]:
    """
    Convert quaternions to 3x3 rotation matrices.

    Uses the Hamilton convention with quaternion format (w, x, y, z)
    where w is the scalar component.

    Args:
        quats: Quaternions in (w, x, y, z) format.
            Shape: (N, 4) or (4,)
            dtype: float32
            Should be normalized (||q|| = 1), but will work with unnormalized

    Returns:
        rot_matrices: 3x3 rotation matrices.
            Shape: (N, 3, 3) or (3, 3)
            dtype: float32
            Orthogonal matrices with det = +1

    Mathematical Operations:
        For quaternion q = (w, x, y, z):

        R = [1 - 2(y² + z²),    2(xy - wz),      2(xz + wy)    ]
            [2(xy + wz),        1 - 2(x² + z²),  2(yz - wx)    ]
            [2(xz - wy),        2(yz + wx),      1 - 2(x² + y²)]

    Conventions:
        - Hamilton convention: q = w + xi + yj + zk
        - Right-handed coordinate system
        - Active rotation (rotates vectors, not coordinate frame)

    Example:
        >>> # Identity rotation
        >>> q = np.array([[1.0, 0.0, 0.0, 0.0]])
        >>> R = quaternion_to_rotation_matrix(q)
        >>> print(R)  # 3x3 identity matrix

        >>> # 90-degree rotation around Z-axis
        >>> q = np.array([[np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]])
        >>> R = quaternion_to_rotation_matrix(q)
    """
    is_tensor = isinstance(quats, Tensor)
    single = quats.ndim == 1

    if single:
        quats = quats[None, :]  # Add batch dimension

    if is_tensor:
        # Normalize quaternions
        quats = quats / (torch.norm(quats, dim=-1, keepdim=True) + 1e-8)

        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        # Compute rotation matrix elements
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)

        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - w * x)

        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)

        # Stack into (N, 3, 3)
        R = torch.stack(
            [
                torch.stack([r00, r01, r02], dim=-1),
                torch.stack([r10, r11, r12], dim=-1),
                torch.stack([r20, r21, r22], dim=-1),
            ],
            dim=1,
        )

        if single:
            R = R[0]

        return R.float()

    else:
        # NumPy implementation
        quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)

        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)

        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - w * x)

        r20 = 2 * (x * z - w * y)
        r21 = 2 * (y * z + w * x)
        r22 = 1 - 2 * (x * x + y * y)

        R = np.stack(
            [
                np.stack([r00, r01, r02], axis=-1),
                np.stack([r10, r11, r12], axis=-1),
                np.stack([r20, r21, r22], axis=-1),
            ],
            axis=1,
        )

        if single:
            R = R[0]

        return R.astype(np.float32)


def rotation_matrix_to_quaternion(
    rot_matrices: Union[np.ndarray, Tensor],
) -> Union[np.ndarray, Tensor]:
    """
    Convert 3x3 rotation matrices to quaternions.

    Inverse of quaternion_to_rotation_matrix. Uses Shepperd's method
    for numerical stability.

    Args:
        rot_matrices: 3x3 rotation matrices.
            Shape: (N, 3, 3) or (3, 3)
            dtype: float32
            Should be orthogonal with det = +1

    Returns:
        quats: Quaternions in (w, x, y, z) format.
            Shape: (N, 4) or (4,)
            dtype: float32
            Normalized (||q|| = 1)

    Example:
        >>> R = np.eye(3, dtype=np.float32)
        >>> q = rotation_matrix_to_quaternion(R)
        >>> print(q)  # [1, 0, 0, 0]
    """
    # TODO: Implement rotation matrix to quaternion conversion
    raise NotImplementedError("rotation_matrix_to_quaternion not yet implemented")


def build_covariance_3d(
    scales: Union[np.ndarray, Tensor],
    rotations: Union[np.ndarray, Tensor],
) -> Union[np.ndarray, Tensor]:
    """
    Build 3D covariance matrices from scales and rotations.

    Each 3D Gaussian is parameterized by axis-aligned scales and a rotation.
    The covariance matrix is: Sigma = R @ S @ S.T @ R.T
    where S = diag(scales) and R is the rotation matrix.

    Args:
        scales: Per-Gaussian axis scales.
            Shape: (N, 3)
            dtype: float32
            Units: Meters (standard deviation along each axis)

        rotations: Per-Gaussian rotation quaternions (w, x, y, z).
            Shape: (N, 4)
            dtype: float32
            Should be normalized

    Returns:
        covariances: 3x3 covariance matrices.
            Shape: (N, 3, 3)
            dtype: float32
            Symmetric positive semi-definite matrices

    Mathematical Operations:
        For each Gaussian i:
            S_i = diag(scales[i])                    # (3, 3) diagonal
            R_i = quaternion_to_rotation_matrix(rotations[i])  # (3, 3)
            Sigma_i = R_i @ S_i @ S_i.T @ R_i.T      # (3, 3)

    Example:
        >>> scales = np.array([[0.1, 0.1, 0.1]])  # Isotropic
        >>> rotations = np.array([[1, 0, 0, 0]])  # Identity
        >>> cov = build_covariance_3d(scales, rotations)
        >>> print(cov)  # diag([0.01, 0.01, 0.01])
    """
    # TODO: Implement 3D covariance construction
    raise NotImplementedError("build_covariance_3d not yet implemented")


def project_covariance_to_2d(
    cov_3d: Union[np.ndarray, Tensor],
    means_cam: Union[np.ndarray, Tensor],
    fx: float,
    fy: float,
) -> Union[np.ndarray, Tensor]:
    """
    Project 3D covariance to 2D image plane covariance.

    Uses the Jacobian of the projection to transform 3D Gaussian
    covariances to their 2D equivalents for rasterization.

    Args:
        cov_3d: 3D covariance matrices.
            Shape: (N, 3, 3)
            dtype: float32

        means_cam: Gaussian centers in camera coordinates.
            Shape: (N, 3)
            dtype: float32
            [X, Y, Z] where Z > 0 (in front of camera)

        fx: Focal length x (pixels).
        fy: Focal length y (pixels).

    Returns:
        cov_2d: 2D covariance matrices.
            Shape: (N, 2, 2)
            dtype: float32

    Mathematical Operations:
        Jacobian of perspective projection at point (X, Y, Z):
            J = [fx/Z,  0,    -fx*X/Z²]
                [0,     fy/Z, -fy*Y/Z²]

        2D covariance: Sigma_2d = J @ Sigma_3d @ J.T

    Example:
        >>> cov_3d = np.eye(3).reshape(1, 3, 3) * 0.01
        >>> means = np.array([[0, 0, 2.0]])  # 2 meters in front
        >>> cov_2d = project_covariance_to_2d(cov_3d, means, fx=500, fy=500)
    """
    # TODO: Implement 2D covariance projection
    raise NotImplementedError("project_covariance_to_2d not yet implemented")


def sigmoid(x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Sigmoid activation function.

    Used to convert logit-space opacity to [0, 1] range.

    Args:
        x: Input values in logit space.
            Shape: Any
            dtype: float32
            Range: (-inf, inf)

    Returns:
        Output values.
            Shape: Same as input
            Range: (0, 1)
    """
    if isinstance(x, Tensor):
        return torch.sigmoid(x)
    return 1.0 / (1.0 + np.exp(-x))


def inverse_sigmoid(x: Union[np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:
    """
    Inverse sigmoid (logit) function.

    Used to convert [0, 1] opacity to logit space for optimization.

    Args:
        x: Input values.
            Shape: Any
            dtype: float32
            Range: (0, 1) exclusive

    Returns:
        Output values in logit space.
            Shape: Same as input
            Range: (-inf, inf)
    """
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
    """
    Create a 3x3 camera intrinsics matrix K.

    Args:
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).
        cx: Principal point x (pixels).
        cy: Principal point y (pixels).
        device: Torch device.

    Returns:
        K: Camera intrinsics matrix.
            Shape: (3, 3)
            Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    K = torch.tensor(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )
    return K


def create_front_camera_pose(
    device: str = "cpu",
) -> Tensor:
    """
    Create identity camera pose (camera at origin looking down -Z).

    This is the default pose for rendering the front view, matching
    how the depth was captured (camera facing the subject).

    Args:
        device: Torch device.

    Returns:
        viewmat: 4x4 world-to-camera transformation matrix.
            Shape: (4, 4)
            Identity matrix (no transformation).

    Coordinate System:
        Camera coordinates:
            - X: Right
            - Y: Down
            - Z: Forward (into the scene)
    """
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

    Useful for generating novel views by rotating the camera around
    the reconstructed subject.

    Args:
        azimuth: Horizontal rotation angle in radians.
            0 = front, pi/2 = left side, pi = back, -pi/2 = right side

        elevation: Vertical rotation angle in radians.
            0 = eye level, pi/4 = looking down from above, -pi/4 = looking up

        radius: Distance from target point in meters.

        target: 3D point to orbit around (x, y, z).
            Default: (0, 0, 1.5) - center of typical human subject

        device: Torch device.

    Returns:
        viewmat: 4x4 world-to-camera transformation matrix.
            Shape: (4, 4)

    Example:
        >>> # Front view
        >>> pose_front = create_orbit_camera_pose(0, 0, 2.0)

        >>> # Left side view (90 degrees)
        >>> pose_left = create_orbit_camera_pose(np.pi/2, 0, 2.0)

        >>> # View from above
        >>> pose_above = create_orbit_camera_pose(0, np.pi/4, 2.0)
    """
    # Camera position on sphere
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)

    camera_pos = np.array([x, y, z]) + np.array(target)

    # Look-at target
    forward = np.array(target) - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    # Up vector (world Y is down, so use -Y as up)
    world_up = np.array([0.0, -1.0, 0.0])

    # Right vector
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)

    # Recompute up to ensure orthogonality
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    # Build rotation matrix (camera axes in world coordinates)
    # Note: camera looks along +Z, so forward = +Z
    R = np.stack([right, -up, forward], axis=1)  # (3, 3)

    # Build 4x4 transformation matrix (world to camera)
    # R^T @ (p - t) = camera coordinates
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R.T
    viewmat[:3, 3] = -R.T @ camera_pos

    return torch.tensor(viewmat, dtype=torch.float32, device=device)


def get_view_direction(
    means: Tensor,
    camera_pos: Tensor,
) -> Tensor:
    """
    Compute view directions from Gaussian centers to camera.

    Used for evaluating spherical harmonics at the correct view angle.

    Args:
        means: Gaussian center positions.
            Shape: (N, 3)

        camera_pos: Camera position in world coordinates.
            Shape: (3,)

    Returns:
        view_dirs: Normalized view direction vectors.
            Shape: (N, 3)
            Points from each Gaussian toward the camera.
    """
    # Direction from each Gaussian to camera
    dirs = camera_pos.unsqueeze(0) - means  # (N, 3)

    # Normalize
    dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)

    return dirs
