"""
PLY Exporter for Gaussian Splatting

This module provides functionality to export 3D Gaussian splats to PLY format,
compatible with standard Gaussian Splatting viewers and the original
3D Gaussian Splatting implementation.

The PLY format stores per-Gaussian attributes:
- Position (xyz)
- Spherical harmonic coefficients (DC + optional higher orders)
- Opacity
- Scale (3D log-scale)
- Rotation (quaternion)

References:
    - Original 3DGS PLY format: https://github.com/graphdeco-inria/gaussian-splatting
    - PLY specification: http://paulbourke.net/dataformats/ply/
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Optional

import numpy as np


def save_gaussian_ply(
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    sh_coeffs: np.ndarray,
    opacities: np.ndarray,
    filepath: Union[str, Path],
    extra_attributes: Optional[dict] = None,
) -> None:
    """
    Save Gaussian splats to PLY file format.

    Exports trained Gaussian parameters to a binary PLY file compatible
    with standard 3D Gaussian Splatting viewers and renderers.

    Args:
        means: Gaussian center positions in world coordinates.
            Shape: (N, 3)
            dtype: float32
            Units: Meters
            Columns: [x, y, z]

        scales: Log-scale parameters for each axis.
            Shape: (N, 3)
            dtype: float32
            Range: Typically [-10, 2] in log-space
            Note: Actual scale = exp(scales)
            Columns: [scale_x, scale_y, scale_z]

        rotations: Rotation quaternions in (w, x, y, z) format.
            Shape: (N, 4)
            dtype: float32
            Should be normalized (||q|| = 1)
            Columns: [rot_w, rot_x, rot_y, rot_z]

        sh_coeffs: Spherical harmonic coefficients for color.
            Shape: (N, num_coeffs, 3) where num_coeffs = (degree+1)²
            dtype: float32
            For degree 0: shape (N, 1, 3) - DC term only
            For degree 3: shape (N, 16, 3) - full SH

            The DC term (sh_coeffs[:, 0, :]) encodes base color.
            Higher-order terms encode view-dependent effects.

        opacities: Opacity values after sigmoid activation.
            Shape: (N,) or (N, 1)
            dtype: float32
            Range: [0, 1] (already activated, not logits)

        filepath: Output path for the PLY file.
            Will create parent directories if needed.
            Should have .ply extension.

        extra_attributes: Optional dictionary of additional per-Gaussian attributes.
            Keys: Attribute names (will be prefixed with "extra_")
            Values: np.ndarray of shape (N,) or (N, k)
            Default: None

    Raises:
        ValueError: If input shapes are incompatible.
        IOError: If file cannot be written.

    File Format:
        The PLY file contains a header followed by binary vertex data.

        Header example:
            ply
            format binary_little_endian 1.0
            element vertex <N>
            property float x
            property float y
            property float z
            property float f_dc_0
            property float f_dc_1
            property float f_dc_2
            property float f_rest_0
            ... (up to f_rest_44 for degree 3)
            property float opacity
            property float scale_0
            property float scale_1
            property float scale_2
            property float rot_0
            property float rot_1
            property float rot_2
            property float rot_3
            end_header

        Each vertex (Gaussian) is stored as a contiguous block of float32 values.

    Compatibility:
        The output format is compatible with:
        - Original 3DGS SIBR viewer
        - gsplat library
        - antimatter15/splat web viewer
        - SuperSplat
        - Luma AI

    Example:
        >>> # Export trained Gaussians
        >>> means = np.random.randn(1000, 3).astype(np.float32)
        >>> scales = np.random.randn(1000, 3).astype(np.float32) * 0.1
        >>> rotations = np.zeros((1000, 4), dtype=np.float32)
        >>> rotations[:, 0] = 1.0  # Identity quaternions
        >>> sh_coeffs = np.random.randn(1000, 1, 3).astype(np.float32) * 0.5
        >>> opacities = np.random.rand(1000).astype(np.float32)
        >>> save_gaussian_ply(means, scales, rotations, sh_coeffs, opacities,
        ...                   "output/gaussians.ply")

        >>> # With higher-degree SH
        >>> sh_coeffs_full = np.random.randn(1000, 16, 3).astype(np.float32) * 0.1
        >>> save_gaussian_ply(means, scales, rotations, sh_coeffs_full, opacities,
        ...                   "output/gaussians_sh3.ply")
    """
    from plyfile import PlyData, PlyElement

    # 1. Validate inputs
    n_gaussians = validate_gaussian_attributes(means, scales, rotations, sh_coeffs, opacities)

    # 2. Create output directory if needed
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 3. Prepare data - ensure float32 and correct shapes
    means = means.astype(np.float32)
    scales = scales.astype(np.float32)
    rotations = rotations.astype(np.float32)
    sh_coeffs = sh_coeffs.astype(np.float32)

    # Flatten opacities
    if opacities.ndim == 2:
        opacities = opacities.squeeze(-1)
    opacities = opacities.astype(np.float32)

    # Normalize quaternions
    quat_norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / (quat_norm + 1e-8)

    # 4. Determine SH degree and prepare SH data
    num_sh_coeffs = sh_coeffs.shape[1]
    sh_degree = int(np.sqrt(num_sh_coeffs)) - 1

    # Flatten SH coefficients: (N, num_coeffs, 3) -> separate DC and rest
    # DC term: first coefficient, 3 channels
    f_dc = sh_coeffs[:, 0, :]  # (N, 3)

    # Rest terms: remaining coefficients, 3 channels each
    if num_sh_coeffs > 1:
        f_rest = sh_coeffs[:, 1:, :].reshape(n_gaussians, -1)  # (N, (num_coeffs-1)*3)
    else:
        f_rest = np.zeros((n_gaussians, 0), dtype=np.float32)

    # 5. Build structured array dtype
    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),  # Normals (set to 0)
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]

    # Add f_rest properties
    num_rest = f_rest.shape[1]
    for i in range(num_rest):
        dtype_list.append((f"f_rest_{i}", "f4"))

    # Add opacity, scale, rotation
    dtype_list.append(("opacity", "f4"))
    dtype_list.extend(
        [
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]
    )

    # Add extra attributes
    if extra_attributes:
        for name, arr in extra_attributes.items():
            if arr.ndim == 1:
                dtype_list.append((f"extra_{name}", "f4"))
            else:
                for i in range(arr.shape[1]):
                    dtype_list.append((f"extra_{name}_{i}", "f4"))

    # 6. Create structured array
    vertices = np.zeros(n_gaussians, dtype=dtype_list)

    # Fill in data
    vertices["x"] = means[:, 0]
    vertices["y"] = means[:, 1]
    vertices["z"] = means[:, 2]

    # Normals (set to 0, not used by most viewers)
    vertices["nx"] = 0
    vertices["ny"] = 0
    vertices["nz"] = 0

    # DC color terms
    vertices["f_dc_0"] = f_dc[:, 0]
    vertices["f_dc_1"] = f_dc[:, 1]
    vertices["f_dc_2"] = f_dc[:, 2]

    # Rest SH terms
    for i in range(num_rest):
        vertices[f"f_rest_{i}"] = f_rest[:, i]

    # Opacity
    vertices["opacity"] = opacities

    # Scales (keep in log-space as per 3DGS format)
    vertices["scale_0"] = scales[:, 0]
    vertices["scale_1"] = scales[:, 1]
    vertices["scale_2"] = scales[:, 2]

    # Rotations (wxyz format)
    vertices["rot_0"] = rotations[:, 0]  # w
    vertices["rot_1"] = rotations[:, 1]  # x
    vertices["rot_2"] = rotations[:, 2]  # y
    vertices["rot_3"] = rotations[:, 3]  # z

    # Extra attributes
    if extra_attributes:
        for name, arr in extra_attributes.items():
            if arr.ndim == 1:
                vertices[f"extra_{name}"] = arr
            else:
                for i in range(arr.shape[1]):
                    vertices[f"extra_{name}_{i}"] = arr[:, i]

    # 7. Write PLY file
    el = PlyElement.describe(vertices, "vertex")
    PlyData([el], text=False).write(str(filepath))


def load_gaussian_ply(
    filepath: Union[str, Path],
) -> dict:
    """
    Load Gaussian splats from PLY file.

    Reads a PLY file in the 3D Gaussian Splatting format and returns
    all Gaussian parameters as NumPy arrays.

    Args:
        filepath: Path to the PLY file.
            Must exist and be readable.

    Returns:
        dict: Dictionary containing Gaussian parameters:
            - 'means': (N, 3) float32 - positions
            - 'scales': (N, 3) float32 - log-scales
            - 'rotations': (N, 4) float32 - quaternions (wxyz)
            - 'sh_coeffs': (N, num_coeffs, 3) float32 - SH coefficients
            - 'opacities': (N,) float32 - opacity values
            - 'num_gaussians': int - total count

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is invalid.

    Example:
        >>> data = load_gaussian_ply("trained_model/point_cloud.ply")
        >>> print(f"Loaded {data['num_gaussians']} Gaussians")
        >>> print(f"Position range: {data['means'].min():.2f} to {data['means'].max():.2f}")
    """
    from plyfile import PlyData

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PLY file not found: {filepath}")

    # Read PLY file
    plydata = PlyData.read(str(filepath))
    vertex = plydata["vertex"]
    n_gaussians = len(vertex)

    # Extract positions
    means = np.stack(
        [
            vertex["x"],
            vertex["y"],
            vertex["z"],
        ],
        axis=1,
    ).astype(np.float32)

    # Extract scales
    scales = np.stack(
        [
            vertex["scale_0"],
            vertex["scale_1"],
            vertex["scale_2"],
        ],
        axis=1,
    ).astype(np.float32)

    # Extract rotations
    rotations = np.stack(
        [
            vertex["rot_0"],
            vertex["rot_1"],
            vertex["rot_2"],
            vertex["rot_3"],
        ],
        axis=1,
    ).astype(np.float32)

    # Extract opacities
    opacities = np.array(vertex["opacity"]).astype(np.float32)

    # Extract SH coefficients
    # First, get DC terms
    f_dc = np.stack(
        [
            vertex["f_dc_0"],
            vertex["f_dc_1"],
            vertex["f_dc_2"],
        ],
        axis=1,
    ).astype(
        np.float32
    )  # (N, 3)

    # Count f_rest properties
    # vertex.data.dtype.names gives the property names
    property_names = vertex.data.dtype.names
    f_rest_names = [p for p in property_names if p.startswith("f_rest_")]
    num_rest = len(f_rest_names)

    if num_rest > 0:
        # Sort by index
        f_rest_names = sorted(f_rest_names, key=lambda x: int(x.split("_")[-1]))
        f_rest = np.stack([vertex[name] for name in f_rest_names], axis=1).astype(np.float32)
        # Reshape: (N, num_rest) -> (N, num_rest//3, 3)
        f_rest = f_rest.reshape(n_gaussians, -1, 3)
        # Combine DC and rest
        sh_coeffs = np.concatenate([f_dc[:, np.newaxis, :], f_rest], axis=1)
    else:
        sh_coeffs = f_dc[:, np.newaxis, :]  # (N, 1, 3)

    return {
        "means": means,
        "scales": scales,
        "rotations": rotations,
        "sh_coeffs": sh_coeffs,
        "opacities": opacities,
        "num_gaussians": n_gaussians,
    }


def construct_ply_header(
    num_vertices: int,
    sh_degree: int = 0,
    extra_properties: Optional[list] = None,
) -> str:
    """
    Construct PLY header string for Gaussian splat data.

    Args:
        num_vertices: Number of Gaussians (vertices).
        sh_degree: Spherical harmonic degree (0-3).
            0: 1 coefficient (DC only)
            1: 4 coefficients
            2: 9 coefficients
            3: 16 coefficients (full)
        extra_properties: List of (name, dtype) tuples for extra attributes.

    Returns:
        str: Complete PLY header including "end_header\n".

    Example:
        >>> header = construct_ply_header(1000, sh_degree=0)
        >>> print(header[:50])  # "ply\nformat binary_little_endian 1.0\n..."
    """
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]

    # Add f_rest properties based on SH degree
    num_sh_coeffs = get_sh_coefficient_count(sh_degree)
    num_rest = (num_sh_coeffs - 1) * 3  # -1 for DC, *3 for RGB
    for i in range(num_rest):
        lines.append(f"property float f_rest_{i}")

    # Add remaining properties
    lines.extend(
        [
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
        ]
    )

    # Add extra properties
    if extra_properties:
        for name, dtype in extra_properties:
            lines.append(f"property {dtype} {name}")

    lines.append("end_header")

    return "\n".join(lines) + "\n"


def get_sh_coefficient_count(degree: int) -> int:
    """
    Get the number of spherical harmonic coefficients for a given degree.

    Args:
        degree: SH degree (0-3).

    Returns:
        int: Number of coefficients = (degree + 1)²
            degree 0: 1
            degree 1: 4
            degree 2: 9
            degree 3: 16
    """
    return (degree + 1) ** 2


def validate_gaussian_attributes(
    means: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    sh_coeffs: np.ndarray,
    opacities: np.ndarray,
) -> int:
    """
    Validate shapes and types of Gaussian attributes.

    Args:
        means: Position array to validate.
        scales: Scale array to validate.
        rotations: Rotation quaternion array to validate.
        sh_coeffs: SH coefficient array to validate.
        opacities: Opacity array to validate.

    Returns:
        int: Number of Gaussians (N) if valid.

    Raises:
        ValueError: If any array has incorrect shape or type.
    """
    # Check means
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"means must have shape (N, 3), got {means.shape}")
    n_gaussians = means.shape[0]

    # Check scales
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError(f"scales must have shape (N, 3), got {scales.shape}")
    if scales.shape[0] != n_gaussians:
        raise ValueError(f"scales has {scales.shape[0]} rows, expected {n_gaussians}")

    # Check rotations
    if rotations.ndim != 2 or rotations.shape[1] != 4:
        raise ValueError(f"rotations must have shape (N, 4), got {rotations.shape}")
    if rotations.shape[0] != n_gaussians:
        raise ValueError(f"rotations has {rotations.shape[0]} rows, expected {n_gaussians}")

    # Check sh_coeffs
    if sh_coeffs.ndim != 3 or sh_coeffs.shape[2] != 3:
        raise ValueError(f"sh_coeffs must have shape (N, num_coeffs, 3), got {sh_coeffs.shape}")
    if sh_coeffs.shape[0] != n_gaussians:
        raise ValueError(f"sh_coeffs has {sh_coeffs.shape[0]} rows, expected {n_gaussians}")

    # Validate SH degree
    num_coeffs = sh_coeffs.shape[1]
    valid_counts = [1, 4, 9, 16]  # (degree+1)^2 for degrees 0-3
    if num_coeffs not in valid_counts:
        raise ValueError(f"sh_coeffs has {num_coeffs} coefficients, expected one of {valid_counts}")

    # Check opacities
    if opacities.ndim == 2:
        opacities = opacities.squeeze(-1)
    if opacities.ndim != 1:
        raise ValueError(
            f"opacities must have shape (N,) or (N, 1), got shape with {opacities.ndim} dims"
        )
    if opacities.shape[0] != n_gaussians:
        raise ValueError(f"opacities has {opacities.shape[0]} elements, expected {n_gaussians}")

    return n_gaussians
