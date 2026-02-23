"""
Minimal test script for gsplat rasterization.

Tests that gsplat is installed correctly and can render random Gaussians.
"""

import torch
import numpy as np


def test_gsplat_basic():
    """Test basic gsplat rasterization with random Gaussians."""
    try:
        from gsplat import rasterization
    except ImportError as e:
        print(f"Failed to import gsplat: {e}")
        return False

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create random Gaussians
    N = 100  # Number of Gaussians
    means = torch.randn(N, 3, device=device) * 0.5  # Random positions
    means[:, 2] += 2.0  # Shift to be in front of camera

    # Random quaternions (wxyz format) - need to normalize
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    # Random scales (positive)
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01

    # Random opacities
    opacities = torch.rand(N, device=device) * 0.5 + 0.5

    # Random colors (RGB, not SH)
    colors = torch.rand(N, 3, device=device)

    # Camera parameters
    H, W = 256, 256
    fx, fy = 500.0, 500.0
    cx, cy = W / 2, H / 2

    # Camera intrinsics matrix K (3x3)
    Ks = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=torch.float32, device=device).unsqueeze(0)  # (1, 3, 3)

    # View matrix (identity - camera at origin looking down -Z)
    viewmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4, 4)

    print(f"Means shape: {means.shape}")
    print(f"Quats shape: {quats.shape}")
    print(f"Scales shape: {scales.shape}")
    print(f"Opacities shape: {opacities.shape}")
    print(f"Colors shape: {colors.shape}")
    print(f"Ks shape: {Ks.shape}")
    print(f"Viewmats shape: {viewmats.shape}")

    # Rasterize
    try:
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
            sh_degree=None,  # Not using SH, just direct RGB
        )

        print("\nRasterization successful!")
        print(f"Rendered colors shape: {render_colors.shape}")
        print(f"Rendered alphas shape: {render_alphas.shape}")
        print(f"Color range: [{render_colors.min():.3f}, {render_colors.max():.3f}]")
        print(f"Alpha range: [{render_alphas.min():.3f}, {render_alphas.max():.3f}]")

        # Save test image
        import cv2
        rgb = render_colors[0].cpu().numpy()  # (H, W, 3)
        rgb = np.clip(rgb, 0, 1)
        bgr = (rgb[:, :, ::-1] * 255).astype(np.uint8)
        cv2.imwrite("C:/Users/Ashritha/Desktop/Human3D/test_gsplat_output.png", bgr)
        print("\nSaved test image to test_gsplat_output.png")

        return True

    except Exception as e:
        print(f"Rasterization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gsplat_with_sh():
    """Test gsplat rasterization with spherical harmonics."""
    try:
        from gsplat import rasterization
    except ImportError as e:
        print(f"Failed to import gsplat: {e}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTesting SH rendering on device: {device}")

    # Create random Gaussians
    N = 100
    means = torch.randn(N, 3, device=device) * 0.5
    means[:, 2] += 2.0

    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01
    opacities = torch.rand(N, device=device) * 0.5 + 0.5

    # SH coefficients for degree 0 (DC only): shape (N, 1, 3)
    # For gsplat, colors with SH should be (N, K, 3) where K = (degree+1)^2
    sh_degree = 0
    K = (sh_degree + 1) ** 2  # = 1 for degree 0
    sh_coeffs = torch.randn(N, K, 3, device=device) * 0.5

    H, W = 256, 256
    fx, fy = 500.0, 500.0
    cx, cy = W / 2, H / 2

    Ks = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=torch.float32, device=device).unsqueeze(0)

    viewmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)

    try:
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_coeffs,  # SH coefficients
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB",
            sh_degree=sh_degree,  # Tell gsplat we're using SH
        )

        print("SH Rasterization successful!")
        print(f"Rendered colors shape: {render_colors.shape}")
        print(f"Color range: [{render_colors.min():.3f}, {render_colors.max():.3f}]")

        return True

    except Exception as e:
        print(f"SH Rasterization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Testing gsplat basic rasterization")
    print("=" * 50)
    success1 = test_gsplat_basic()

    print("\n" + "=" * 50)
    print("Testing gsplat with spherical harmonics")
    print("=" * 50)
    success2 = test_gsplat_with_sh()

    print("\n" + "=" * 50)
    print(f"Results: Basic={'PASS' if success1 else 'FAIL'}, SH={'PASS' if success2 else 'FAIL'}")
    print("=" * 50)
