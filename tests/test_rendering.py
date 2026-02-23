"""
Test script for Gaussian Splatting rendering pipeline.

Tests that we can:
1. Initialize Gaussians from synthetic data
2. Render a front view
3. Save the rendered image
"""

import sys
from pathlib import Path

import numpy as np
import cv2

from human3d.reconstruct.gaussian_trainer import (
    GaussianTrainer,
    GaussianConfig,
    CameraParams,
)


def create_test_image_with_depth():
    """Create a synthetic test image with depth and mask."""
    H, W = 256, 256

    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    rgb[: H // 2, : W // 2] = [255, 0, 0]
    rgb[: H // 2, W // 2 :] = [0, 255, 0]
    rgb[H // 2 :, : W // 2] = [0, 0, 255]
    rgb[H // 2 :, W // 2 :] = [255, 255, 0]

    for y in range(H):
        for x in range(W):
            factor = 0.5 + 0.5 * (x / W)
            rgb[y, x] = (rgb[y, x] * factor).astype(np.uint8)

    depth = np.full((H, W), 2.0, dtype=np.float32)
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    depth = depth - (0.5 * np.exp(-(dist**2) / (2 * 50**2)))

    radius = min(H, W) // 3
    mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).astype(np.uint8)

    return rgb, depth, mask


def test_render_view():
    """Test rendering from initialized Gaussians."""
    print("=" * 60)
    print("Testing Gaussian Splatting Rendering")
    print("=" * 60)

    rgb, depth, mask = create_test_image_with_depth()
    H, W = rgb.shape[:2]

    print(f"\n1. Image size: {W}x{H}, Masked pixels: {mask.sum()}")

    camera = CameraParams(
        fx=500.0,
        fy=500.0,
        cx=(W - 1) / 2.0,
        cy=(H - 1) / 2.0,
        width=W,
        height=H,
    )

    config = GaussianConfig(sh_degree=0, opacity_init=0.9, position_noise=0.0)

    trainer = GaussianTrainer(
        rgb=rgb,
        depth=depth,
        mask=mask,
        camera_params=camera,
        config=config,
        device="cpu",
    )

    num_gaussians = trainer.initialize_gaussians()
    print(f"2. Initialized {num_gaussians} Gaussians")

    rendered_rgb, rendered_depth, rendered_alpha = trainer.render_view()

    print(f"3. Rendered RGB shape: {rendered_rgb.shape}")
    print(f"   RGB range: [{rendered_rgb.min():.3f}, {rendered_rgb.max():.3f}]")
    print(f"   Alpha range: [{rendered_alpha.min():.3f}, {rendered_alpha.max():.3f}]")

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    rgb_np = rendered_rgb.detach().cpu().numpy()
    rgb_np = np.clip(rgb_np, 0, 1)
    bgr_np = (rgb_np[:, :, ::-1] * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "test_render.png"), bgr_np)

    cv2.imwrite(
        str(output_dir / "test_original.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(str(output_dir / "test_mask.png"), mask * 255)

    alpha_np = rendered_alpha.detach().cpu().numpy()
    cv2.imwrite(
        str(output_dir / "test_render_alpha.png"), (alpha_np * 255).astype(np.uint8)
    )

    depth_np = rendered_depth.detach().cpu().numpy()
    depth_valid = depth_np[depth_np < float("inf")]
    if len(depth_valid) > 0:
        depth_vis = (depth_np - depth_valid.min()) / (
            depth_valid.max() - depth_valid.min() + 1e-6
        )
        depth_vis = np.clip(depth_vis, 0, 1)
        depth_vis = cv2.applyColorMap(
            (depth_vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        cv2.imwrite(str(output_dir / "test_render_depth.png"), depth_vis)

    print(f"\n4. Outputs saved to: {output_dir}")
    print("=" * 60)

    return True


def test_render_with_real_image():
    """Test rendering with a real image if available."""
    sample_paths = [
        Path(__file__).parent.parent / "samples" / "test.jpg",
        Path(__file__).parent.parent / "samples" / "test.png",
        Path(__file__).parent.parent / "test_image.jpg",
    ]

    sample_path = None
    for p in sample_paths:
        if p.exists():
            sample_path = p
            break

    if sample_path is None:
        print("\nNo sample image found. Skipping real image test.")
        return True

    print(f"\nTesting with real image: {sample_path}")

    img = cv2.imread(str(sample_path))
    if img is None:
        print(f"Failed to load image: {sample_path}")
        return False

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    depth = np.full((H, W), 2.0, dtype=np.float32)
    mask = np.ones((H, W), dtype=np.uint8)

    camera = CameraParams(
        fx=1000.0,
        fy=1000.0,
        cx=(W - 1) / 2.0,
        cy=(H - 1) / 2.0,
        width=W,
        height=H,
    )

    config = GaussianConfig(sh_degree=0, opacity_init=0.9)

    trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
    trainer.initialize_gaussians()

    rendered_rgb, _, _ = trainer.render_view()

    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    rgb_np = rendered_rgb.detach().cpu().numpy()
    rgb_np = np.clip(rgb_np, 0, 1)
    bgr_np = (rgb_np[:, :, ::-1] * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "real_image_render.png"), bgr_np)

    return True


if __name__ == "__main__":
    success1 = test_render_view()
    success2 = test_render_with_real_image()

    if success1 and success2:
        print("\n\nAll tests passed!")
    else:
        print("\n\nSome tests failed!")
        sys.exit(1)
