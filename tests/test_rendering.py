"""
Test script for Gaussian Splatting rendering pipeline.

Tests that we can:
1. Initialize Gaussians from synthetic data
2. Render a front view
3. Save the rendered image
"""

import sys
from pathlib import Path
import importlib.util

import numpy as np
import cv2

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def _import_gaussian_trainer():
    """Import gaussian_trainer directly to avoid package dependencies."""
    import types

    # Import gaussian_utils first
    spec_utils = importlib.util.spec_from_file_location(
        "gaussian_utils",
        Path(__file__).parent.parent
        / "src"
        / "human3d"
        / "reconstruct"
        / "gaussian_utils.py",
    )
    gaussian_utils = importlib.util.module_from_spec(spec_utils)
    spec_utils.loader.exec_module(gaussian_utils)

    # Set up module hierarchy
    if "human3d" not in sys.modules:
        sys.modules["human3d"] = types.ModuleType("human3d")
    if "human3d.reconstruct" not in sys.modules:
        sys.modules["human3d.reconstruct"] = types.ModuleType("human3d.reconstruct")
        sys.modules["human3d"].reconstruct = sys.modules["human3d.reconstruct"]

    sys.modules["human3d.reconstruct.gaussian_utils"] = gaussian_utils
    sys.modules["human3d.reconstruct"].gaussian_utils = gaussian_utils

    # Import gaussian_trainer
    spec_trainer = importlib.util.spec_from_file_location(
        "human3d.reconstruct.gaussian_trainer",
        Path(__file__).parent.parent
        / "src"
        / "human3d"
        / "reconstruct"
        / "gaussian_trainer.py",
    )
    gaussian_trainer = importlib.util.module_from_spec(spec_trainer)
    sys.modules["human3d.reconstruct.gaussian_trainer"] = gaussian_trainer
    spec_trainer.loader.exec_module(gaussian_trainer)

    return gaussian_trainer


def create_test_image_with_depth():
    """Create a synthetic test image with depth and mask."""
    H, W = 256, 256

    # Create a colorful test image with a gradient
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Create color blocks
    rgb[: H // 2, : W // 2] = [255, 0, 0]  # Red top-left
    rgb[: H // 2, W // 2 :] = [0, 255, 0]  # Green top-right
    rgb[H // 2 :, : W // 2] = [0, 0, 255]  # Blue bottom-left
    rgb[H // 2 :, W // 2 :] = [255, 255, 0]  # Yellow bottom-right

    # Add some gradient for more variation
    for y in range(H):
        for x in range(W):
            factor = 0.5 + 0.5 * (x / W)
            rgb[y, x] = (rgb[y, x] * factor).astype(np.uint8)

    # Create depth map (plane at z=2.0 with slight variation)
    depth = np.full((H, W), 2.0, dtype=np.float32)
    # Add dome shape to make it more interesting
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    depth = depth - (0.5 * np.exp(-(dist**2) / (2 * 50**2)))

    # Create circular mask
    radius = min(H, W) // 3
    mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).astype(np.uint8)

    return rgb, depth, mask


def test_render_view():
    """Test rendering from initialized Gaussians."""
    print("=" * 60)
    print("Testing Gaussian Splatting Rendering")
    print("=" * 60)

    # Import trainer
    gt = _import_gaussian_trainer()

    # Create test data
    print("\n1. Creating synthetic test data...")
    rgb, depth, mask = create_test_image_with_depth()
    H, W = rgb.shape[:2]

    print(f"   Image size: {W}x{H}")
    print(f"   Masked pixels: {mask.sum()}")

    # Create camera params
    camera = gt.CameraParams(
        fx=500.0,
        fy=500.0,
        cx=(W - 1) / 2.0,
        cy=(H - 1) / 2.0,
        width=W,
        height=H,
    )

    # Create config
    config = gt.GaussianConfig(
        sh_degree=0,  # Use degree 0 for faster testing
        opacity_init=0.9,
        position_noise=0.0,  # No noise for cleaner test
    )

    # Initialize trainer
    print("\n2. Initializing Gaussian trainer...")
    trainer = gt.GaussianTrainer(
        rgb=rgb,
        depth=depth,
        mask=mask,
        camera_params=camera,
        config=config,
        device="cpu",  # Force CPU for testing
    )

    # Initialize Gaussians
    print("3. Initializing Gaussians from depth map...")
    num_gaussians = trainer.initialize_gaussians()
    print(f"   Initialized {num_gaussians} Gaussians")

    # Check parameter shapes
    print("\n4. Checking parameter shapes...")
    print(f"   means: {trainer.means.shape}")
    print(f"   scales: {trainer.scales.shape}")
    print(f"   rotations: {trainer.rotations.shape}")
    print(f"   sh_coeffs: {trainer.sh_coeffs.shape}")
    print(f"   opacities: {trainer.opacities.shape}")

    # Render front view
    print("\n5. Rendering front view...")
    rendered_rgb, rendered_depth, rendered_alpha = trainer.render_view()

    print(f"   Rendered RGB shape: {rendered_rgb.shape}")
    print(f"   Rendered depth shape: {rendered_depth.shape}")
    print(f"   Rendered alpha shape: {rendered_alpha.shape}")
    print(f"   RGB range: [{rendered_rgb.min():.3f}, {rendered_rgb.max():.3f}]")
    print(f"   Alpha range: [{rendered_alpha.min():.3f}, {rendered_alpha.max():.3f}]")

    # Save outputs
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("\n6. Saving rendered images...")

    # Save rendered RGB
    rgb_np = rendered_rgb.detach().cpu().numpy()
    rgb_np = np.clip(rgb_np, 0, 1)
    bgr_np = (rgb_np[:, :, ::-1] * 255).astype(np.uint8)
    render_path = output_dir / "test_render.png"
    cv2.imwrite(str(render_path), bgr_np)
    print(f"   Saved: {render_path}")

    # Save original RGB for comparison
    original_path = output_dir / "test_original.png"
    cv2.imwrite(str(original_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"   Saved: {original_path}")

    # Save mask
    mask_path = output_dir / "test_mask.png"
    cv2.imwrite(str(mask_path), mask * 255)
    print(f"   Saved: {mask_path}")

    # Save rendered alpha
    alpha_np = rendered_alpha.detach().cpu().numpy()
    alpha_np = (alpha_np * 255).astype(np.uint8)
    alpha_path = output_dir / "test_render_alpha.png"
    cv2.imwrite(str(alpha_path), alpha_np)
    print(f"   Saved: {alpha_path}")

    # Save rendered depth (normalized for visualization)
    depth_np = rendered_depth.detach().cpu().numpy()
    depth_valid = depth_np[depth_np < float("inf")]
    if len(depth_valid) > 0:
        depth_vis = (depth_np - depth_valid.min()) / (
            depth_valid.max() - depth_valid.min() + 1e-6
        )
        depth_vis = np.clip(depth_vis, 0, 1)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_path = output_dir / "test_render_depth.png"
        cv2.imwrite(str(depth_path), depth_vis)
        print(f"   Saved: {depth_path}")

    print("\n" + "=" * 60)
    print("Rendering test completed!")
    print(f"Check outputs in: {output_dir}")
    print("=" * 60)

    return True


def test_render_with_real_image():
    """Test rendering with a real image if available."""
    # Check if there's a sample image
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
        print("To test with a real image, place an image at:")
        for p in sample_paths:
            print(f"  - {p}")
        return True

    print(f"\n{'=' * 60}")
    print(f"Testing with real image: {sample_path}")
    print("=" * 60)

    # Load image
    img = cv2.imread(str(sample_path))
    if img is None:
        print(f"Failed to load image: {sample_path}")
        return False

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    # Create synthetic depth (since we don't have real depth)
    depth = np.full((H, W), 2.0, dtype=np.float32)

    # Create full mask
    mask = np.ones((H, W), dtype=np.uint8)

    # Import and run
    gt = _import_gaussian_trainer()

    camera = gt.CameraParams(
        fx=1000.0,
        fy=1000.0,
        cx=(W - 1) / 2.0,
        cy=(H - 1) / 2.0,
        width=W,
        height=H,
    )

    config = gt.GaussianConfig(sh_degree=0, opacity_init=0.9)

    trainer = gt.GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
    trainer.initialize_gaussians()

    rendered_rgb, _, _ = trainer.render_view()

    # Save
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    rgb_np = rendered_rgb.detach().cpu().numpy()
    rgb_np = np.clip(rgb_np, 0, 1)
    bgr_np = (rgb_np[:, :, ::-1] * 255).astype(np.uint8)

    output_path = output_dir / "real_image_render.png"
    cv2.imwrite(str(output_path), bgr_np)
    print(f"Saved: {output_path}")

    return True


if __name__ == "__main__":
    success1 = test_render_view()
    success2 = test_render_with_real_image()

    if success1 and success2:
        print("\n\nAll tests passed!")
    else:
        print("\n\nSome tests failed!")
        sys.exit(1)
