#!/usr/bin/env python3
"""
Train Gaussian Splatting from RGB-D input.

This script demonstrates the full Gaussian Splatting training pipeline:
1. Load RGB image, depth map, and segmentation mask
2. Initialize Gaussians from depth
3. Optimize Gaussian parameters
4. Export to PLY format

Usage:
    python scripts/train_gaussians.py --input data/person.jpg --output outputs/

    # With custom iterations
    python scripts/train_gaussians.py --input data/person.jpg --iterations 3000

    # Quick preview
    python scripts/train_gaussians.py --input data/person.jpg --quick

Example:
    # Full training
    python scripts/train_gaussians.py \\
        --input data/sample.jpg \\
        --output outputs/sample \\
        --iterations 1000 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to path for development
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train 3D Gaussian Splatting from RGB-D input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python train_gaussians.py --input image.jpg

    # Custom output directory
    python train_gaussians.py --input image.jpg --output outputs/my_run

    # Quick preview (500 iterations)
    python train_gaussians.py --input image.jpg --quick

    # High quality (5000 iterations)
    python train_gaussians.py --input image.jpg --iterations 5000

View results:
    Upload the output .ply file to https://antimatter15.com/splat/
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input RGB image (jpg/png)",
    )

    parser.add_argument(
        "--depth",
        "-d",
        type=str,
        default=None,
        help="Path to depth map (.npy). If not provided, will be estimated using MiDaS",
    )

    parser.add_argument(
        "--mask",
        "-m",
        type=str,
        default=None,
        help="Path to segmentation mask (.npy or image). If not provided, full image is used",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs",
        help="Output directory (default: outputs/)",
    )

    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=1000,
        help="Number of training iterations (default: 1000)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick preview mode (500 iterations, no densification)",
    )

    parser.add_argument(
        "--sh-degree",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Spherical harmonics degree (0=view-independent, 3=full, default: 0)",
    )

    parser.add_argument(
        "--focal-length",
        type=float,
        default=500.0,
        help="Camera focal length in pixels (default: 500)",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save intermediate renders every N iterations (default: 500, 0=disabled)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    """Load RGB image from file."""
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_depth(path: str) -> np.ndarray:
    """Load depth map from .npy file."""
    depth = np.load(path)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D, got shape {depth.shape}")
    return depth.astype(np.float32)


def load_mask(path: str) -> np.ndarray:
    """Load segmentation mask."""
    if path.endswith(".npy"):
        mask = np.load(path)
    else:
        import cv2

        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {path}")
        mask = (mask > 127).astype(np.uint8)
    return mask


def estimate_depth_midas(rgb: np.ndarray) -> np.ndarray:
    """Estimate depth using MiDaS."""
    print("Estimating depth with MiDaS...")
    import torch

    # Load MiDaS
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform

    # Prepare input
    input_batch = transform(rgb).to(device)

    # Predict
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
    return depth.astype(np.float32)


def main():
    """Main training function."""
    args = parse_args()

    # Import here to avoid slow startup
    import torch
    from human3d.reconstruct.gaussian_trainer import (
        GaussianTrainer,
        GaussianConfig,
        CameraParams,
    )

    print("=" * 60)
    print("  3D Gaussian Splatting Training")
    print("=" * 60)

    # 1. Load input data
    print("\n[1/4] Loading input data...")
    rgb = load_image(args.input)
    h, w = rgb.shape[:2]
    print(f"  Image: {args.input} ({w}x{h})")

    # Load or estimate depth
    if args.depth:
        depth = load_depth(args.depth)
        print(f"  Depth: {args.depth}")
    else:
        depth = estimate_depth_midas(rgb)
        print("  Depth: estimated with MiDaS")

    # Load or create mask
    if args.mask:
        mask = load_mask(args.mask)
        print(f"  Mask: {args.mask} ({mask.sum()} pixels)")
    else:
        mask = np.ones((h, w), dtype=np.uint8)
        print(f"  Mask: full image ({mask.sum()} pixels)")

    # 2. Configure training
    print("\n[2/4] Configuring training...")

    # Quick mode adjustments
    if args.quick:
        iterations = 500
        densify_from = 10000  # Disable densification
        print("  Mode: Quick preview")
    else:
        iterations = args.iterations
        densify_from = 500
        print("  Mode: Full training")

    print(f"  Iterations: {iterations}")
    print(f"  SH degree: {args.sh_degree}")
    print(f"  Device: {args.device}")

    # Create camera params
    camera = CameraParams(
        fx=args.focal_length,
        fy=args.focal_length,
        cx=(w - 1) / 2.0,
        cy=(h - 1) / 2.0,
        width=w,
        height=h,
    )

    # Create config
    config = GaussianConfig(
        sh_degree=args.sh_degree,
        num_iterations=iterations,
        densify_from_iter=densify_from,
    )

    # 3. Train Gaussians
    print("\n[3/4] Training Gaussians...")
    start_time = time.time()

    trainer = GaussianTrainer(rgb, depth, mask, camera, config, device=args.device)
    num_gaussians = trainer.initialize_gaussians()
    print(f"  Initialized: {num_gaussians} Gaussians")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run optimization
    history = trainer.optimize(
        num_iterations=iterations,
        log_every=100 if not args.quick else 50,
        save_every=args.save_every,
        output_dir=str(output_dir),
    )

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Final Gaussians: {trainer.num_gaussians}")

    # 4. Export results
    print("\n[4/4] Exporting results...")

    # Save PLY
    ply_path = output_dir / "gaussians.ply"
    trainer.export_ply(str(ply_path))

    # Save final render
    rendered_rgb, _, _ = trainer.render_view()
    rendered_np = (rendered_rgb.detach().cpu().numpy() * 255).astype(np.uint8)

    import cv2

    render_path = output_dir / "final_render.png"
    cv2.imwrite(str(render_path), cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR))
    print(f"  Saved render: {render_path}")

    # Save comparison
    comparison = np.concatenate([rgb, rendered_np], axis=1)
    comparison_path = output_dir / "comparison.png"
    cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"  Saved comparison: {comparison_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\n  Output directory: {output_dir}")
    print(f"  PLY file: {ply_path}")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Gaussians: {trainer.num_gaussians}")
    print("\n  View your result:")
    print("    1. Go to https://antimatter15.com/splat/")
    print(f"    2. Drag and drop: {ply_path}")
    print()


if __name__ == "__main__":
    main()
